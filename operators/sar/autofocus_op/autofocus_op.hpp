// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"

// Phase gradient autofocus on a conjugate-grid image (OPERATOR_API.md →
// AutofocusOp). Estimates a residual phase along the azimuth-conjugate axis
// and applies it in-image; the per-pulse estimate is published separately so
// it can correct a formation on any geometry through PhaseCorrectionOp (D20).
//
// In:  image            complex64/complex128, (num_v, num_u). The formed SLC.
// Out: image_out        complex64/complex128, same shape. Corrected image.
// Out: phase_correction float32/float64, (num_pulses). Optional. Indexed by
//                       pulse, not azimuth bin; length is the named azimuth
//                       axis.
// Out: sharpness        float32/float64, (num_iterations + 1). Optional. Focus
//                       metric before each iteration and after the last.
// Out: termination      int32, (1). Optional. Why the loop stopped
//                       (AutofocusTermination). Provenance, not quality.
// Out: iterations       int32, (1). Optional. Iterations whose phase is in
//                       phase_correction; 0 when the estimate was refused.
//
// algorithm is Static (default pga, sole value). azimuth_axis is Static and
// required (no default) — PGA is meaningless on the wrong axis and a mismatch
// is not dimensionally checkable, so it is never inferred. The four remaining
// parameters are [live]: each has a same-named input port joined to the image
// trigger with Exact{} + SafeDefault{}, declared unconditionally so that
// wiring the port is what enables the override. Precision is the template
// instantiation (T = float → complex64/float32, double → complex128/float64).
namespace holoscan::radar {

struct AutofocusParams {
  AutofocusAlgorithm algorithm{AutofocusAlgorithm::kPga};
  AutofocusAzimuthAxis azimuth_axis{AutofocusAzimuthAxis::kUnspecified};
  std::int64_t num_iterations{8};
  double window_fraction{0.5};
  std::int64_t min_scatterers{16};
  double max_correction{std::numeric_limits<double>::infinity()};
};

HOLOSCAN_PARAM_SCHEMA(
    AutofocusParams, holoscan::param_schema("com.nvidia.holoscan.radar.AutofocusParams", 1),
    holoscan::field<&AutofocusParams::algorithm>("algorithm")
        .description("Phase-gradient autofocus. Static; pga is the sole value."),
    holoscan::field<&AutofocusParams::azimuth_axis>("azimuth_axis")
        .description("Which trailing image axis is slow-time-conjugate (u or v). "
                     "Static and required; naming it asserts the image is conjugate."),
    holoscan::field<&AutofocusParams::num_iterations>("num_iterations")
        .description("Refinement iterations, an upper bound rather than a count: the stage stops "
                     "early once an iteration fails to improve sharpness. Live. Default 8."),
    holoscan::field<&AutofocusParams::window_fraction>("window_fraction")
        .description("Fraction of the azimuth extent in the first window; shrinks "
                     "geometrically across iterations. Live. Default 0.5."),
    holoscan::field<&AutofocusParams::min_scatterers>("min_scatterers")
        .description("Minimum strong scatterers required; below it the image passes "
                     "through and phase_correction is zero. Live. Default 16."),
    holoscan::field<&AutofocusParams::max_correction>("max_correction")
        .description("Peak-to-peak ceiling in radians on the estimate; uncapped when "
                     "omitted. Live."));

namespace detail {
template <typename T>
struct AutofocusState;
}  // namespace detail

// Ceiling on [live] num_iterations, and so on the sharpness pool.
//
// HSDK 5 sizes a Tensor output pool at graph build, but num_iterations is
// [live] and sharpness is (num_iterations + 1) long, so an override could
// outgrow a pool sized from the build-time Param. Rather than make the
// parameter static — which the contract does not — the pool is sized to this
// ceiling and an override beyond it is refused. 256 slots of a real scalar is
// a rounding error next to the image, so the generous bound costs nothing.
inline constexpr std::int64_t kMaxAutofocusIterations = 255;
inline constexpr std::int64_t kMaxAutofocusSharpnessSlots = kMaxAutofocusIterations + 1;

template <typename T>
class AutofocusOp : public holoscan::Operator<AutofocusParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "AutofocusOp supports float and double precision");

 public:
  explicit AutofocusOp(const MaxTensorExtents& extents);
  ~AutofocusOp() override;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> image{"image"};
  Input<std::int64_t> num_iterations{"num_iterations"};
  Input<double> window_fraction{"window_fraction"};
  Input<std::int64_t> min_scatterers{"min_scatterers"};
  Input<double> max_correction{"max_correction"};

  Output<Tensor> image_out{"image_out"};
  Output<Tensor> phase_correction{"phase_correction"};
  Output<Tensor> sharpness{"sharpness"};
  Output<Tensor> termination{"termination"};
  Output<Tensor> iterations{"iterations"};

 private:
  MaxTensorExtents extents_;
  std::int64_t max_azimuth_{0};
  std::size_t max_image_byte_span_{0U};
  std::size_t max_phase_byte_span_{0U};
  std::size_t max_sharpness_byte_span_{0U};
  std::size_t max_status_byte_span_{0U};

  std::shared_ptr<detail::AutofocusState<T>> state_;

  // One PGA run produces the image, the phase vector and the sharpness trace
  // together, and writes all three to device memory. The image goes straight
  // into its pool loan, but the other two are optional ports that may not be
  // wired at all, so the run cannot target their loans directly. It targets
  // these instead — allocated once at start(), at the same pool bounds — and
  // each wired port then takes a device-to-device copy. Both vectors are
  // scalars-per-pulse, so the extra copy is immaterial.
  T* phase_scratch_{nullptr};
  T* sharpness_scratch_{nullptr};

  // Stream-ordered lifetime of this tick's input.
  Tensor retained_input_;

  // Host staging for the two int32 status ports. Members rather than locals
  // because the host-to-device copy is asynchronous on the compute stream and
  // must not read a dead stack slot; one each, so the two emits cannot race.
  std::int32_t termination_staging_{0};
  std::int32_t iterations_staging_{0};

  // phase_correction, sharpness, termination and iterations are optional
  // outputs; an application may leave any unwired, and an emit to an unconsumed
  // port must not take the corrected image down with it.
  LatchedOutput phase_correction_latch_;
  LatchedOutput sharpness_latch_;
  LatchedOutput termination_latch_;
  LatchedOutput iterations_latch_;
};

extern template class AutofocusOp<float>;
extern template class AutofocusOp<double>;

}  // namespace holoscan::radar
