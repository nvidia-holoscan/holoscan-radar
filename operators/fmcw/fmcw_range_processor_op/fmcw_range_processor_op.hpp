// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/math/constants.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct RangeStageState;  // kernel-side caches; defined in range_stage.hpp
}  // namespace detail

// FMCW range compression: windowed FFT along fast-time. This header is
// the contract reference for the operator's ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  adc_cube       — complex64/complex128 (IQ IF) or float32/float64
//                       (real IF), (..., num_chirps, samples_per_chirp).
// Out: range_profiles — complex64/complex128,
//                       (..., num_chirps, num_range_bins).
// Out: range_bins     — optional, float32/float64, (num_range_bins,);
//                       emitted only when a waveform descriptor is set.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
//
// [EA HSDK5 ONLY] range_window and waveform are [live] in OPERATOR_API.md
// but stay Params-only here: WindowSpec and FMCWWaveformDescriptor own
// vectors / optionals (HSDK5_PORT.md §9). POD scalars (apply_dechirp,
// fft_norm, propagation_speed, nfft) have sticky Input overrides.

struct FMCWRangeProcessorParams {
  WindowSpec range_window{};
  bool apply_dechirp{false};
  // Default-constructed = none/absent; is_valid() → false.
  FMCWWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  double propagation_speed{kDefaultPropagationSpeed};
  // 0 = exact (no zero-padding); ≥ samples_per_chirp when non-zero.
  int64_t nfft{0};
};

HOLOSCAN_PARAM_SCHEMA(
    FMCWRangeProcessorParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.FMCWRangeProcessorParams", 1),
    holoscan::field<&FMCWRangeProcessorParams::range_window>("range_window")
        .description(
            "WindowSpec applied along fast-time before the range FFT "
            "(coherent-gain normalized). Default: none (rectangular)."),
    holoscan::field<&FMCWRangeProcessorParams::apply_dechirp>("apply_dechirp")
        .description(
            "Conjugate-multiply the input against the reference chirp "
            "along fast-time before windowing/FFT. Requires a waveform "
            "and complex-IF input when true."),
    holoscan::field<&FMCWRangeProcessorParams::waveform>("waveform")
        .description(
            "FMCW chirp descriptor; source for the dechirp reference "
            "and the range_bins axis. Default-constructed = none. "
            "[live] in API but Params-only on EA (HSDK5_PORT §9)."),
    holoscan::field<&FMCWRangeProcessorParams::fft_norm>("fft_norm")
        .description(
            "FFT normalization convention {backward, ortho, forward}; "
            "backward keeps coherent-integration gain visible."),
    holoscan::field<&FMCWRangeProcessorParams::propagation_speed>(
        "propagation_speed")
        .description(
            "Wave speed (m/s) for beat-frequency to range conversion "
            "(range_bins only)."),
    holoscan::field<&FMCWRangeProcessorParams::nfft>("nfft")
        .description(
            "Fast-time FFT length >= samples_per_chirp; larger zero-pads "
            "for a finer interpolated range grid. 0 = samples_per_chirp."));

static_assert(
    holoscan::param_schema_availability<FMCWRangeProcessorParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in "
    "FMCWRangeProcessorParams's namespace");

inline constexpr std::uint32_t kFMCWRangeProcessorMaxRank = 8U;
// 256 MiB
inline constexpr std::size_t kFMCWRangeProcessorMaxByteSpan = 256ULL << 20;

template <typename T>
class FMCWRangeProcessorOp
    : public holoscan::Operator<FMCWRangeProcessorParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWRangeProcessorOp supports float and double precision");

 public:
  FMCWRangeProcessorOp() = default;
  ~FMCWRangeProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> adc_cube{"adc_cube"};
  Output<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> range_bins{"range_bins"};

  // Sticky [live] override ports for POD scalars (OPERATOR_API.md).
  // [EA HSDK5 ONLY] range_window and waveform are Params-only (§9).
  Input<bool> apply_dechirp_in{"apply_dechirp"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<double> propagation_speed_in{"propagation_speed"};
  Input<int64_t> nfft_in{"nfft"};

 private:
  Tensor retained_input_;

  // Sticky [live] state for POD scalars.
  std::optional<bool> sticky_apply_dechirp_;
  std::optional<FftNorm> sticky_fft_norm_;
  std::optional<double> sticky_propagation_speed_;
  std::optional<int64_t> sticky_nfft_;

  // Kernel-side caches (window coefficients, dechirp reference, seen FFT
  // lengths); allocated on first compute().
  std::shared_ptr<detail::RangeStageState<T>> state_;
};

// Explicit instantiations live in fmcw_range_processor_op.cu.
extern template class FMCWRangeProcessorOp<float>;
extern template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
