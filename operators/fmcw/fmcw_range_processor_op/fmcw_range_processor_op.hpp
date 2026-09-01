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
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

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
//                       (real IF), (num_chirps, samples_per_chirp) or with
//                       named leading N_rx / (M_tx, N_rx).
// Out: range_profiles — complex64/complex128,
//                       same leading + (num_chirps, num_range_bins).
// Out: range_bins     — optional, float32/float64, (num_range_bins,);
//                       emitted only when a waveform descriptor is set.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
// Fat outs require MaxTensorExtents at graph.op() (ctor).
//
struct FMCWRangeProcessorParams {
  WindowSpec range_window{};
  bool apply_dechirp{false};
  // Default-constructed = none/absent; is_valid() → false.
  FMCWWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  double propagation_speed{kDefaultPropagationSpeed};
  // 0 = exact (no zero-padding); ≥ samples_per_chirp when non-zero.
  int64_t nfft{0};
  // [live] override ports need no wiring flag: an unwired one reads as a
  // miss (OPERATOR_API.md).
  bool sticky_range_window{true};
  bool sticky_apply_dechirp{true};
  bool sticky_waveform{true};
  bool sticky_fft_norm{true};
  bool sticky_nfft{true};
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
            "and the range_bins axis. Default-constructed = none."),
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
            "for a finer interpolated range grid. 0 = samples_per_chirp."),
    holoscan::field<&FMCWRangeProcessorParams::sticky_range_window>(
        "sticky_range_window")
        .description(
            "Keep the last Exact range_window match on a miss. False "
            "reverts to Params."),
    holoscan::field<&FMCWRangeProcessorParams::sticky_apply_dechirp>(
        "sticky_apply_dechirp")
        .description(
            "Keep the last Exact apply_dechirp match on a miss. False "
            "reverts to Params."),
    holoscan::field<&FMCWRangeProcessorParams::sticky_waveform>(
        "sticky_waveform")
        .description(
            "Keep the last Exact waveform match on a miss. False reverts "
            "to Params."),
    holoscan::field<&FMCWRangeProcessorParams::sticky_fft_norm>(
        "sticky_fft_norm")
        .description("Keep the last Exact fft_norm match on a miss."),
    holoscan::field<&FMCWRangeProcessorParams::sticky_nfft>("sticky_nfft")
        .description("Keep the last Exact nfft match on a miss."));

static_assert(
    holoscan::param_schema_availability<FMCWRangeProcessorParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in "
    "FMCWRangeProcessorParams's namespace");

template <typename T>
class FMCWRangeProcessorOp
    : public holoscan::Operator<FMCWRangeProcessorParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWRangeProcessorOp supports float and double precision");

 public:
  explicit FMCWRangeProcessorOp(const MaxTensorExtents& extents);
  ~FMCWRangeProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> adc_cube{"adc_cube"};
  Output<Tensor> range_profiles{"range_profiles"};
  Output<Tensor> range_bins{"range_bins"};

  // [live] Exact override ports (OPERATOR_API.md).
  Input<wire::WindowSpecT> range_window_in{"range_window"};
  Input<bool> apply_dechirp_in{"apply_dechirp"};
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<int64_t> nfft_in{"nfft"};

 private:
  Tensor retained_input_;

  // range_bins is optional/latched (OPERATOR_API.md).
  LatchedOutput range_bins_out_;

  // Last Exact-matched override (used when sticky_* is true).
  std::optional<WindowSpec> sticky_range_window_;
  std::optional<bool> sticky_apply_dechirp_;
  std::optional<FMCWWaveformDescriptor> sticky_waveform_;
  std::optional<FftNorm> sticky_fft_norm_;
  std::optional<int64_t> sticky_nfft_;

  // Kernel-side caches (window coefficients, dechirp reference, seen FFT
  // lengths); allocated on first compute().
  std::shared_ptr<detail::RangeStageState<T>> state_;

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
  std::size_t range_bins_byte_span_{0U};
};

// Explicit instantiations live in fmcw_range_processor_op.cu.
extern template class FMCWRangeProcessorOp<float>;
extern template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
