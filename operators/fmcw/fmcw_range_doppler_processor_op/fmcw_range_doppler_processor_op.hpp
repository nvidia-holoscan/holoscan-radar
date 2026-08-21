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

namespace holoscan::radar {

namespace detail {
template <typename T>
struct RangeStageState;    // defined in range_stage.hpp
template <typename T>
struct DopplerStageState;  // defined in doppler_stage.hpp
}  // namespace detail

// Fused range + optional clutter + Doppler processing in one graph node.
// This header is the contract reference for the operator's ports, shapes,
// dtypes, and parameters. Drop-in replacement for the split
// FMCWRangeProcessorOp -> [ClutterFilterOp] -> FMCWDopplerProcessorOp chain
// where the intermediate range profiles are not needed; produces identical
// rd_maps (within tolerance) by calling the same stage functions.
//
// In:  adc_cube      — complex64/complex128 (IQ IF) or float32/float64
//                      (real IF), (..., num_chirps, samples_per_chirp).
// Out: rd_maps       — complex64/complex128,
//                      (..., num_doppler, num_range_bins).
// Out: range_bins    — optional, float32/float64, (num_range_bins,).
// Out: velocity_bins — optional, float32/float64, (num_doppler,).
//
// Precision is fixed at compile time by the template instantiation.
//
// [EA HSDK5 ONLY] range_window, doppler_window, and waveform are [live] in
// OPERATOR_API.md but stay Params-only here: WindowSpec and
// FMCWWaveformDescriptor own vectors / optionals (HSDK5_PORT.md §9). POD
// scalars (apply_dechirp, clutter_filter, fft_norm, center_zero_doppler,
// nfft_range, nfft_doppler) have sticky Input overrides.

struct FMCWRangeDopplerProcessorParams {
  WindowSpec range_window{};
  WindowSpec doppler_window{};
  bool apply_dechirp{false};
  ClutterFilterMethod clutter_filter{ClutterFilterMethod::kNone};
  // Default-constructed = none/absent; is_valid() -> false.
  FMCWWaveformDescriptor waveform{};
  FftNorm fft_norm{FftNorm::kBackward};
  bool center_zero_doppler{true};
  double propagation_speed{kDefaultPropagationSpeed};
  // 0 = exact (no zero-padding); >= samples_per_chirp when non-zero.
  int64_t nfft_range{0};
  // 0 = exact (no zero-padding); >= post-clutter chirp count when non-zero.
  int64_t nfft_doppler{0};
  bool expect_apply_dechirp{false};
  bool expect_clutter_filter{false};
  bool expect_fft_norm{false};
  bool expect_center_zero_doppler{false};
  bool expect_nfft_range{false};
  bool expect_nfft_doppler{false};
  bool sticky_apply_dechirp{true};
  bool sticky_clutter_filter{true};
  bool sticky_fft_norm{true};
  bool sticky_center_zero_doppler{true};
  bool sticky_nfft_range{true};
  bool sticky_nfft_doppler{true};
};

HOLOSCAN_PARAM_SCHEMA(
    FMCWRangeDopplerProcessorParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.FMCWRangeDopplerProcessorParams", 1),
    holoscan::field<&FMCWRangeDopplerProcessorParams::range_window>(
        "range_window")
        .description(
            "WindowSpec applied along fast-time before the range FFT "
            "(coherent-gain normalized). Default: none (rectangular)."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::doppler_window>(
        "doppler_window")
        .description(
            "WindowSpec applied along the chirp (slow-time) axis before "
            "the Doppler FFT (coherent-gain normalized). Default: none."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::apply_dechirp>(
        "apply_dechirp")
        .description(
            "Software dechirp before the range FFT. Requires a waveform "
            "and complex-IF input when true."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::clutter_filter>(
        "clutter_filter")
        .description(
            "Optional inline clutter stage between the range and Doppler "
            "stages {none, mean_subtraction, single_delay, double_delay}; "
            "same semantics as ClutterFilterOp's method. The delay "
            "cancellers reduce num_doppler."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::waveform>("waveform")
        .description(
            "Dechirp reference and the range_bins / velocity_bins axes. "
            "Required for software dechirp and/or those outputs. "
            "[live] in API but Params-only on EA (HSDK5_PORT §9)."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::fft_norm>("fft_norm")
        .description(
            "One normalization convention {backward, ortho, forward} for "
            "both transforms."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::center_zero_doppler>(
        "center_zero_doppler")
        .description(
            "Center zero-Doppler on the Doppler axis; else DC at bin 0."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::propagation_speed>(
        "propagation_speed")
        .description(
            "Wave speed (m/s) for the range_bins output."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::nfft_range>(
        "nfft_range")
        .description(
            "Fast-time FFT length >= samples_per_chirp; larger zero-pads "
            "for a finer interpolated range grid. 0 = samples_per_chirp."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::nfft_doppler>(
        "nfft_doppler")
        .description(
            "Slow-time FFT length >= the post-clutter chirp count for the "
            "Doppler stage. 0 = the post-clutter chirp count."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::expect_apply_dechirp>(
        "expect_apply_dechirp")
        .description(
            "Set when the application wires the apply_dechirp override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::expect_clutter_filter>(
        "expect_clutter_filter")
        .description(
            "Set when the application wires the clutter_filter override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::expect_fft_norm>(
        "expect_fft_norm")
        .description(
            "Set when the application wires the fft_norm override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::
                       expect_center_zero_doppler>(
        "expect_center_zero_doppler")
        .description(
            "Set when the application wires the center_zero_doppler "
            "override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::expect_nfft_range>(
        "expect_nfft_range")
        .description(
            "Set when the application wires the nfft_range override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::expect_nfft_doppler>(
        "expect_nfft_doppler")
        .description(
            "Set when the application wires the nfft_doppler override."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::sticky_apply_dechirp>(
        "sticky_apply_dechirp")
        .description(
            "Keep the last Exact apply_dechirp match on a miss. Ignored "
            "unless expect_apply_dechirp."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::sticky_clutter_filter>(
        "sticky_clutter_filter")
        .description(
            "Keep the last Exact clutter_filter match on a miss. Ignored "
            "unless expect_clutter_filter."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::sticky_fft_norm>(
        "sticky_fft_norm")
        .description(
            "Keep the last Exact fft_norm match on a miss. Ignored unless "
            "expect_fft_norm."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::
                       sticky_center_zero_doppler>(
        "sticky_center_zero_doppler")
        .description(
            "Keep the last Exact center_zero_doppler match on a miss. "
            "Ignored unless expect_center_zero_doppler."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::sticky_nfft_range>(
        "sticky_nfft_range")
        .description(
            "Keep the last Exact nfft_range match on a miss. Ignored "
            "unless expect_nfft_range."),
    holoscan::field<&FMCWRangeDopplerProcessorParams::sticky_nfft_doppler>(
        "sticky_nfft_doppler")
        .description(
            "Keep the last Exact nfft_doppler match on a miss. Ignored "
            "unless expect_nfft_doppler."));

static_assert(
    holoscan::param_schema_availability<FMCWRangeDopplerProcessorParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in "
    "FMCWRangeDopplerProcessorParams's namespace");

template <typename T>
class FMCWRangeDopplerProcessorOp
    : public holoscan::Operator<FMCWRangeDopplerProcessorParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWRangeDopplerProcessorOp supports float and double "
                "precision");

 public:
  explicit FMCWRangeDopplerProcessorOp(
      const MaxTensorExtents& extents);
  ~FMCWRangeDopplerProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> adc_cube{"adc_cube"};
  Output<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> range_bins{"range_bins"};
  Output<Tensor> velocity_bins{"velocity_bins"};

  // [live] Exact override ports for POD scalars, gated by expect_*.
  // [EA HSDK5 ONLY] range_window, doppler_window, waveform are
  // Params-only (§9).
  Input<bool> apply_dechirp_in{"apply_dechirp"};
  Input<ClutterFilterMethod> clutter_filter_in{"clutter_filter"};
  Input<FftNorm> fft_norm_in{"fft_norm"};
  Input<bool> center_zero_doppler_in{"center_zero_doppler"};
  Input<int64_t> nfft_range_in{"nfft_range"};
  Input<int64_t> nfft_doppler_in{"nfft_doppler"};

 private:
  Tensor retained_input_;

  // Sticky [live] state for POD scalars.
  std::optional<bool> sticky_apply_dechirp_;
  std::optional<ClutterFilterMethod> sticky_clutter_filter_;
  std::optional<FftNorm> sticky_fft_norm_;
  std::optional<bool> sticky_center_zero_doppler_;
  std::optional<int64_t> sticky_nfft_range_;
  std::optional<int64_t> sticky_nfft_doppler_;

  // Kernel-side caches for the two FFT stages.
  std::shared_ptr<detail::RangeStageState<T>> range_state_;
  std::shared_ptr<detail::DopplerStageState<T>> doppler_state_;

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
  std::size_t range_bins_byte_span_{0U};
  std::size_t velocity_bins_byte_span_{0U};
};

// Explicit instantiations live in fmcw_range_doppler_processor_op.cu.
extern template class FMCWRangeDopplerProcessorOp<float>;
extern template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
