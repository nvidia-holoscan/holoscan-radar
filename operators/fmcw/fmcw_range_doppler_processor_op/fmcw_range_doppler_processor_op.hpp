// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"

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
// FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp chain
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
template <typename T>
class FMCWRangeDopplerProcessorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWRangeDopplerProcessorOp supports float and double "
                "precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FMCWRangeDopplerProcessorOp)

  FMCWRangeDopplerProcessorOp() = default;
  ~FMCWRangeDopplerProcessorOp() override;

  // Registers the argument setters for the custom parameter types
  // (radar_common/sdk/arg_setters.hpp) before the base class processes args.
  void initialize() override;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Union of the split operators' parameters, same semantics; all overridable.
  Parameter<WindowSpec> range_window_;
  Parameter<WindowSpec> doppler_window_;
  Parameter<bool> apply_dechirp_;
  Parameter<ClutterFilterMethod> clutter_filter_;
  Parameter<FMCWWaveformDescriptor> waveform_;
  Parameter<FftNorm> fft_norm_;
  Parameter<bool> center_zero_doppler_;
  Parameter<double> propagation_speed_;
  Parameter<int64_t> nfft_range_;
  Parameter<int64_t> nfft_doppler_;

  // Effective override values as of the last tick (override-aware).
  WindowSpec effective_range_window_{};
  WindowSpec effective_doppler_window_{};
  bool effective_apply_dechirp_{false};
  ClutterFilterMethod effective_clutter_filter_{ClutterFilterMethod::kNone};
  std::optional<FMCWWaveformDescriptor> effective_waveform_{};
  FftNorm effective_fft_norm_{FftNorm::kBackward};
  bool effective_center_zero_doppler_{true};
  double effective_propagation_speed_{kDefaultPropagationSpeed};
  std::optional<int64_t> effective_nfft_range_{};
  std::optional<int64_t> effective_nfft_doppler_{};

  // Kernel-side caches for the two FFT stages.
  std::shared_ptr<detail::RangeStageState<T>> range_state_;
  std::shared_ptr<detail::DopplerStageState<T>> doppler_state_;

  // Stream-ordered lifetime guard for the previous tick's input.
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in fmcw_range_doppler_processor_op.cu.
extern template class FMCWRangeDopplerProcessorOp<float>;
extern template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
