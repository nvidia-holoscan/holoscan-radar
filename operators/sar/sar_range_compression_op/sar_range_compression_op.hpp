// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct RangeCompressionState;  // caches; defined in range_compression_stage.hpp
}  // namespace detail

// SAR range compression: windowed, optionally zero-padded inverse transform
// along fast-time, taking phase history to range profiles. This header is the
// contract reference for the operator's ports, shapes, dtypes and parameters;
// module-wide conventions live in operators/common/README.md and the locked
// contract in docs/planning/SAR/OPERATOR_API.md.
//
// In:  phase_history  — complex64/complex128, (..., num_pulses, num_samples).
// Out: range_profiles — complex64/complex128,
//                       (..., num_pulses, num_range_bins), where
//                       num_range_bins = nfft (default num_samples).
// Out: range_bins     — optional, float32/float64, (num_range_bins,); emitted
//                       only when a waveform descriptor is available.
// Out: range_window   — optional, WindowSpec; the window actually applied,
//                       emitted on the first tick and whenever it changes.
//
// **The transform is the inverse one**, unlike FMCWRangeProcessorOp. Phase
// history is the frequency-domain form, so compression here runs from
// frequency to range; FMCW's dechirped input is already a time-domain beat
// signal. `fft_norm = backward` therefore applies the 1/N *here*, which is the
// same spelling that leaves the FMCW stage unscaled.
//
// **The range axis is centred.** A range-axis fftshift puts zero differential
// range at bin N/2, so bins below it are nearer than the reference point and
// bins above it are farther. This is a correctness requirement rather than a
// presentation choice: read as a linearly increasing axis, an unshifted
// transform sends every sample nearer than the reference to a negative bin
// index, where BackprojectionOp's bounds check discards it.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
template <typename T>
class SARRangeCompressionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SARRangeCompressionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SARRangeCompressionOp)

  SARRangeCompressionOp() = default;
  ~SARRangeCompressionOp() override;

  // Registers the argument setters for the custom parameter types
  // (radar_common/sdk/arg_setters.hpp) before the base class processes args.
  void initialize() override;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Parameters per the locked contract; all overridable (each has a
  // same-named non-blocking override input port declared in setup()).
  Parameter<WindowSpec> range_window_;
  Parameter<LFMWaveformDescriptor> waveform_;
  Parameter<FftNorm> fft_norm_;
  Parameter<int64_t> nfft_;
  Parameter<double> propagation_speed_;

  // Effective override values as of the last tick (sticky), used to detect
  // override-triggered rebuilds for the warn-once-and-cache rule.
  WindowSpec effective_window_{};
  std::optional<LFMWaveformDescriptor> effective_waveform_{};
  FftNorm effective_fft_norm_{FftNorm::kBackward};
  std::optional<int64_t> effective_nfft_{};
  double effective_propagation_speed_{kDefaultPropagationSpeed};

  // Whether the window on the descriptor output port is current. Set false at
  // start and by any window override, so the port emits once at the first tick
  // and then only on change.
  bool window_announced_{false};

  // Kernel-side caches (window coefficients, seen transform lengths, the
  // pre-shift scratch buffer); allocated on first compute().
  std::shared_ptr<detail::RangeCompressionState<T>> state_;

  // Keeps the previous tick's input alive until this tick's kernels are
  // enqueued (stream-ordered lifetime guard).
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in sar_range_compression_op.cu.
extern template class SARRangeCompressionOp<float>;
extern template class SARRangeCompressionOp<double>;

}  // namespace holoscan::radar
