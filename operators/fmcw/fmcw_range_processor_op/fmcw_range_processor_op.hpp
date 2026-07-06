// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
template <typename T>
class FMCWRangeProcessorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWRangeProcessorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FMCWRangeProcessorOp)

  FMCWRangeProcessorOp() = default;
  ~FMCWRangeProcessorOp() override;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Parameters per the locked contract; all overridable (each has a
  // same-named non-blocking override input port declared in setup()).
  Parameter<WindowSpec> range_window_;
  Parameter<bool> apply_dechirp_;
  Parameter<FMCWWaveformDescriptor> waveform_;
  Parameter<FftNorm> fft_norm_;
  Parameter<double> propagation_speed_;
  Parameter<int64_t> nfft_;

  // Effective override values as of the last tick (override-aware), used to
  // detect override-triggered rebuilds for the warn-once-and-cache rule.
  WindowSpec effective_window_{};
  bool effective_apply_dechirp_{false};
  std::optional<FMCWWaveformDescriptor> effective_waveform_{};
  FftNorm effective_fft_norm_{FftNorm::kBackward};
  double effective_propagation_speed_{kDefaultPropagationSpeed};
  std::optional<int64_t> effective_nfft_{};

  // Kernel-side caches (window coefficients, dechirp reference, seen FFT
  // lengths); allocated on first compute().
  std::shared_ptr<detail::RangeStageState<T>> state_;

  // Keeps the previous tick's input alive until this tick's kernels are
  // enqueued (stream-ordered lifetime guard).
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in fmcw_range_processor_op.cu.
extern template class FMCWRangeProcessorOp<float>;
extern template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
