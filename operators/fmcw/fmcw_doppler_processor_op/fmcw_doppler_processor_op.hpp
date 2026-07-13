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
struct DopplerStageState;  // kernel-side caches; defined in doppler_stage.hpp
}  // namespace detail

// FMCW Doppler compression: windowed FFT along slow-time. This header is
// the contract reference for the operator's ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  range_profiles — complex64/complex128,
//                       (..., num_chirps, num_range_bins).
// Out: rd_maps        — complex64/complex128,
//                       (..., num_doppler, num_range_bins);
//                       zero-Doppler centered by default.
// Out: velocity_bins  — optional, float32/float64, (num_doppler,);
//                       emitted only when a waveform descriptor is set.
//
// Precision is fixed at compile time by the template instantiation.
template <typename T>
class FMCWDopplerProcessorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FMCWDopplerProcessorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FMCWDopplerProcessorOp)

  FMCWDopplerProcessorOp() = default;
  ~FMCWDopplerProcessorOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Parameters per the locked contract; all overridable (each has a
  // same-named non-blocking override input port declared in setup()).
  Parameter<WindowSpec> doppler_window_;
  Parameter<bool> center_zero_doppler_;
  Parameter<FMCWWaveformDescriptor> waveform_;
  Parameter<FftNorm> fft_norm_;
  Parameter<int64_t> nfft_;

  // Effective override values as of the last tick (override-aware), used to
  // detect override-triggered rebuilds for the warn-once-and-cache rule.
  WindowSpec effective_window_{};
  bool effective_center_zero_doppler_{true};
  std::optional<FMCWWaveformDescriptor> effective_waveform_{};
  FftNorm effective_fft_norm_{FftNorm::kBackward};
  std::optional<int64_t> effective_nfft_{};

  // Kernel-side caches (window coefficients, seen FFT lengths); allocated
  // on first compute().
  std::shared_ptr<detail::DopplerStageState<T>> state_;

  // Stream-ordered lifetime guard for the previous tick's input.
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in fmcw_doppler_processor_op.cu.
extern template class FMCWDopplerProcessorOp<float>;
extern template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
