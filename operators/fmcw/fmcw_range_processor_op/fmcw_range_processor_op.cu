// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"

#include <stdexcept>

#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
#include "radar_common/sdk/override_param.hpp"

namespace holoscan::radar {

template <typename T>
FMCWRangeProcessorOp<T>::~FMCWRangeProcessorOp() = default;

template <typename T>
void FMCWRangeProcessorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("adc_cube");
  spec.output<holoscan::TensorMap>("range_profiles");
  // Optional axis output: emitted only when a waveform descriptor is
  // available; kNone so an unconnected port never gates scheduling.
  spec.output<holoscan::TensorMap>("range_bins")
      .condition(ConditionType::kNone);

  spec.param(range_window_, "range_window", "Range window",
             "WindowSpec applied along fast-time before the range FFT "
             "(coherent-gain normalized). Default: none (rectangular).",
             WindowSpec{});
  add_override_port<WindowSpec>(spec, "range_window");

  spec.param(apply_dechirp_, "apply_dechirp", "Apply software dechirp",
             "Conjugate-multiply the input against the reference chirp "
             "along fast-time before windowing/FFT. Requires a waveform "
             "and complex-IF input when true.",
             false);
  add_override_port<bool>(spec, "apply_dechirp");

  spec.param(waveform_, "waveform", "FMCW waveform descriptor",
             "Shared chirp descriptor; source for the dechirp reference "
             "and the range_bins axis. Required when apply_dechirp is true "
             "and/or range_bins is used.",
             ParameterFlag::kOptional);
  add_override_port<FMCWWaveformDescriptor>(spec, "waveform");

  spec.param(fft_norm_, "fft_norm", "FFT normalization",
             "FFT normalization convention {backward, ortho, forward}; "
             "backward keeps coherent-integration gain visible.",
             FftNorm::kBackward);
  add_override_port<FftNorm>(spec, "fft_norm");

  spec.param(propagation_speed_, "propagation_speed", "Propagation speed",
             "Wave speed (m/s) for the beat-frequency to range conversion "
             "(range_bins only).",
             kDefaultPropagationSpeed);
  add_override_port<double>(spec, "propagation_speed");

  spec.param(nfft_, "nfft", "Fast-time FFT length",
             "FFT length >= samples_per_chirp; larger zero-pads for a "
             "finer interpolated range grid. Default: samples_per_chirp.",
             ParameterFlag::kOptional);
  add_override_port<int64_t>(spec, "nfft");
}

template <typename T>
void FMCWRangeProcessorOp<T>::compute(InputContext& op_input,
                                      OutputContext& op_output,
                                      ExecutionContext& context) {
  // Implemented in a subsequent commit.
  throw std::runtime_error("FMCWRangeProcessorOp: not implemented yet");
}

template class FMCWRangeProcessorOp<float>;
template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
