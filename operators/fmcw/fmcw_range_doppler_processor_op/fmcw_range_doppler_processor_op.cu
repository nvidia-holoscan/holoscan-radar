// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"

#include <stdexcept>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
#include "radar_common/sdk/override_param.hpp"

namespace holoscan::radar {

template <typename T>
FMCWRangeDopplerProcessorOp<T>::~FMCWRangeDopplerProcessorOp() = default;

template <typename T>
void FMCWRangeDopplerProcessorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("adc_cube");
  spec.output<holoscan::TensorMap>("rd_maps");
  // Optional axis outputs: emitted only when a waveform descriptor is
  // available; kNone so unconnected ports never gate scheduling.
  spec.output<holoscan::TensorMap>("range_bins")
      .condition(ConditionType::kNone);
  spec.output<holoscan::TensorMap>("velocity_bins")
      .condition(ConditionType::kNone);

  spec.param(range_window_, "range_window", "Range window",
             "WindowSpec applied along fast-time before the range FFT "
             "(coherent-gain normalized). Default: none (rectangular).",
             WindowSpec{});
  add_override_port<WindowSpec>(spec, "range_window");

  spec.param(doppler_window_, "doppler_window", "Doppler window",
             "WindowSpec applied along the chirp (slow-time) axis before "
             "the Doppler FFT (coherent-gain normalized). Default: none.",
             WindowSpec{});
  add_override_port<WindowSpec>(spec, "doppler_window");

  spec.param(apply_dechirp_, "apply_dechirp", "Apply software dechirp",
             "Software dechirp before the range FFT. Requires a waveform "
             "and complex-IF input when true.",
             false);
  add_override_port<bool>(spec, "apply_dechirp");

  spec.param(clutter_filter_, "clutter_filter", "Inline clutter filter",
             "Optional inline clutter stage between the range and Doppler "
             "stages {none, mean_subtraction, single_delay, double_delay}; "
             "same semantics as ClutterFilterOp's method. The delay "
             "cancellers reduce num_doppler.",
             ClutterFilterMethod::kNone);
  add_override_port<ClutterFilterMethod>(spec, "clutter_filter");

  spec.param(waveform_, "waveform", "FMCW waveform descriptor",
             "Dechirp reference and the range_bins / velocity_bins axes. "
             "Required for software dechirp and/or those outputs.",
             ParameterFlag::kOptional);
  add_override_port<FMCWWaveformDescriptor>(spec, "waveform");

  spec.param(fft_norm_, "fft_norm", "FFT normalization",
             "One normalization convention {backward, ortho, forward} for "
             "both transforms.",
             FftNorm::kBackward);
  add_override_port<FftNorm>(spec, "fft_norm");

  spec.param(center_zero_doppler_, "center_zero_doppler",
             "Center zero-Doppler",
             "Center zero-Doppler on the Doppler axis; else DC at bin 0.",
             true);
  add_override_port<bool>(spec, "center_zero_doppler");

  spec.param(propagation_speed_, "propagation_speed", "Propagation speed",
             "Wave speed (m/s) for the range_bins output.",
             kDefaultPropagationSpeed);
  add_override_port<double>(spec, "propagation_speed");

  spec.param(nfft_range_, "nfft_range", "Fast-time FFT length",
             "FFT length >= samples_per_chirp for the range stage. "
             "Default: samples_per_chirp.",
             ParameterFlag::kOptional);
  add_override_port<int64_t>(spec, "nfft_range");

  spec.param(nfft_doppler_, "nfft_doppler", "Slow-time FFT length",
             "FFT length >= the post-clutter chirp count for the Doppler "
             "stage. Default: the post-clutter chirp count.",
             ParameterFlag::kOptional);
  add_override_port<int64_t>(spec, "nfft_doppler");
}

template <typename T>
void FMCWRangeDopplerProcessorOp<T>::compute(InputContext& op_input,
                                             OutputContext& op_output,
                                             ExecutionContext& context) {
  // Implemented in a subsequent commit: range stage → optional
  // clutter stage → Doppler stage on async intermediates, via the same
  // stage functions as the split operators.
  throw std::runtime_error("FMCWRangeDopplerProcessorOp: not implemented yet");
}

template class FMCWRangeDopplerProcessorOp<float>;
template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
