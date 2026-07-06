// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"

#include <stdexcept>

#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
#include "radar_common/sdk/override_param.hpp"

namespace holoscan::radar {

template <typename T>
FMCWDopplerProcessorOp<T>::~FMCWDopplerProcessorOp() = default;

template <typename T>
void FMCWDopplerProcessorOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("range_profiles");
  spec.output<holoscan::TensorMap>("rd_maps");
  // Optional axis output: emitted only when a waveform descriptor is
  // available; kNone so an unconnected port never gates scheduling.
  spec.output<holoscan::TensorMap>("velocity_bins")
      .condition(ConditionType::kNone);

  spec.param(doppler_window_, "doppler_window", "Doppler window",
             "WindowSpec applied along the chirp (slow-time) axis before "
             "the Doppler FFT (coherent-gain normalized). Default: none.",
             WindowSpec{});
  add_override_port<WindowSpec>(spec, "doppler_window");

  spec.param(center_zero_doppler_, "center_zero_doppler",
             "Center zero-Doppler",
             "fftshift the Doppler axis so zero-Doppler is centered "
             "(conventional RD layout); else DC at bin 0.",
             true);
  add_override_port<bool>(spec, "center_zero_doppler");

  spec.param(waveform_, "waveform", "FMCW waveform descriptor",
             "Source for the velocity_bins axis (carrier -> wavelength, "
             "chirp repetition interval -> PRF). Required only when "
             "velocity_bins is used.",
             ParameterFlag::kOptional);
  add_override_port<FMCWWaveformDescriptor>(spec, "waveform");

  spec.param(fft_norm_, "fft_norm", "FFT normalization",
             "FFT normalization convention {backward, ortho, forward}; "
             "backward keeps coherent-integration gain visible.",
             FftNorm::kBackward);
  add_override_port<FftNorm>(spec, "fft_norm");

  spec.param(nfft_, "nfft", "Slow-time FFT length",
             "FFT length >= num_chirps; larger interpolates the velocity "
             "grid over the same span. Default: num_chirps.",
             ParameterFlag::kOptional);
  add_override_port<int64_t>(spec, "nfft");
}

template <typename T>
void FMCWDopplerProcessorOp<T>::compute(InputContext& op_input,
                                        OutputContext& op_output,
                                        ExecutionContext& context) {
  // Implemented in a subsequent commit.
  throw std::runtime_error("FMCWDopplerProcessorOp: not implemented yet");
}

template class FMCWDopplerProcessorOp<float>;
template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
