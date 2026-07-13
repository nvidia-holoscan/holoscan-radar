// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return sizeof(T) == 4 ? "float" : "double";
}

}  // namespace

template <typename T>
FMCWDopplerProcessorOp<T>::~FMCWDopplerProcessorOp() = default;

template <typename T>
void FMCWDopplerProcessorOp<T>::initialize() {
  // Register argument setters for the custom parameter types (WindowSpec,
  // FMCWWaveformDescriptor, FftNorm) before Operator::initialize()
  // processes the arguments. Idempotent.
  ensure_radar_arg_setters();
  Operator::initialize();

  // Seed the effective override values from the now-populated parameters
  // (the parameter supplies the initial effective value; overrides are
  // sticky per the contract) and allocate the kernel-side caches. Both are
  // parameter-/host-only: the device-sized artifacts (windows, dechirp
  // reference, FFT plans) stay lazy in the ConfigCaches because their
  // sizes come from the input cube, unknown until data arrives.
  state_ = std::make_shared<detail::DopplerStageState<T>>();
  effective_window_ = doppler_window_.get();
  effective_center_zero_doppler_ = center_zero_doppler_.get();
  effective_fft_norm_ = fft_norm_.get();
  effective_waveform_ = waveform_.try_get();
  effective_nfft_ = nfft_.try_get();
}

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
                                        [[maybe_unused]]
                                        ExecutionContext& context) {
  const std::string op_name =
      std::string("FMCWDopplerProcessorOp<") + precision_name<T>() + ">";

  // Module convention: accept a single-entry TensorMap regardless of its
  // key (upstream operators may name the tensor freely).
  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "range_profiles", op_name);

  // Resolve overrides against the previous effective values. Only
  // the costly configuration values (window, nfft) drive the warn-once
  // rule; the others are free to change silently.
  bool window_changed = false;
  effective_window_ = resolve_override_param<WindowSpec>(
      op_input, "doppler_window", effective_window_, &window_changed);
  effective_center_zero_doppler_ = resolve_override_param<bool>(
      op_input, "center_zero_doppler", effective_center_zero_doppler_,
      nullptr);
  effective_fft_norm_ = resolve_override_param<FftNorm>(
      op_input, "fft_norm", effective_fft_norm_, nullptr);
  if (auto waveform_override = receive_override<FMCWWaveformDescriptor>(
          op_input, "waveform")) {
    effective_waveform_ = std::move(*waveform_override);
  }
  bool nfft_changed = false;
  if (auto nfft_override =
          receive_override<int64_t>(op_input, "nfft")) {
    nfft_changed = !effective_nfft_.has_value() ||
                   *effective_nfft_ != *nfft_override;
    effective_nfft_ = *nfft_override;
  }
  const bool warn_on_rebuild = window_changed || nfft_changed;

  // Validate the input: complex dtype matching the instantiation, CUDA-
  // accessible, rank >= 2, compact row-major.
  classify_input_dtype<T>(*input, "range_profiles", /*allow_real=*/false);
  require_cuda_accessible(*input, "range_profiles", op_name);
  const BatchedShape batched = collapse_batch(*input, "range_profiles");
  const int64_t num_chirps = batched.rows;
  const int64_t num_range_bins = batched.cols;
  if (effective_nfft_.has_value() && *effective_nfft_ < num_chirps) {
    std::ostringstream msg;
    msg << op_name << ": nfft (" << *effective_nfft_
        << ") must be >= num_chirps (" << num_chirps
        << ") on input port 'range_profiles'";
    throw std::invalid_argument(msg.str());
  }

  // Synchronizes upstream work onto the operator's stream and configures
  // the output ports to emit it. Must follow the receive() above.
  cudaStream_t stream = op_input.receive_cuda_stream("range_profiles");

  detail::DopplerStageConfig cfg;
  cfg.window = effective_window_;
  cfg.center_zero_doppler = effective_center_zero_doppler_;
  cfg.waveform = effective_waveform_;
  cfg.fft_norm = effective_fft_norm_;
  cfg.nfft = effective_nfft_;

  const int64_t num_doppler =
      detail::doppler_num_bins(num_chirps, effective_nfft_);

  // Optional axis output, only when an effective waveform is available.
  // Enqueued before the Doppler FFT: its host-staged upload waits on the
  // stream, so this ordering keeps that wait off the FFT hot path.
  if (effective_waveform_.has_value()) {
    auto velocity = make_device_tensor({num_doppler}, real_dtype<T>());
    detail::fill_velocity_bins<T>(cfg, num_doppler,
                                  static_cast<T*>(velocity->data()), stream);
    holoscan::TensorMap velocity_map;
    velocity_map["velocity_bins"] = std::move(velocity);
    op_output.emit(velocity_map, "velocity_bins");
  }

  auto rd_maps = make_device_tensor(
      full_rank_shape(batched, num_doppler, num_range_bins),
      complex_dtype<T>());
  detail::run_doppler_stage<T>(cfg, *state_, input->data(), batched.batch,
                               num_chirps, num_range_bins, rd_maps->data(),
                               num_doppler, warn_on_rebuild, stream);
  holoscan::TensorMap rd_map;
  rd_map["rd_maps"] = std::move(rd_maps);
  op_output.emit(rd_map, "rd_maps");

  // Stream-ordered lifetime guard: keep this tick's input alive until the
  // next tick so enqueued kernels never outlive their source buffer.
  retained_input_ = std::move(input);
}

template class FMCWDopplerProcessorOp<float>;
template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
