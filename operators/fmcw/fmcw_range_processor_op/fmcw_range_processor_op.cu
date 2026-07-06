// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

}  // namespace

template <typename T>
FMCWRangeProcessorOp<T>::~FMCWRangeProcessorOp() = default;

template <typename T>
void FMCWRangeProcessorOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

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
                                      [[maybe_unused]] ExecutionContext&
                                          context) {
  const std::string who =
      std::string("FMCWRangeProcessorOp<") + precision_name<T>() + ">";

  // 1. Receive the data input: a single-entry TensorMap, lenient on the
  // key (module convention).
  auto maybe_map = op_input.receive<holoscan::TensorMap>("adc_cube");
  if (!maybe_map) {
    throw std::runtime_error(who +
                             ": no message received on port adc_cube");
  }
  holoscan::TensorMap in_map = std::move(maybe_map.value());
  if (in_map.size() != 1) {
    throw std::invalid_argument(
        who + ": expected a single-entry TensorMap on port adc_cube, "
              "received " +
        std::to_string(in_map.size()) + " entries");
  }
  std::shared_ptr<holoscan::Tensor> input = in_map.begin()->second;
  if (!input) {
    throw std::runtime_error(who + ": null tensor received on adc_cube");
  }

  // First tick: allocate the kernel caches and seed the effective override
  // values from the parameters (the parameter supplies the initial
  // effective value; overrides are sticky per the contract).
  if (!state_) {
    state_ = std::make_shared<detail::RangeStageState<T>>();
    effective_window_ = range_window_.get();
    effective_apply_dechirp_ = apply_dechirp_.get();
    effective_waveform_ = waveform_.try_get();
    effective_fft_norm_ = fft_norm_.get();
    effective_propagation_speed_ = propagation_speed_.get();
    effective_nfft_ = nfft_.try_get();
  }

  // 2. Resolve the overridable parameters against the previous effective
  // values: an arriving override remains effective on later ticks until
  // another override arrives (sticky). warn_on_rebuild is raised only when
  // an override changed a costly-config value (window / waveform / nfft):
  // the initial parameter-driven build stays silent, cached configs never
  // warn.
  bool warn_on_rebuild = false;
  bool changed = false;
  effective_window_ = resolve_override_param(op_input, "range_window",
                                         effective_window_, &changed);
  warn_on_rebuild = warn_on_rebuild || changed;
  effective_apply_dechirp_ = resolve_override_param(
      op_input, "apply_dechirp", effective_apply_dechirp_, nullptr);
  effective_fft_norm_ = resolve_override_param(op_input, "fft_norm",
                                           effective_fft_norm_, nullptr);
  effective_propagation_speed_ =
      resolve_override_param(op_input, "propagation_speed",
                         effective_propagation_speed_, nullptr);

  auto waveform_override =
      receive_override<FMCWWaveformDescriptor>(op_input, "waveform");
  if (waveform_override.has_value()) {
    if (!effective_waveform_.has_value() ||
        *waveform_override != *effective_waveform_) {
      warn_on_rebuild = true;
    }
    effective_waveform_ = std::move(waveform_override);
  }

  auto nfft_override = receive_override<int64_t>(op_input, "nfft");
  if (nfft_override.has_value()) {
    if (!effective_nfft_.has_value() ||
        *nfft_override != *effective_nfft_) {
      warn_on_rebuild = true;
    }
    effective_nfft_ = *nfft_override;
  }

  // 3. Classify and validate the input against the locked contract.
  const InputPrecision precision =
      classify_input_dtype<T>(*input, "adc_cube", /*allow_real=*/true);
  const bool complex_input = precision == InputPrecision::kComplex;
  const BatchedShape batched = collapse_batch(*input, "adc_cube");

  if (effective_nfft_.has_value() && *effective_nfft_ < batched.cols) {
    throw std::invalid_argument(
        who + ": nfft (" + std::to_string(*effective_nfft_) +
        ") must be >= samples_per_chirp (" + std::to_string(batched.cols) +
        ") on port adc_cube");
  }
  if (effective_apply_dechirp_) {
    if (!effective_waveform_.has_value()) {
      throw std::invalid_argument(
          who + ": apply_dechirp = true requires a waveform descriptor "
                "(dechirp reference source)");
    }
    if (!complex_input) {
      throw std::invalid_argument(
          who + ": apply_dechirp = true requires complex-IF input on port "
                "adc_cube (received real-IF samples; software dechirp "
                "produces complex samples, inconsistent with the real-IF "
                "output size)");
    }
  }

  // 4. Operator stream (input streams synchronized; output ports emit it).
  cudaStream_t stream = op_input.receive_cuda_stream("adc_cube");

  // Host-memory input: stage to the device, stream-ordered. (Pageable H2D
  // copies synchronize on the host, so the source may be released after.)
  std::shared_ptr<holoscan::Tensor> device_input = input;
  if (!is_cuda_accessible(*input)) {
    device_input = make_device_tensor(input->shape(), input->dtype());
    if (cudaMemcpyAsync(device_input->data(), input->data(),
                        static_cast<size_t>(input->nbytes()),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(who + ": H2D staging copy failed");
    }
  }

  // 5. Run the range stage into a freshly allocated output tensor.
  const int64_t num_bins =
      detail::range_num_bins(complex_input, batched.cols, effective_nfft_);

  detail::RangeStageConfig cfg;
  cfg.window = effective_window_;
  cfg.apply_dechirp = effective_apply_dechirp_;
  cfg.waveform = effective_waveform_;
  cfg.fft_norm = effective_fft_norm_;
  cfg.propagation_speed = effective_propagation_speed_;
  cfg.nfft = effective_nfft_;

  auto profiles = make_device_tensor(
      full_rank_shape(batched, batched.rows, num_bins), complex_dtype<T>());
  detail::run_range_stage<T>(cfg, *state_, device_input->data(),
                             complex_input, batched.batch, batched.rows,
                             batched.cols, profiles->data(), num_bins,
                             warn_on_rebuild, stream);

  holoscan::TensorMap profiles_map;
  profiles_map.insert({"range_profiles", profiles});
  op_output.emit(profiles_map, "range_profiles");

  // 6. Optional range axis: only when an effective waveform is available.
  if (effective_waveform_.has_value()) {
    auto bins = make_device_tensor({num_bins}, real_dtype<T>());
    detail::fill_range_bins<T>(cfg, batched.cols, num_bins,
                               static_cast<T*>(bins->data()), stream);
    holoscan::TensorMap bins_map;
    bins_map.insert({"range_bins", bins});
    op_output.emit(bins_map, "range_bins");
  }

  // 7. Stream-ordered lifetime guard: keep this tick's input alive until
  // the next tick (the previous tick's guard is released here, after this
  // tick's kernels are enqueued).
  retained_input_ = std::move(device_input);
}

template class FMCWRangeProcessorOp<float>;
template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
