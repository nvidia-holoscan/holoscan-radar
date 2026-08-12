// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/sar_range_compression_op/range_compression_stage.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

// The prefix every diagnostic from this operator carries. It depends only on
// the instantiation, so it is built once per precision rather than assembled
// on each tick for the error paths that almost never fire.
template <typename T>
const std::string& op_name() {
  static const std::string name =
      std::string("SARRangeCompressionOp<") + precision_name<T>() + ">";
  return name;
}

}  // namespace

template <typename T>
SARRangeCompressionOp<T>::~SARRangeCompressionOp() = default;

template <typename T>
void SARRangeCompressionOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();

  // Seed the effective override values from the now-populated parameters (the
  // parameter supplies the initial effective value; overrides are sticky) and
  // allocate the caches. Both are parameter-/host-only: the device-sized
  // artifacts (window taps, transform plans, the shift scratch) stay lazy
  // because their sizes come from the input, unknown until data arrives.
  state_ = std::make_shared<detail::RangeCompressionState<T>>();
  effective_window_ = range_window_.get();
  effective_waveform_ = waveform_.try_get();
  effective_fft_norm_ = fft_norm_.get();
  effective_nfft_ = nfft_.try_get();
  effective_propagation_speed_ = propagation_speed_.get();
  window_announced_ = false;
}

template <typename T>
void SARRangeCompressionOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("phase_history");
  spec.output<holoscan::TensorMap>("range_profiles");
  // Optional outputs: kNone so an unconnected port never gates scheduling.
  spec.output<holoscan::TensorMap>("range_bins")
      .condition(ConditionType::kNone);
  spec.output<WindowSpec>("range_window").condition(ConditionType::kNone);

  spec.param(range_window_, "range_window", "Range window",
             "WindowSpec applied along fast-time before the transform "
             "(coherent-gain normalized). The dominant control on "
             "impulse-response sidelobes. Default: none (rectangular).",
             WindowSpec{});
  add_override_port<WindowSpec>(spec, "range_window");

  spec.param(waveform_, "waveform", "LFM waveform descriptor",
             "Source for the range_bins axis. Required when range_bins is "
             "used; unused otherwise.",
             ParameterFlag::kOptional);
  add_override_port<LFMWaveformDescriptor>(spec, "waveform");

  spec.param(fft_norm_, "fft_norm", "Transform normalization",
             "Which direction carries the 1/N {backward, ortho, forward}. "
             "This stage's transform is the inverse one, so backward applies "
             "the 1/N here.",
             FftNorm::kBackward);
  add_override_port<FftNorm>(spec, "fft_norm");

  spec.param(nfft_, "nfft", "Fast-time transform length",
             "Transform length >= num_samples; larger zero-pads for a finer "
             "interpolated range grid over the same span. Default: "
             "num_samples.",
             ParameterFlag::kOptional);
  add_override_port<int64_t>(spec, "nfft");

  spec.param(propagation_speed_, "propagation_speed", "Propagation speed",
             "Wave speed (m/s) for the frequency to range conversion "
             "(range_bins only).",
             kDefaultPropagationSpeed);
  add_override_port<double>(spec, "propagation_speed");
}

template <typename T>
void SARRangeCompressionOp<T>::compute(InputContext& op_input,
                                       OutputContext& op_output,
                                       [[maybe_unused]] ExecutionContext&
                                           context) {
  const std::string& who = op_name<T>();

  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "phase_history", who);

  // Resolve the overridable parameters against the previous effective values
  // (sticky). warn_on_rebuild is raised only when an override changed a
  // costly-config value: the initial parameter-driven build stays silent and
  // cached configurations never warn.
  bool warn_on_rebuild = false;
  bool changed = false;
  effective_window_ = resolve_override_param(op_input, "range_window",
                                             effective_window_, &changed);
  if (changed) {
    warn_on_rebuild = true;
    // The descriptor output exists so a writer records the window that was
    // actually applied rather than a hand-mirrored copy, which means a change
    // has to re-announce.
    window_announced_ = false;
  }
  effective_fft_norm_ = resolve_override_param(op_input, "fft_norm",
                                               effective_fft_norm_, nullptr);
  effective_propagation_speed_ = resolve_override_param(
      op_input, "propagation_speed", effective_propagation_speed_, nullptr);

  auto waveform_override =
      receive_override<LFMWaveformDescriptor>(op_input, "waveform");
  if (waveform_override.has_value()) {
    effective_waveform_ = std::move(waveform_override);
  }

  auto nfft_override = receive_override<int64_t>(op_input, "nfft");
  if (nfft_override.has_value()) {
    if (!effective_nfft_.has_value() || *nfft_override != *effective_nfft_) {
      warn_on_rebuild = true;
    }
    effective_nfft_ = *nfft_override;
  }

  // Validate against the locked contract. Phase history is complex by
  // contract -- there is no real-input path here, unlike the FMCW stage.
  classify_input_dtype<T>(*input, "phase_history", /*allow_real=*/false);
  const BatchedShape batched = collapse_batch(*input, "phase_history");

  if (effective_nfft_.has_value() && *effective_nfft_ < batched.cols) {
    throw std::invalid_argument(
        who + ": nfft (" + std::to_string(*effective_nfft_) +
        ") must be >= num_samples (" + std::to_string(batched.cols) +
        ") on port phase_history");
  }

  cudaStream_t stream = op_input.receive_cuda_stream("phase_history");
  require_cuda_accessible(*input, "phase_history", who);
  // The stage builds a dense MatX view over the raw pointer.
  require_contiguous(*input, "phase_history", who);

  const int64_t num_bins =
      detail::compression_num_bins(batched.cols, effective_nfft_);

  detail::RangeCompressionConfig cfg;
  cfg.window = effective_window_;
  cfg.waveform = effective_waveform_;
  cfg.fft_norm = effective_fft_norm_;
  cfg.nfft = effective_nfft_;
  cfg.propagation_speed = effective_propagation_speed_;

  auto profiles = make_device_tensor(
      full_rank_shape(batched, batched.rows, num_bins), complex_dtype<T>());
  detail::run_range_compression<T>(cfg, *state_, input->data(), batched.batch,
                                   batched.rows, batched.cols,
                                   profiles->data(), num_bins, warn_on_rebuild,
                                   stream);

  // receive_cuda_stream already configures outputs to carry the internal
  // stream; set_cuda_stream additionally covers manual-stream setups. It
  // only accepts Holoscan-managed streams, so skip the default stream
  // (no stream pool and no upstream stream this tick).
  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "range_profiles");
  }
  holoscan::TensorMap profiles_map;
  profiles_map.insert({"range_profiles", profiles});
  op_output.emit(profiles_map, "range_profiles");

  // Optional centred range axis: only when an effective waveform is available.
  if (effective_waveform_.has_value() && effective_waveform_->is_valid()) {
    auto bins = make_device_tensor({num_bins}, real_dtype<T>());
    detail::fill_compressed_range_bins<T>(cfg, batched.cols, num_bins,
                                          static_cast<T*>(bins->data()),
                                          stream);
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "range_bins");
    }
    holoscan::TensorMap bins_map;
    bins_map.insert({"range_bins", bins});
    op_output.emit(bins_map, "range_bins");
  }

  // Optional descriptor output: the window actually applied, once at the
  // first tick and then only when it changes.
  if (!window_announced_) {
    op_output.emit(effective_window_, "range_window");
    window_announced_ = true;
  }

  // Stream-ordered lifetime guard: keep this tick's input alive until the next
  // tick (the previous tick's guard is released here, after this tick's
  // kernels are enqueued).
  retained_input_ = std::move(input);
}

template class SARRangeCompressionOp<float>;
template class SARRangeCompressionOp<double>;

}  // namespace holoscan::radar
