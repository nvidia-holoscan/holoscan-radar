// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
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
FMCWRangeDopplerProcessorOp<T>::~FMCWRangeDopplerProcessorOp() = default;

template <typename T>
void FMCWRangeDopplerProcessorOp<T>::initialize() {
  // Register argument setters for the custom parameter types (WindowSpec,
  // FMCWWaveformDescriptor, FftNorm, ClutterFilterMethod) before
  // Operator::initialize() processes the arguments. Idempotent.
  ensure_radar_arg_setters();
  Operator::initialize();

  // Seed the effective override values from the now-populated parameters
  // (the parameter supplies the initial effective value; overrides are
  // sticky per the contract) and allocate the kernel-side caches. Both are
  // parameter-/host-only: the device-sized artifacts (windows, dechirp
  // reference, FFT plans) stay lazy in the ConfigCaches because their
  // sizes come from the input cube, unknown until data arrives.
  range_state_ = std::make_shared<detail::RangeStageState<T>>();
  doppler_state_ = std::make_shared<detail::DopplerStageState<T>>();
  effective_range_window_ = range_window_.get();
  effective_doppler_window_ = doppler_window_.get();
  effective_apply_dechirp_ = apply_dechirp_.get();
  effective_clutter_filter_ = clutter_filter_.get();
  effective_waveform_ = waveform_.try_get();
  effective_fft_norm_ = fft_norm_.get();
  effective_center_zero_doppler_ = center_zero_doppler_.get();
  effective_propagation_speed_ = propagation_speed_.get();
  effective_nfft_range_ = nfft_range_.try_get();
  effective_nfft_doppler_ = nfft_doppler_.try_get();
}

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
                                             [[maybe_unused]]
                                             ExecutionContext& context) {
  const std::string who =
      std::string("FMCWRangeDopplerProcessorOp<") + precision_name<T>() +
      ">";

  // 1. Receive the data input: a single-entry TensorMap, lenient on the
  // key (module convention).
  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "adc_cube", who);

  // 2. Resolve overrides against the previous effective values.
  // Only the costly configuration values drive the warn-once rule, split
  // per stage: range_window / waveform / nfft_range feed the range stage's
  // flag, doppler_window / nfft_doppler the Doppler stage's.
  bool warn_range = false;
  bool warn_doppler = false;
  bool changed = false;
  effective_range_window_ = resolve_override_param<WindowSpec>(
      op_input, "range_window", effective_range_window_, &changed);
  warn_range = warn_range || changed;
  effective_doppler_window_ = resolve_override_param<WindowSpec>(
      op_input, "doppler_window", effective_doppler_window_, &changed);
  warn_doppler = warn_doppler || changed;
  effective_apply_dechirp_ = resolve_override_param<bool>(
      op_input, "apply_dechirp", effective_apply_dechirp_, nullptr);
  effective_clutter_filter_ = resolve_override_param<ClutterFilterMethod>(
      op_input, "clutter_filter", effective_clutter_filter_, nullptr);
  if (auto waveform_override = receive_override<FMCWWaveformDescriptor>(
          op_input, "waveform")) {
    // The dechirp reference is a range-stage cache.
    if (!effective_waveform_.has_value() ||
        *waveform_override != *effective_waveform_) {
      warn_range = true;
    }
    effective_waveform_ = std::move(*waveform_override);
  }
  effective_fft_norm_ = resolve_override_param<FftNorm>(
      op_input, "fft_norm", effective_fft_norm_, nullptr);
  effective_center_zero_doppler_ = resolve_override_param<bool>(
      op_input, "center_zero_doppler", effective_center_zero_doppler_,
      nullptr);
  effective_propagation_speed_ = resolve_override_param<double>(
      op_input, "propagation_speed", effective_propagation_speed_, nullptr);
  if (auto nfft_override =
          receive_override<int64_t>(op_input, "nfft_range")) {
    if (!effective_nfft_range_.has_value() ||
        *effective_nfft_range_ != *nfft_override) {
      warn_range = true;
    }
    effective_nfft_range_ = *nfft_override;
  }
  if (auto nfft_override =
          receive_override<int64_t>(op_input, "nfft_doppler")) {
    if (!effective_nfft_doppler_.has_value() ||
        *effective_nfft_doppler_ != *nfft_override) {
      warn_doppler = true;
    }
    effective_nfft_doppler_ = *nfft_override;
  }

  // 3. Classify and validate the input against the locked contract.
  const InputPrecision precision =
      classify_input_dtype<T>(*input, "adc_cube", /*allow_real=*/true);
  const bool complex_input = precision == InputPrecision::kComplex;
  const BatchedShape batched = collapse_batch(*input, "adc_cube");
  const int64_t num_chirps = batched.rows;
  const int64_t samples_per_chirp = batched.cols;

  if (effective_nfft_range_.has_value() &&
      *effective_nfft_range_ < samples_per_chirp) {
    throw std::invalid_argument(
        who + ": nfft_range (" + std::to_string(*effective_nfft_range_) +
        ") must be >= samples_per_chirp (" +
        std::to_string(samples_per_chirp) + ") on port adc_cube");
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
  if (effective_clutter_filter_ == ClutterFilterMethod::kSingleDelay &&
      num_chirps < 2) {
    throw std::invalid_argument(
        who + ": clutter_filter 'single_delay' needs num_chirps >= 2, "
              "got " +
        std::to_string(num_chirps) + " on port adc_cube");
  }
  if (effective_clutter_filter_ == ClutterFilterMethod::kDoubleDelay &&
      num_chirps < 3) {
    throw std::invalid_argument(
        who + ": clutter_filter 'double_delay' needs num_chirps >= 3, "
              "got " +
        std::to_string(num_chirps) + " on port adc_cube");
  }
  const int64_t chirps_out = detail::clutter_num_chirps_out(
      effective_clutter_filter_, num_chirps);
  if (effective_nfft_doppler_.has_value() &&
      *effective_nfft_doppler_ < chirps_out) {
    throw std::invalid_argument(
        who + ": nfft_doppler (" + std::to_string(*effective_nfft_doppler_) +
        ") must be >= the post-clutter chirp count (" +
        std::to_string(chirps_out) + ")");
  }

  // 4. Operator stream (input streams synchronized; output ports emit it).
  cudaStream_t stream = op_input.receive_cuda_stream("adc_cube");

  // CUDA-accessible input only (module-wide; ingest operators upload
  // before the processing chain — STATUS.md decision 2026-07-10).
  require_cuda_accessible(*input, "adc_cube", who);

  // 5. Build the two stage configs from the effective params — the range
  // stage runs as the SAME stage call the split operator makes, and the
  // clutter math is the SAME single-source expression (with_clutter_expr)
  // composed into the SAME Doppler entry point, which is what guarantees
  // split-vs-fused parity by construction.
  detail::RangeStageConfig range_cfg;
  range_cfg.window = effective_range_window_;
  range_cfg.apply_dechirp = effective_apply_dechirp_;
  range_cfg.waveform = effective_waveform_;
  range_cfg.fft_norm = effective_fft_norm_;
  range_cfg.propagation_speed = effective_propagation_speed_;
  range_cfg.nfft = effective_nfft_range_;

  detail::DopplerStageConfig doppler_cfg;
  doppler_cfg.window = effective_doppler_window_;
  doppler_cfg.center_zero_doppler = effective_center_zero_doppler_;
  doppler_cfg.waveform = effective_waveform_;
  doppler_cfg.fft_norm = effective_fft_norm_;
  doppler_cfg.nfft = effective_nfft_doppler_;

  const int64_t num_range_bins = detail::range_num_bins(
      complex_input, samples_per_chirp, effective_nfft_range_);
  const int64_t num_doppler =
      detail::doppler_num_bins(chirps_out, effective_nfft_doppler_);

  // 6. Optional axis outputs, only when an effective waveform is
  // available. Enqueued before the FFT stages: their host-staged uploads
  // wait on the stream, so this ordering keeps those waits off the FFT
  // hot path.
  if (effective_waveform_.has_value()) {
    auto range_bins = make_device_tensor({num_range_bins}, real_dtype<T>());
    detail::fill_range_bins<T>(range_cfg, samples_per_chirp, num_range_bins,
                               static_cast<T*>(range_bins->data()), stream);
    holoscan::TensorMap range_bins_map;
    range_bins_map.insert({"range_bins", std::move(range_bins)});
    op_output.emit(range_bins_map, "range_bins");

    auto velocity = make_device_tensor({num_doppler}, real_dtype<T>());
    detail::fill_velocity_bins<T>(doppler_cfg, num_doppler,
                                  static_cast<T*>(velocity->data()), stream);
    holoscan::TensorMap velocity_map;
    velocity_map.insert({"velocity_bins", std::move(velocity)});
    op_output.emit(velocity_map, "velocity_bins");
  }

  // 7. Range stage into a stream-ordered intermediate (intermediates never
  // leave the operator; MATX_ASYNC_DEVICE_MEMORY frees stream-ordered at
  // scope exit, after the enqueued kernels).
  auto profiles = matx::make_tensor<cuda::std::complex<T>>(
      {batched.batch, num_chirps, num_range_bins},
      matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  detail::run_range_stage<T>(range_cfg, *range_state_, input->data(),
                             complex_input, batched.batch, num_chirps,
                             samples_per_chirp, profiles.Data(),
                             num_range_bins, warn_range, stream);

  // 8+9. Clutter fused into the Doppler pre-FFT expression: the clutter
  // math (single-sourced from clutter_stage.hpp's with_clutter_expr, the
  // same expressions ClutterFilterOp materializes) feeds run_doppler_stage's
  // expression entry point directly, so MatX evaluates it inside the one
  // fused pass that materializes the FFT input — no post-clutter cube
  // write+read or extra kernel launch. kNone passes `profiles` through
  // unchanged (zero added kernels), matching the split chain omitting
  // ClutterFilterOp. Agreement with the split chain — numerical parity to
  // within the permitted ULP bound, not byte identity — is enforced by the
  // UlpParity tests. Everything is enqueued inside the visitor, so
  // `profiles` outlives the kernels via its stream-ordered free.
  auto rd_maps = make_device_tensor(
      full_rank_shape(batched, num_doppler, num_range_bins),
      complex_dtype<T>());
  detail::with_clutter_expr(
      effective_clutter_filter_, profiles, num_chirps,
      [&](const auto& expr) {
        detail::run_doppler_stage<T>(doppler_cfg, *doppler_state_, expr,
                                     rd_maps->data(), num_doppler,
                                     warn_doppler, stream);
      });
  holoscan::TensorMap rd_map;
  rd_map.insert({"rd_maps", std::move(rd_maps)});
  op_output.emit(rd_map, "rd_maps");

  // 10. Stream-ordered lifetime guard: keep this tick's input alive until
  // the next tick so enqueued kernels never outlive their source buffer.
  retained_input_ = std::move(input);
}

template class FMCWRangeDopplerProcessorOp<float>;
template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
