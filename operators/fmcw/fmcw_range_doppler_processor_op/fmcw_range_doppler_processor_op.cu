// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

}  // namespace

template <typename T>
FMCWRangeDopplerProcessorOp<T>::FMCWRangeDopplerProcessorOp(
    const MaxTensorExtents& extents) {
  const char* who = "FMCWRangeDopplerProcessorOp";
  const std::int64_t max_chirps =
      require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins = require_extent(
      extents.max_range_bins, "max_range_bins", who);
  const std::int64_t max_rx =
      extent_rx_or_channels(extents);
  const std::int64_t max_tx =
      extent_or_one(extents.max_tx);
  if (max_rx <= 0 || max_tx <= 0) {
    throw std::invalid_argument(
        "FMCWRangeDopplerProcessorOp: max_rx/max_tx "
        "extents must be positive when set");
  }
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(
      2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_range_bins},
      who);
  range_bins_byte_span_ =
      byte_span_real_cube<T>({max_range_bins}, who);
  velocity_bins_byte_span_ =
      byte_span_real_cube<T>({max_chirps}, who);
}

template <typename T>
FMCWRangeDopplerProcessorOp<T>::~FMCWRangeDopplerProcessorOp() = default;

template <typename T>
void FMCWRangeDopplerProcessorOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorBounds rb_bounds{
      .max_rank = 1U,
      .max_byte_span = range_bins_byte_span_};
  const TensorBounds vb_bounds{
      .max_rank = 1U,
      .max_byte_span = velocity_bins_byte_span_};

  spec.input(adc_cube).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(range_window_in).queue_depth(2U);
  spec.input(doppler_window_in).queue_depth(2U);
  spec.input(waveform_in).queue_depth(2U);
  spec.output(rd_maps)
      .produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
  spec.output(range_bins)
      .produces_tensor(pooled_tensor_output(kDevice, rb_bounds));
  spec.output(velocity_bins)
      .produces_tensor(pooled_tensor_output(kDevice, vb_bounds));

  // [live] override ports.
  spec.input(apply_dechirp_in).queue_depth(2U);
  spec.input(clutter_filter_in).queue_depth(2U);
  spec.input(fft_norm_in).queue_depth(2U);
  spec.input(center_zero_doppler_in).queue_depth(2U);
  spec.input(nfft_range_in).queue_depth(2U);
  spec.input(nfft_doppler_in).queue_depth(2U);
}

template <typename T>
Contract FMCWRangeDopplerProcessorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{adc_cube});
  add_live_override_to_contract(result, range_window_in);
  add_live_override_to_contract(result, doppler_window_in);
  add_live_override_to_contract(result, apply_dechirp_in);
  add_live_override_to_contract(result, clutter_filter_in);
  add_live_override_to_contract(result, waveform_in);
  add_live_override_to_contract(result, fft_norm_in);
  add_live_override_to_contract(result, center_zero_doppler_in);
  add_live_override_to_contract(result, nfft_range_in);
  add_live_override_to_contract(result, nfft_doppler_in);
  return result;
}

template <typename T>
expected<void, Error> FMCWRangeDopplerProcessorOp<T>::compute(
    ExecutionContext& context) {
  const std::string op_name =
      std::string("FMCWRangeDopplerProcessorOp<") + precision_name<T>() +
      ">";

  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, op_name));

  const FMCWRangeDopplerProcessorParams& pinned =
      context.params<FMCWRangeDopplerProcessorParams>();

  // Resolve Exact [live] overrides. Schema conversion only runs for a
  // matched wire sample; a tick without one uses the Params value.
  HOLOSCAN_TRY(
      auto eff_range_window,
      resolve_live_override_via(
          range_window_in, pinned.range_window,
          [](const wire::WindowSpecT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto eff_doppler_window,
      resolve_live_override_via(
          doppler_window_in, pinned.doppler_window,
          [](const wire::WindowSpecT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(auto eff_apply_dechirp,
               resolve_live_override(
                   apply_dechirp_in, pinned.apply_dechirp));
  HOLOSCAN_TRY(auto eff_clutter_filter,
               resolve_live_override(
                   clutter_filter_in, pinned.clutter_filter));
  HOLOSCAN_TRY(
      auto eff_waveform_value,
      resolve_live_override_via(
          waveform_in, pinned.waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(auto eff_fft_norm,
               resolve_live_override(
                   fft_norm_in, pinned.fft_norm));
  HOLOSCAN_TRY(auto eff_center_zero_doppler,
               resolve_live_override(
                   center_zero_doppler_in, pinned.center_zero_doppler));
  HOLOSCAN_TRY(auto eff_nfft_range_raw,
               resolve_live_override(
                   nfft_range_in, pinned.nfft_range));
  HOLOSCAN_TRY(auto eff_nfft_doppler_raw,
               resolve_live_override(
                   nfft_doppler_in, pinned.nfft_doppler));

  // nfft 0 = "use natural length" (Params "none" convention).
  const std::optional<int64_t> eff_nfft_range =
      eff_nfft_range_raw != 0 ? std::optional<int64_t>{eff_nfft_range_raw}
                               : std::nullopt;
  const std::optional<int64_t> eff_nfft_doppler =
      eff_nfft_doppler_raw != 0
          ? std::optional<int64_t>{eff_nfft_doppler_raw}
          : std::nullopt;

  const bool waveform_present = eff_waveform_value.is_valid();
  const std::optional<FMCWWaveformDescriptor> eff_waveform =
      waveform_present ? std::optional{eff_waveform_value} : std::nullopt;

  // Classify and validate input against the locked contract.
  InputPrecision precision;
  try {
    precision = classify_input_dtype<T>(input, "adc_cube",
                                        /*allow_real=*/true);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const bool complex_input = precision == InputPrecision::kComplex;

  try {
    require_cuda_accessible(input, "adc_cube", op_name);
    require_contiguous(input, "adc_cube", op_name);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  BatchedShape batched;
  try {
    batched = collapse_batch(input, "adc_cube");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const std::size_t max_leading =
      named_tx_rx_leading_rank(extents_);
  if (batched.leading.size() > max_leading) {
    return fail_invalid(unexpected_leading_message(
        op_name.c_str(), batched.leading.size(),
        max_leading));
  }
  const std::int64_t max_rx_val =
      extent_rx_or_channels(extents_);
  const std::int64_t max_tx_val =
      extent_or_one(extents_.max_tx);
  if (batched.batch > max_tx_val * max_rx_val) {
    return fail_invalid(
        op_name + ": leading product " +
        std::to_string(batched.batch) +
        " exceeds max_tx*max_rx from "
        "MaxTensorExtents");
  }

  const int64_t num_chirps = batched.rows;
  const int64_t samples_per_chirp = batched.cols;

  if (eff_nfft_range.has_value() && *eff_nfft_range < samples_per_chirp) {
    return fail_invalid(
        op_name + ": nfft_range (" + std::to_string(*eff_nfft_range) +
        ") must be >= samples_per_chirp (" +
        std::to_string(samples_per_chirp) + ") on port adc_cube");
  }
  if (eff_apply_dechirp) {
    if (!waveform_present) {
      return fail_invalid(
          op_name +
          ": apply_dechirp = true requires a waveform descriptor "
          "(dechirp reference source)");
    }
    if (!complex_input) {
      return fail_invalid(
          op_name +
          ": apply_dechirp = true requires complex-IF input on port "
          "adc_cube (received real-IF samples; software dechirp "
          "produces complex samples, inconsistent with the real-IF "
          "output size)");
    }
  }
  if (eff_clutter_filter == ClutterFilterMethod::kSingleDelay &&
      num_chirps < 2) {
    return fail_invalid(
        op_name +
        ": clutter_filter 'single_delay' needs num_chirps >= 2, "
        "got " +
        std::to_string(num_chirps) + " on port adc_cube");
  }
  if (eff_clutter_filter == ClutterFilterMethod::kDoubleDelay &&
      num_chirps < 3) {
    return fail_invalid(
        op_name +
        ": clutter_filter 'double_delay' needs num_chirps >= 3, "
        "got " +
        std::to_string(num_chirps) + " on port adc_cube");
  }
  const int64_t chirps_out =
      detail::clutter_num_chirps_out(eff_clutter_filter, num_chirps);
  if (eff_nfft_doppler.has_value() && *eff_nfft_doppler < chirps_out) {
    return fail_invalid(
        op_name + ": nfft_doppler (" +
        std::to_string(*eff_nfft_doppler) +
        ") must be >= the post-clutter chirp count (" +
        std::to_string(chirps_out) + ")");
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 op_name + ": no CUDA compute stream"});
  }

  if (!range_state_) {
    range_state_ = std::make_shared<detail::RangeStageState<T>>();
  }
  if (!doppler_state_) {
    doppler_state_ = std::make_shared<detail::DopplerStageState<T>>();
  }

  // Build the two stage configs from effective parameters.
  detail::RangeStageConfig range_cfg;
  range_cfg.window = eff_range_window;
  range_cfg.apply_dechirp = eff_apply_dechirp;
  range_cfg.waveform = eff_waveform;
  range_cfg.fft_norm = eff_fft_norm;
  range_cfg.propagation_speed = pinned.propagation_speed;
  range_cfg.nfft = eff_nfft_range;

  detail::DopplerStageConfig doppler_cfg;
  doppler_cfg.window = eff_doppler_window;
  doppler_cfg.center_zero_doppler = eff_center_zero_doppler;
  doppler_cfg.waveform = eff_waveform;
  doppler_cfg.fft_norm = eff_fft_norm;
  doppler_cfg.nfft = eff_nfft_doppler;

  const int64_t num_range_bins =
      detail::range_num_bins(complex_input, samples_per_chirp, eff_nfft_range);
  const int64_t num_doppler =
      detail::doppler_num_bins(chirps_out, eff_nfft_doppler);

  // Optional axis outputs, only when an effective waveform is available.
  // Enqueued before the FFT stages: their host-staged uploads wait on the
  // stream, so this ordering keeps those waits off the FFT hot path.
  // Each axis gates on its own latch: one port going unconsumed must not
  // stop producing the other.
  std::optional<TensorOutputLoan> bins_loan;
  std::optional<TensorOutputLoan> velocity_loan;
  if (waveform_present && range_bins_out_.active()) {
    HOLOSCAN_TRY(auto rb_loan,
                 prepare_device_tensor(
                     range_bins, context,
                     std::vector<int64_t>{num_range_bins},
                     real_dtype<T>(),
                     [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_range_bins<T>(
                             range_cfg, samples_per_chirp, num_range_bins,
                             static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       }
                       return {};
                     }));
    bins_loan.emplace(std::move(rb_loan));
  }
  if (waveform_present && velocity_bins_out_.active()) {
    HOLOSCAN_TRY(auto vb_loan,
                 prepare_device_tensor(
                     velocity_bins, context,
                     std::vector<int64_t>{num_doppler},
                     real_dtype<T>(),
                     [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_velocity_bins<T>(
                             doppler_cfg, num_doppler,
                             static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       }
                       return {};
                     }));
    velocity_loan.emplace(std::move(vb_loan));
  }

  // Range stage into a stream-ordered intermediate (intermediates never
  // leave the operator; MATX_ASYNC_DEVICE_MEMORY frees stream-ordered at
  // scope exit, after the enqueued kernels).
  auto profiles = matx::make_tensor<cuda::std::complex<T>>(
      {batched.batch, num_chirps, num_range_bins},
      matx::MATX_ASYNC_DEVICE_MEMORY, stream);

  const void* input_ptr = input.data();
  try {
    detail::run_range_stage<T>(range_cfg, *range_state_, input_ptr,
                               complex_input, batched.batch, num_chirps,
                               samples_per_chirp, profiles.Data(),
                               num_range_bins, stream);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Clutter fused into the Doppler pre-FFT expression: the clutter math
  // (single-sourced from clutter_stage.hpp's with_clutter_expr) feeds
  // run_doppler_stage's expression entry point directly, so MatX evaluates
  // it inside the one fused pass that materializes the FFT input.
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, num_doppler, num_range_bins);
  const DLDataType out_dtype = complex_dtype<T>();

  HOLOSCAN_TRY(auto rd_loan,
               prepare_device_tensor(
                   rd_maps, context, out_shape, out_dtype,
                   [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                     try {
                       detail::with_clutter_expr(
                           eff_clutter_filter, profiles, num_chirps,
                           [&](const auto& expr) {
                             detail::run_doppler_stage<T>(
                                 doppler_cfg, *doppler_state_, expr,
                                 out_ptr, num_doppler, stream);
                           });
                     } catch (const std::invalid_argument& e) {
                       return fail_invalid(e.what());
                     }
                     return {};
                   }));

  // Emit all prepared loans together.
  if (bins_loan.has_value()) {
    range_bins_out_.emit(range_bins, context, std::move(*bins_loan),
                         "FMCWRangeDopplerProcessorOp.range_bins");
  }
  if (velocity_loan.has_value()) {
    velocity_bins_out_.emit(velocity_bins, context, std::move(*velocity_loan),
                            "FMCWRangeDopplerProcessorOp.velocity_bins");
  }
  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps, context, std::move(rd_loan)));

  retained_input_ = std::move(input);
  return {};
}

template class FMCWRangeDopplerProcessorOp<float>;
template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
