// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_doppler_processor_op/fmcw_range_doppler_processor_op.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

constexpr const char* kOpName = "FMCWRangeDopplerProcessorOp";

// Kernel knobs for one fused tick: the two FFT stages plus the inline
// clutter method. Waveform / fft_norm are copied onto both stage configs.
struct FusedStageConfig {
  detail::RangeStageConfig range;
  detail::DopplerStageConfig doppler;
  ClutterFilterMethod clutter_filter{ClutterFilterMethod::kNone};
};

// Stage knobs plus output-axis lengths derived from this tick's input.
struct FusedStagePlan {
  FusedStageConfig stage;
  int64_t chirps_out;
  int64_t num_range_bins;
  int64_t num_doppler;
};

// Merge Exact [live] ports onto pinned Params and map sentinels to the
// stage form (nfft_* == 0 and an empty waveform become nullopt).
expected<FusedStageConfig, Error> resolve_params(
    Input<wire::WindowSpecT>& range_window_in, Input<wire::WindowSpecT>& doppler_window_in,
    Input<bool>& apply_dechirp_in, Input<ClutterFilterMethod>& clutter_filter_in,
    Input<wire::FMCWWaveformT>& waveform_in, Input<FftNorm>& fft_norm_in,
    Input<bool>& center_zero_doppler_in, Input<int64_t>& nfft_range_in,
    Input<int64_t>& nfft_doppler_in, const FMCWRangeDopplerProcessorParams& pinned) {
  HOLOSCAN_TRY(auto range_window, resolve_live_override_via(range_window_in, pinned.range_window,
                                                            [](const wire::WindowSpecT& value) {
                                                              return from_wire(value);
                                                            }));
  HOLOSCAN_TRY(auto doppler_window,
               resolve_live_override_via(doppler_window_in, pinned.doppler_window,
                                         [](const wire::WindowSpecT& value) {
                                           return from_wire(value);
                                         }));
  HOLOSCAN_TRY(auto apply_dechirp, resolve_live_override(apply_dechirp_in, pinned.apply_dechirp));
  HOLOSCAN_TRY(auto clutter_filter,
               resolve_live_override(clutter_filter_in, pinned.clutter_filter));
  HOLOSCAN_TRY(auto waveform, resolve_live_override_via(waveform_in, pinned.waveform,
                                                        [](const wire::FMCWWaveformT& value) {
                                                          return from_wire(value);
                                                        }));
  HOLOSCAN_TRY(auto fft_norm, resolve_live_override(fft_norm_in, pinned.fft_norm));
  HOLOSCAN_TRY(auto center_zero_doppler,
               resolve_live_override(center_zero_doppler_in, pinned.center_zero_doppler));
  HOLOSCAN_TRY(auto nfft_range_raw, resolve_live_override(nfft_range_in, pinned.nfft_range));
  HOLOSCAN_TRY(auto nfft_doppler_raw, resolve_live_override(nfft_doppler_in, pinned.nfft_doppler));

  std::optional<int64_t> nfft_range;
  if (nfft_range_raw != 0) {
    nfft_range = nfft_range_raw;
  }
  std::optional<int64_t> nfft_doppler;
  if (nfft_doppler_raw != 0) {
    nfft_doppler = nfft_doppler_raw;
  }
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform.is_valid()) {
    opt_waveform = std::move(waveform);
  }

  return FusedStageConfig{
      .range = detail::RangeStageConfig{.window = std::move(range_window),
                                        .apply_dechirp = apply_dechirp,
                                        .waveform = opt_waveform,
                                        .fft_norm = fft_norm,
                                        .propagation_speed = pinned.propagation_speed,
                                        .nfft = std::move(nfft_range)},
      .doppler = detail::DopplerStageConfig{.window = std::move(doppler_window),
                                            .center_zero_doppler = center_zero_doppler,
                                            .waveform = std::move(opt_waveform),
                                            .fft_norm = fft_norm,
                                            .nfft = std::move(nfft_doppler)},
      .clutter_filter = clutter_filter};
}

// Classify the input precision and return whether the input is complex-IF.
template <typename T>
expected<bool, Error> classify_complex_input(const Tensor& input) {
  try {
    const InputPrecision precision =
        classify_input_dtype<T>(input, "adc_cube", /*allow_real=*/true);
    return precision == InputPrecision::kComplex;
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
}

// Check the fused config against this tick's input and size both axes.
expected<FusedStagePlan, Error> build_fused_stage_plan(FusedStageConfig stage, bool complex_input,
                                                       const BatchedShape& batched) {
  if (stage.range.nfft.has_value() && *stage.range.nfft < batched.cols) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": nfft_range (" +
                                     std::to_string(*stage.range.nfft) +
                                     ") must be >= samples_per_chirp (" +
                                     std::to_string(batched.cols) + ") on port adc_cube"});
  }
  if (stage.range.apply_dechirp) {
    if (!stage.range.waveform.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   std::string(kOpName) +
                                       ": apply_dechirp = true requires a waveform descriptor "
                                       "(dechirp reference source)"});
    }
    if (!complex_input) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   std::string(kOpName) +
                                       ": apply_dechirp = true requires complex-IF input on port "
                                       "adc_cube (received real-IF samples; software dechirp "
                                       "produces complex samples, inconsistent with the real-IF "
                                       "output size)"});
    }
  }
  if (stage.clutter_filter == ClutterFilterMethod::kSingleDelay && batched.rows < 2) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": clutter_filter 'single_delay' needs num_chirps >= 2, "
                                     "got " +
                                     std::to_string(batched.rows) + " on port adc_cube"});
  }
  if (stage.clutter_filter == ClutterFilterMethod::kDoubleDelay && batched.rows < 3) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": clutter_filter 'double_delay' needs num_chirps >= 3, "
                                     "got " +
                                     std::to_string(batched.rows) + " on port adc_cube"});
  }

  const int64_t chirps_out = detail::clutter_num_chirps_out(stage.clutter_filter, batched.rows);
  if (stage.doppler.nfft.has_value() && *stage.doppler.nfft < chirps_out) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": nfft_doppler (" + std::to_string(*stage.doppler.nfft) +
            ") must be >= the post-clutter chirp count (" + std::to_string(chirps_out) + ")"});
  }

  const int64_t num_range_bins =
      detail::range_num_bins(complex_input, batched.cols, stage.range.nfft);
  const int64_t num_doppler = detail::doppler_num_bins(chirps_out, stage.doppler.nfft);
  return FusedStagePlan{.stage = std::move(stage),
                        .chirps_out = chirps_out,
                        .num_range_bins = num_range_bins,
                        .num_doppler = num_doppler};
}

}  // namespace

template <typename T>
FMCWRangeDopplerProcessorOp<T>::FMCWRangeDopplerProcessorOp(const MaxTensorExtents& extents) {
  range_state_ = std::make_shared<detail::RangeStageState<T>>();
  doppler_state_ = std::make_shared<detail::DopplerStageState<T>>();
  const char* who = "FMCWRangeDopplerProcessorOp";
  const std::int64_t max_chirps = require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins = require_extent(extents.max_range_bins, "max_range_bins", who);
  const std::int64_t max_doppler_bins =
      extents.max_doppler_bins.has_value()
          ? require_extent(extents.max_doppler_bins, "max_doppler_bins", who)
          : max_chirps;
  const std::int64_t max_rx = extent_rx_or_channels(extents);
  const std::int64_t max_tx = extent_or_one(extents.max_tx);
  if (max_rx <= 0 || max_tx <= 0) {
    throw std::invalid_argument(
        "FMCWRangeDopplerProcessorOp: max_rx/max_tx "
        "extents must be positive when set");
  }
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(2U + named_tx_rx_leading_rank(extents_));
  const std::int64_t max_output_doppler = std::max(max_chirps, max_doppler_bins);
  max_byte_span_ =
      byte_span_complex_cube<T>({max_tx, max_rx, max_output_doppler, max_range_bins}, who);
  range_bins_byte_span_ = byte_span_real_cube<T>({max_range_bins}, who);
  velocity_bins_byte_span_ = byte_span_real_cube<T>({max_output_doppler}, who);
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
  const TensorBounds fat_bounds{.max_rank = max_rank_, .max_byte_span = max_byte_span_};
  const TensorBounds rb_bounds{.max_rank = 1U, .max_byte_span = range_bins_byte_span_};
  const TensorBounds vb_bounds{.max_rank = 1U, .max_byte_span = velocity_bins_byte_span_};

  spec.input(adc_cube).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rd_maps).produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
  spec.output(range_bins).produces_tensor(pooled_tensor_output(kDevice, rb_bounds));
  spec.output(velocity_bins).produces_tensor(pooled_tensor_output(kDevice, vb_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(range_window_in).optional().queue_depth(2U);
  spec.input(doppler_window_in).optional().queue_depth(2U);
  spec.input(apply_dechirp_in).optional().queue_depth(2U);
  spec.input(clutter_filter_in).optional().queue_depth(2U);
  spec.input(waveform_in).optional().queue_depth(2U);
  spec.input(fft_norm_in).optional().queue_depth(2U);
  spec.input(center_zero_doppler_in).optional().queue_depth(2U);
  spec.input(nfft_range_in).optional().queue_depth(2U);
  spec.input(nfft_doppler_in).optional().queue_depth(2U);
}

template <typename T>
Contract FMCWRangeDopplerProcessorOp<T>::contract() const {
  // Trigger on the data tensor. Every Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss. Optional axis outs stay omitted so an unwired sink does
  // not gate scheduling.
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
expected<void, Error> FMCWRangeDopplerProcessorOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive adc_cube; require a device-contiguous buffer.
  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, kOpName));
  try {
    require_cuda_accessible(input, "adc_cube", kOpName);
    require_contiguous(input, "adc_cube", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Complex vs real IF, and collapse named leading axes (N_rx / M_tx).
  HOLOSCAN_TRY(auto complex_input, classify_complex_input<T>(input));
  HOLOSCAN_TRY(auto batched, collapse_named_tx_rx_batch(input, extents_, "adc_cube", kOpName));

  // This tick's Params: pinned defaults with any matched [live] override.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const FMCWRangeDopplerProcessorParams& pinned = context.params<FMCWRangeDopplerProcessorParams>();
  HOLOSCAN_TRY(auto stage,
               resolve_params(range_window_in, doppler_window_in, apply_dechirp_in,
                              clutter_filter_in, waveform_in, fft_norm_in, center_zero_doppler_in,
                              nfft_range_in, nfft_doppler_in, pinned));

  // Bind the fused config to this input's dtype/shape.
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan, build_fused_stage_plan(std::move(stage), complex_input, batched));

  // MatX work runs on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }
  // All retained artifacts on this instance share one static byte budget.
  range_state_->premultipliers.bind(cache_budget_);
  range_state_->windows.bind(cache_budget_);
  range_state_->seen_fft_keys.bind(cache_budget_);
  doppler_state_->windows.bind(cache_budget_);
  doppler_state_->seen_fft_keys.bind(cache_budget_);
  ConfigCacheBudget::Tick cache_tick(
      *cache_budget_, context.params<FMCWRangeDopplerProcessorParams>().cache_budget_bytes, [] {
        cudaDeviceSynchronize();
      });

  // Range compression into a stream-ordered intermediate (never a port).
  // MATX_ASYNC_DEVICE_MEMORY frees at scope exit after the enqueued kernels.
  auto profiles = matx::make_tensor<cuda::std::complex<T>>(
      {batched.batch, batched.rows, plan.num_range_bins}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
  const void* input_ptr = input.data();
  try {
    // Execute the device stage on the input runtime stream.
    detail::run_range_stage<T>(plan.stage.range, *range_state_, input_ptr, complex_input,
                               batched.batch, batched.rows, batched.cols, profiles.Data(),
                               plan.num_range_bins, stream);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Clutter fused into the Doppler pre-FFT expression: with_clutter_expr
  // feeds run_doppler_stage's expression entry point, so MatX evaluates it
  // inside the one fused pass that materializes the FFT input.
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, plan.num_doppler, plan.num_range_bins);
  const DLDataType out_dtype = complex_dtype<T>();

  // Allocate exact-sized outputs before launching device work.
  HOLOSCAN_TRY(auto rd_loan,
               prepare_device_tensor(
                   rd_maps, context, out_shape, out_dtype,
                   [&](void* dst, std::size_t) -> expected<void, Error> {
                     try {
                       const detail::CachedWindow<T>* clutter_window = nullptr;
                       if (plan.stage.clutter_filter == ClutterFilterMethod::kMeanSubtraction &&
                           plan.stage.doppler.window.type != WindowType::kNone) {
                         clutter_window = &detail::get_cached_window<T>(
                             plan.stage.doppler.window, batched.rows, doppler_state_->windows,
                             "Doppler window", stream);
                         detail::require_nonzero_window_sum(plan.stage.doppler.window,
                                                            *clutter_window);
                       }
                       detail::with_clutter_expr<T>(
                           plan.stage.clutter_filter, profiles, batched.rows, clutter_window,
                           [&](const auto& expr) {
                             detail::run_doppler_stage<T>(plan.stage.doppler, *doppler_state_, expr,
                                                          dst, plan.num_doppler, stream);
                           });
                     } catch (const std::invalid_argument& e) {
                       return fail_invalid(e.what());
                     }
                     return {};
                   }));
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps, context, std::move(rd_loan)));

  // Optional axes: waveform present and each latched sink still active.
  // Independent of the FFTs and of each other; an unwired emit is absorbed.
  if (plan.stage.range.waveform.has_value() && range_bins_out_.active()) {
    HOLOSCAN_TRY(auto bins_loan,
                 prepare_device_tensor(
                     range_bins, context, std::vector<int64_t>{plan.num_range_bins},
                     real_dtype<T>(), [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_range_bins<T>(plan.stage.range, batched.cols,
                                                    plan.num_range_bins, static_cast<T*>(dst),
                                                    stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       } catch (const std::runtime_error& e) {
                         return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                       }
                       return {};
                     }));
    range_bins_out_.emit(range_bins, context, std::move(bins_loan),
                         "FMCWRangeDopplerProcessorOp.range_bins");
  }
  if (plan.stage.doppler.waveform.has_value() && velocity_bins_out_.active()) {
    HOLOSCAN_TRY(auto vel_loan,
                 prepare_device_tensor(
                     velocity_bins, context, std::vector<int64_t>{plan.num_doppler},
                     real_dtype<T>(), [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_velocity_bins<T>(plan.stage.doppler, plan.num_doppler,
                                                       static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       } catch (const std::runtime_error& e) {
                         return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                       }
                       return {};
                     }));
    velocity_bins_out_.emit(velocity_bins, context, std::move(vel_loan),
                            "FMCWRangeDopplerProcessorOp.velocity_bins");
  }

  // Keep adc_cube alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class FMCWRangeDopplerProcessorOp<float>;
template class FMCWRangeDopplerProcessorOp<double>;

}  // namespace holoscan::radar
