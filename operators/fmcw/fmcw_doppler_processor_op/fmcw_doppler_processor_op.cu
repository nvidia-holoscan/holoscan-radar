// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
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

constexpr const char* kOpName = "FMCWDopplerProcessorOp";

// Kernel knobs plus output-axis decisions derived from this tick's input
// shape.
struct DopplerStagePlan {
  detail::DopplerStageConfig stage;
  int64_t num_doppler;
};

// Merge Exact [live] ports onto pinned Params and map sentinels to the
// stage form (nfft == 0 and an empty waveform become nullopt).
expected<detail::DopplerStageConfig, Error> resolve_params(
    Input<wire::WindowSpecT>& doppler_window_in, Input<bool>& center_zero_doppler_in,
    Input<wire::FMCWWaveformT>& waveform_in, Input<FftNorm>& fft_norm_in, Input<int64_t>& nfft_in,
    const FMCWDopplerProcessorParams& pinned) {
  HOLOSCAN_TRY(auto doppler_window,
               resolve_live_override_via(doppler_window_in, pinned.doppler_window,
                                         [](const wire::WindowSpecT& value) {
                                           return from_wire(value);
                                         }));
  HOLOSCAN_TRY(auto center_zero_doppler,
               resolve_live_override(center_zero_doppler_in, pinned.center_zero_doppler));
  HOLOSCAN_TRY(auto waveform, resolve_live_override_via(waveform_in, pinned.waveform,
                                                        [](const wire::FMCWWaveformT& value) {
                                                          return from_wire(value);
                                                        }));
  HOLOSCAN_TRY(auto fft_norm, resolve_live_override(fft_norm_in, pinned.fft_norm));
  HOLOSCAN_TRY(auto nfft_raw, resolve_live_override(nfft_in, pinned.nfft));

  std::optional<int64_t> nfft;
  if (nfft_raw != 0) {
    nfft = nfft_raw;
  }
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform.is_valid()) {
    opt_waveform = std::move(waveform);
  }

  return detail::DopplerStageConfig{.window = std::move(doppler_window),
                                    .center_zero_doppler = center_zero_doppler,
                                    .waveform = std::move(opt_waveform),
                                    .fft_norm = fft_norm,
                                    .nfft = nfft};
}

// Check the stage config against this tick's input and size the Doppler axis.
expected<DopplerStagePlan, Error> build_doppler_stage_plan(detail::DopplerStageConfig stage,
                                                           const BatchedShape& batched) {
  if (stage.nfft.has_value() && *stage.nfft < batched.rows) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": nfft (" + std::to_string(*stage.nfft) +
                                     ") must be >= num_chirps (" + std::to_string(batched.rows) +
                                     ") on input port 'range_profiles'"});
  }

  const int64_t num_doppler = detail::doppler_num_bins(batched.rows, stage.nfft);
  return DopplerStagePlan{.stage = std::move(stage), .num_doppler = num_doppler};
}

}  // namespace

template <typename T>
FMCWDopplerProcessorOp<T>::FMCWDopplerProcessorOp(const MaxTensorExtents& extents)
: state_(std::make_shared<detail::DopplerStageState<T>>()) {  // NOLINT
  const char* who = "FMCWDopplerProcessorOp";
  const std::int64_t max_chirps = require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins = require_extent(extents.max_range_bins, "max_range_bins", who);
  const std::int64_t max_rx = extent_rx_or_channels(extents);
  const std::int64_t max_tx = extent_or_one(extents.max_tx);
  if (max_rx <= 0 || max_tx <= 0) {
    throw std::invalid_argument(
        "FMCWDopplerProcessorOp: max_rx/max_tx extents "
        "must be positive when set");
  }
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>({max_tx, max_rx, max_chirps, max_range_bins}, who);
  velocity_bins_byte_span_ = byte_span_real_cube<T>({max_chirps}, who);
}

template <typename T>
FMCWDopplerProcessorOp<T>::~FMCWDopplerProcessorOp() = default;

template <typename T>
void FMCWDopplerProcessorOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{.max_rank = max_rank_, .max_byte_span = max_byte_span_};
  const TensorBounds vel_bounds{.max_rank = 1U, .max_byte_span = velocity_bins_byte_span_};

  spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rd_maps).produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
  spec.output(velocity_bins).produces_tensor(pooled_tensor_output(kDevice, vel_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(doppler_window_in).optional().queue_depth(2U);
  spec.input(center_zero_doppler_in).optional().queue_depth(2U);
  spec.input(waveform_in).optional().queue_depth(2U);
  spec.input(fft_norm_in).optional().queue_depth(2U);
  spec.input(nfft_in).optional().queue_depth(2U);
}

template <typename T>
Contract FMCWDopplerProcessorOp<T>::contract() const {
  // Trigger on the data tensor. Every Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss. Optional velocity_bins stays omitted so an unwired sink
  // does not gate scheduling.
  Contract result;
  result.trigger(OnEach{range_profiles});
  add_live_override_to_contract(result, doppler_window_in);
  add_live_override_to_contract(result, center_zero_doppler_in);
  add_live_override_to_contract(result, waveform_in);
  add_live_override_to_contract(result, fft_norm_in);
  add_live_override_to_contract(result, nfft_in);
  return result;
}

template <typename T>
expected<void, Error> FMCWDopplerProcessorOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive range_profiles; require a device-contiguous complex cube.
  HOLOSCAN_TRY(auto input, receive_tensor(range_profiles, kOpName));
  try {
    require_cuda_accessible(input, "range_profiles", kOpName);
    require_contiguous(input, "range_profiles", kOpName);
    require_complex<T>(input, "range_profiles");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Collapse named leading axes (N_rx / M_tx).
  HOLOSCAN_TRY(auto batched,
               collapse_named_tx_rx_batch(input, extents_, "range_profiles", kOpName));

  // This tick's Params: pinned defaults with any matched [live] override.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const FMCWDopplerProcessorParams& pinned = context.params<FMCWDopplerProcessorParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(doppler_window_in, center_zero_doppler_in, waveform_in,
                                          fft_norm_in, nfft_in, pinned));

  // Bind the stage config to this input's shape.
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan, build_doppler_stage_plan(std::move(stage), batched));

  // MatX work runs on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }

  // Doppler compression into a pool loan, then publish rd_maps.
  const std::vector<int64_t> out_shape = full_rank_shape(batched, plan.num_doppler, batched.cols);
  const DLDataType dtype = complex_dtype<T>();
  const void* input_ptr = input.data();

  // Allocate exact-sized outputs before launching device work.
  HOLOSCAN_TRY(auto rd_loan, prepare_device_tensor(
                                 rd_maps, context, out_shape, dtype,
                                 [&](void* dst, std::size_t) -> expected<void, Error> {
                                   try {
                                     // Execute the device stage on the input runtime stream.
                                     detail::run_doppler_stage<T>(
                                         plan.stage, *state_, input_ptr, batched.batch,
                                         batched.rows, batched.cols, dst, plan.num_doppler, stream);
                                   } catch (const std::invalid_argument& e) {
                                     return fail_invalid(e.what());
                                   }
                                   return {};
                                 }));
  // Optional velocity_bins is a causal sideband when the map immediately
  // drives DDMDenseDemultiplexOp. Publish it before rd_maps so that consumer's
  // Exact selector sees the matching axis. An unwired emit remains absorbed.
  if (plan.stage.waveform.has_value() && velocity_bins_out_.active()) {
    HOLOSCAN_TRY(auto bins_loan,
                 prepare_device_tensor(
                     velocity_bins, context, std::vector<int64_t>{plan.num_doppler},
                     real_dtype<T>(), [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_velocity_bins<T>(plan.stage, plan.num_doppler,
                                                       static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       } catch (const std::runtime_error& e) {
                         return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                       }
                       return {};
                     }));
    // Publish coherent outputs in dependency order and retain borrowed inputs.
    velocity_bins_out_.emit(velocity_bins, context, std::move(bins_loan),
                            "FMCWDopplerProcessorOp.velocity_bins");
  }

  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps, context, std::move(rd_loan)));

  // Keep range_profiles alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class FMCWDopplerProcessorOp<float>;
template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
