// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/expected_macros.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "DopplerNotchOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<detail::DopplerNotchStageConfig, Error> resolve_stage_config(
    Input<double>& center_velocity_in, Input<double>& stop_half_width_in,
    Input<double>& transition_width_in, const DopplerNotchParams& pinned) {
  HOLOSCAN_TRY(auto center, resolve_live_override(center_velocity_in, pinned.center_velocity));
  HOLOSCAN_TRY(auto stop, resolve_live_override(stop_half_width_in, pinned.stop_half_width));
  HOLOSCAN_TRY(auto transition,
               resolve_live_override(transition_width_in, pinned.transition_width));
  if (!std::isfinite(center)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": center_velocity must be finite"});
  }
  if (!std::isfinite(stop) || stop < 0.0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": stop_half_width must be finite and non-negative"});
  }
  if (!std::isfinite(transition) || transition < 0.0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": transition_width must be finite and non-negative"});
  }
  return detail::DopplerNotchStageConfig{.center_velocity = center,
                                         .stop_half_width = stop,
                                         .transition_width = transition,
                                         .cyclic_velocity_axis = pinned.cyclic_velocity_axis};
}

}  // namespace

template <typename T>
DopplerNotchOp<T>::DopplerNotchOp(const MaxTensorExtents& extents) {
  const int64_t max_doppler = require_extent(extents.max_doppler_bins, "max_doppler_bins", kOpName);
  const int64_t max_range = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  int64_t max_tx = 1;
  int64_t max_rx = 1;
  require_tx_rx_pool_extents(extents, kOpName, &max_tx, &max_rx);
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>({max_tx, max_rx, max_doppler, max_range}, kOpName);
}

template <typename T>
void DopplerNotchOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(velocity_bins).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.output(rd_maps_out).produces_tensor(pooled_tensor_output(device, max_rank_, max_byte_span_));

  spec.input(center_velocity).optional().queue_depth(2U);
  spec.input(stop_half_width).optional().queue_depth(2U);
  spec.input(transition_width).optional().queue_depth(2U);
}

template <typename T>
Contract DopplerNotchOp<T>::contract() const {
  Contract result;
  result
      .trigger(
          OnAligned{.ports = ports(rd_maps, velocity_bins), .window = std::chrono::nanoseconds{0}})
      .input(rd_maps, Exact{})
      .input(velocity_bins, Exact{});
  add_live_override_to_contract(result, center_velocity);
  add_live_override_to_contract(result, stop_half_width);
  add_live_override_to_contract(result, transition_width);
  return result;
}

template <typename T>
expected<void, Error> DopplerNotchOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  HOLOSCAN_TRY(auto axis, receive_tensor(velocity_bins, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
    require_complex<T>(input, "rd_maps");
    require_cuda_accessible(axis, "velocity_bins", kOpName);
    require_contiguous(axis, "velocity_bins", kOpName);
    require_dtype(axis, real_dtype<T>(), "velocity_bins", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  HOLOSCAN_TRY(auto layout, collapse_named_tx_rx_batch(input, extents_, "rd_maps", kOpName));
  if (axis.shape().size() != 1U || axis.shape()[0] != layout.rows) {
    return fail_invalid(std::string(kOpName) +
                        ": velocity_bins must have shape (num_doppler,) matching rd_maps");
  }
  if (layout.rows > *extents_.max_doppler_bins || layout.cols > *extents_.max_range_bins) {
    return fail_invalid(std::string(kOpName) + ": rd_maps shape exceeds MaxTensorExtents");
  }

  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const DopplerNotchParams& pinned = context.params<DopplerNotchParams>();
  HOLOSCAN_TRY(auto config,
               resolve_stage_config(center_velocity, stop_half_width, transition_width, pinned));
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kOpName) + ": no CUDA compute stream"});
  }

  const auto shape = full_rank_shape(layout, layout.rows, layout.cols);
  const int64_t total = layout.batch * layout.rows * layout.cols;
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_device_tensor(rd_maps_out, context, shape, input.dtype(),
                                  [&](void* destination, std::size_t) -> expected<void, Error> {
                                    try {
                                      // Execute the device stage on the input runtime stream.
                                      detail::run_doppler_notch_stage<T>(
                                          static_cast<const cuda::std::complex<T>*>(input.data()),
                                          static_cast<const T*>(axis.data()),
                                          static_cast<cuda::std::complex<T>*>(destination), total,
                                          layout.rows, layout.cols, config, stream);
                                    } catch (const std::exception& e) {
                                      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                                    }
                                    return {};
                                  }));

  retained_rd_maps_ = std::move(input);
  retained_velocity_bins_ = std::move(axis);
  return {};
}

template class DopplerNotchOp<float>;
template class DopplerNotchOp<double>;

}  // namespace holoscan::radar
