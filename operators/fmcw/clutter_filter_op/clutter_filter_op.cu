// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
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

constexpr const char* kOpName = "ClutterFilterOp";

// Kernel knobs plus the output chirp count derived from this tick's input.
struct ClutterStagePlan {
  detail::ClutterStageConfig stage;
  int64_t chirps_out;
};

// Merge the Exact [live] method onto pinned Params. There are no Params
// sentinels to map.
expected<detail::ClutterStageConfig, Error> resolve_params(
    Input<ClutterFilterMethod>& method_in, Input<wire::WindowSpecT>& doppler_window_in,
    const ClutterFilterParams& pinned) {
  HOLOSCAN_TRY(auto method, resolve_live_override(method_in, pinned.method));
  HOLOSCAN_TRY(auto doppler_window,
               resolve_live_override_via(doppler_window_in, pinned.doppler_window,
                                         [](const wire::WindowSpecT& value) {
                                           return from_wire(value);
                                         }));
  return detail::ClutterStageConfig{.method = method, .window = std::move(doppler_window)};
}

// Check the stage config against this tick's input and size the chirp axis.
expected<ClutterStagePlan, Error> build_clutter_stage_plan(detail::ClutterStageConfig stage,
                                                           const BatchedShape& batched,
                                                           const MaxTensorExtents& extents) {
  if (batched.rows > *extents.max_chirps || batched.cols > *extents.max_range_bins) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": input chirps/range_bins exceed MaxTensorExtents"});
  }
  if (stage.method == ClutterFilterMethod::kSingleDelay && batched.rows < 2) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": method 'single_delay' needs num_chirps >= 2, got " +
                                     std::to_string(batched.rows)});
  }
  if (stage.method == ClutterFilterMethod::kDoubleDelay && batched.rows < 3) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) +
                                     ": method 'double_delay' needs num_chirps >= 3, got " +
                                     std::to_string(batched.rows)});
  }
  const int64_t chirps_out = detail::clutter_num_chirps_out(stage.method, batched.rows);
  return ClutterStagePlan{.stage = std::move(stage), .chirps_out = chirps_out};
}

}  // namespace

template <typename T>
ClutterFilterOp<T>::ClutterFilterOp(const MaxTensorExtents& extents)
: state_(std::make_shared<detail::ClutterStageState<T>>()) {  // NOLINT
  const char* who = kOpName;
  const std::int64_t max_chirps = require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins = require_extent(extents.max_range_bins, "max_range_bins", who);
  std::int64_t max_tx = 1;
  std::int64_t max_rx = 1;
  require_tx_rx_pool_extents(extents, who, &max_tx, &max_rx);
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>({max_tx, max_rx, max_chirps, max_range_bins}, who);
}

template <typename T>
ClutterFilterOp<T>::~ClutterFilterOp() = default;

template <typename T>
void ClutterFilterOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{.max_rank = max_rank_, .max_byte_span = max_byte_span_};

  spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(range_profiles_out).produces_tensor(pooled_tensor_output(kDevice, fat_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(method).optional().queue_depth(2U);
  spec.input(doppler_window_in).optional().queue_depth(2U);
}

template <typename T>
Contract ClutterFilterOp<T>::contract() const {
  // Trigger on the data tensor. The Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss.
  Contract result;
  result.trigger(OnEach{range_profiles});
  add_live_override_to_contract(result, method);
  add_live_override_to_contract(result, doppler_window_in);
  return result;
}

template <typename T>
expected<void, Error> ClutterFilterOp<T>::compute(ExecutionContext& context) {
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
  const ClutterFilterParams& pinned = context.params<ClutterFilterParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(method, doppler_window_in, pinned));

  // Bind the stage config to this input's shape.
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan, build_clutter_stage_plan(std::move(stage), batched, extents_));

  // MatX work runs on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }
  // All retained artifacts on this instance share one static byte budget.
  state_->windows.bind(cache_budget_);
  ConfigCacheBudget::Tick cache_tick(*cache_budget_,
                                     context.params<ClutterFilterParams>().cache_budget_bytes, [] {
                                       cudaDeviceSynchronize();
                                     });

  // Clutter suppression into a pool loan, then publish range_profiles_out.
  const std::vector<int64_t> out_shape = full_rank_shape(batched, plan.chirps_out, batched.cols);
  const DLDataType dtype = complex_dtype<T>();
  const void* input_ptr = input.data();

  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_device_tensor(range_profiles_out, context, out_shape, dtype,
                                  [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                                    try {
                                      // Execute the device stage on the input runtime stream.
                                      detail::run_clutter_stage<T>(plan.stage, *state_, input_ptr,
                                                                   batched.batch, batched.rows,
                                                                   batched.cols, out_ptr, stream);
                                    } catch (const std::invalid_argument& e) {
                                      return fail_invalid(e.what());
                                    }
                                    return {};
                                  }));

  // Keep range_profiles alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class ClutterFilterOp<float>;
template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
