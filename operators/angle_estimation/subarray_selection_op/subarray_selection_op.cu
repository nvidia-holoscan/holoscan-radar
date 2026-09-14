// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/dsp/subarray_selection.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

namespace detail {

struct SubarraySelectionState {
  bool has_selection{false};
  ArrayGeometry geometry_key{};
  std::vector<int64_t> subarray_key{};
  int64_t min_subarray_size_key{0};
  SubarraySelection selection{};
  std::shared_ptr<holoscan::Tensor> channel_index{};
  int64_t channel_index_capacity{0};
};

}  // namespace detail

namespace {

constexpr const char* kOpName = "SubarraySelectionOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor,
                                    const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " is not writable: " +
            mut.error().message});
  }
  return *mut;
}

struct SubarraySelectionStageConfig {
  ArrayGeometry geometry;
  std::vector<int64_t> subarray;
  int64_t min_subarray_size{0};
};

struct SubarraySelectionStagePlan {
  SubarraySelectionStageConfig stage;
  int64_t num_items{0};
  int64_t num_channels{0};
  int64_t num_snapshots{0};
  bool has_snapshot_axis{false};
};

expected<SubarraySelectionStageConfig, Error> resolve_params(
    Input<WireArrayGeometry>& geometry_port, Input<Tensor>& subarray_port,
    const SubarraySelectionParams& pinned, int64_t live_subarray_bound) {
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          geometry_port, pinned.array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto subarray,
      resolve_live_override_via(
          subarray_port, pinned.subarray,
          [live_subarray_bound](const Tensor& value) {
            return read_host_config_vector<int64_t>(
                value, DLDataType{kDLInt, 64U, 1U},
                live_subarray_bound, "subarray");
          }));
  return SubarraySelectionStageConfig{
      .geometry = std::move(geometry),
      .subarray = std::move(subarray),
      .min_subarray_size = pinned.min_subarray_size,
  };
}

expected<SubarraySelectionStagePlan, Error> build_subarray_stage_plan(
    const Tensor& snapshot, SubarraySelectionStageConfig stage,
    int64_t max_items, int64_t max_snapshots) {
  const auto& shape = snapshot.shape();
  if (shape.size() != 2U && shape.size() != 3U) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot must have shape (num_items, num_channels) or "
            "(num_items, num_channels, num_snapshots)"});
  }
  try {
    stage.geometry.validate();
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, e.what()});
  }

  const int64_t num_channels = shape[1];
  if (num_channels != stage.geometry.num_elements()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot num_channels must match "
            "array_geometry.num_elements"});
  }
  const int64_t num_snapshots = shape.size() == 3U ? shape[2] : 1;
  if ((max_items > 0 && shape[0] > max_items) ||
      (max_snapshots > 0 && num_snapshots > max_snapshots)) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": snapshot shape exceeds MaxTensorExtents"});
  }
  return SubarraySelectionStagePlan{
      .stage = std::move(stage),
      .num_items = shape[0],
      .num_channels = num_channels,
      .num_snapshots = num_snapshots,
      .has_snapshot_axis = shape.size() == 3U,
  };
}

bool selection_cache_hit(const detail::SubarraySelectionState& state,
                         const SubarraySelectionStageConfig& stage) {
  return state.has_selection && state.geometry_key == stage.geometry &&
         state.subarray_key == stage.subarray &&
         state.min_subarray_size_key == stage.min_subarray_size;
}

expected<const SubarraySelection*, Error> prepare_selection(
    detail::SubarraySelectionState& state,
    const SubarraySelectionStageConfig& stage, int64_t max_subarray_channels,
    cudaStream_t stream) {
  if (selection_cache_hit(state, stage)) {
    return &state.selection;
  }

  SubarraySelection selection;
  try {
    selection = build_subarray_selection(
        stage.geometry, stage.subarray, stage.min_subarray_size);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, e.what()});
  }
  if (max_subarray_channels > 0 &&
      selection.subarray_size > max_subarray_channels) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": selected channels exceed "
            "MaxTensorExtents.max_subarray_channels"});
  }

  if (selection.reduced) {
    if (!state.channel_index ||
        state.channel_index_capacity < selection.subarray_size) {
      state.channel_index =
          make_device_tensor({selection.subarray_size},
                             DLDataType{kDLInt, 32U, 1U});
      state.channel_index_capacity = selection.subarray_size;
    }
    HOLOSCAN_TRY(auto index_ptr,
                 writable_ptr(*state.channel_index, "channel_index"));
    if (cudaMemcpyAsync(
            index_ptr, selection.channel_index.data(),
            selection.channel_index.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          std::string(kOpName) + ": channel-index H2D failed"});
    }
    HOLOSCAN_LOG_INFO(
        "{}: selected {}/{} channels (uniform sub-array)", kOpName,
        selection.subarray_size, selection.num_channels);
  }

  state.geometry_key = stage.geometry;
  state.subarray_key = stage.subarray;
  state.min_subarray_size_key = stage.min_subarray_size;
  state.selection = std::move(selection);
  state.has_selection = true;
  return &state.selection;
}

expected<void, Error> validate_optional_mask(
    const std::optional<Tensor>& maybe_valid,
    const SubarraySelectionStagePlan& plan) {
  if (!maybe_valid.has_value()) {
    return {};
  }
  const DLDataType dtype = maybe_valid->dtype();
  const bool dtype_ok =
      dtype.lanes == 1 && dtype.bits == 8 &&
      (dtype.code == kDLBool || dtype.code == kDLUInt);
  if (!dtype_ok) {
    return fail_invalid(
        "SubarraySelectionOp: is_valid must have dtype bool or uint8");
  }
  const auto& shape = maybe_valid->shape();
  if (shape.size() != 2U || shape[0] != plan.num_items ||
      shape[1] != plan.num_snapshots) {
    return fail_invalid(
        "SubarraySelectionOp: is_valid must have shape (num_items, "
        "num_snapshots) matching snapshot");
  }
  try {
    require_cuda_accessible(*maybe_valid, "is_valid", kOpName);
    require_contiguous(*maybe_valid, "is_valid", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  return {};
}

template <typename T>
expected<TensorOutputLoan, Error> prepare_snapshot_output(
    Output<Tensor>& output, ExecutionContext& context, const Tensor& snapshot,
    const SubarraySelectionStagePlan& plan,
    const SubarraySelection& selection,
    const std::shared_ptr<holoscan::Tensor>& channel_index,
    std::span<const int64_t> output_shape, cudaStream_t stream) {
  const DLDataType dtype = complex_dtype<T>();
  if (!selection.reduced) {
    const std::size_t expected_bytes =
        tensor_byte_span(output_shape, dtype);
    return prepare_device_tensor(
        output, context, output_shape, dtype,
        [&](void* destination,
            std::size_t bytes) -> expected<void, Error> {
          if (bytes != expected_bytes) {
            return fail_runtime(
                "SubarraySelectionOp: snapshot pass-through size mismatch");
          }
          if (cudaMemcpyAsync(destination, snapshot.data(), bytes,
                              cudaMemcpyDeviceToDevice, stream) !=
              cudaSuccess) {
            return fail_runtime(
                "SubarraySelectionOp: snapshot pass-through D2D failed");
          }
          return {};
        });
  }

  return prepare_device_tensor(
      output, context, output_shape, dtype,
      [&](void* destination, std::size_t) -> expected<void, Error> {
        try {
          run_spatial_smoothing_gather<T>(
              static_cast<const cuda::std::complex<T>*>(snapshot.data()),
              /*mask=*/nullptr, plan.num_items, plan.num_channels,
              plan.num_snapshots,
              static_cast<const int32_t*>(channel_index->data()),
              /*fb_flip=*/nullptr, /*num_subarrays=*/1,
              selection.subarray_size, /*forward_backward=*/false,
              static_cast<cuda::std::complex<T>*>(destination),
              /*ss_mask=*/nullptr, stream);
          check_cuda("subarray gather", kOpName);
        } catch (const std::runtime_error& e) {
          return fail_runtime(e.what());
        }
        return {};
      });
}

expected<TensorOutputLoan, Error> prepare_mask_output(
    Output<Tensor>& output, ExecutionContext& context, const Tensor& mask,
    cudaStream_t stream) {
  const std::size_t expected_bytes =
      tensor_byte_span(mask.shape(), mask.dtype());
  return prepare_device_tensor(
      output, context, mask.shape(), mask.dtype(),
      [&](void* destination, std::size_t bytes) -> expected<void, Error> {
        if (bytes != expected_bytes) {
          return fail_runtime(
              "SubarraySelectionOp: is_valid forward size mismatch");
        }
        if (cudaMemcpyAsync(destination, mask.data(), bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
          return fail_runtime(
              "SubarraySelectionOp: is_valid forward D2D failed");
        }
        return {};
      });
}

}  // namespace

template <typename T>
SubarraySelectionOp<T>::SubarraySelectionOp(
    const MaxTensorExtents& extents)
    : state_(std::make_shared<detail::SubarraySelectionState>()) {  // NOLINT
  if (extents.max_subarray_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_subarray_channels, "max_subarray_channels",
        "SubarraySelectionOp");
  } else if (extents.max_virtual_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_virtual_channels, "max_virtual_channels",
        "SubarraySelectionOp");
  } else if (extents.max_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_channels, "max_channels", "SubarraySelectionOp");
  }
  if (!extents.max_items.has_value() &&
      !extents.max_subarray_channels.has_value() &&
      !extents.max_snapshots.has_value()) {
    return;  // Python / non-pooled: leave byte spans at 0.
  }
  const char* who = "SubarraySelectionOp";
  const std::int64_t max_items =
      require_extent(extents.max_items,
                     "max_items", who);
  const std::int64_t max_sub =
      require_extent(extents.max_subarray_channels,
                     "max_subarray_channels", who);
  const std::int64_t max_snap =
      require_extent(extents.max_snapshots,
                     "max_snapshots", who);
  max_items_ = max_items;
  max_snapshots_ = max_snap;
  snapshot_max_byte_span_ = byte_span_complex_cube<T>(
      {max_items, max_sub, max_snap}, who);
  geom_max_byte_span_ = byte_span_elems(
      product_extents({max_sub + 1, 3}, who),
      sizeof(double), who);
  is_valid_max_byte_span_ = byte_span_elems(
      product_extents({max_items, max_snap}, who), 1U, who);
  state_->channel_index =
      make_device_tensor({max_sub}, DLDataType{kDLInt, 32U, 1U});
  state_->channel_index_capacity = max_sub;
}

template <typename T>
void SubarraySelectionOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kHostGeometry{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 64U, 1U},
      .rank = 2U,
  };
  constexpr TensorRepresentation kHostIndexVector{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLInt, 64U, 1U},
      .rank = 1U,
  };
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  // Optional is_valid joins the trigger tick when connected.
  spec.input(is_valid)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kDevice));
  spec.output(snapshot_out)
      .produces_tensor(pooled_tensor_output(
          kDevice, 3U, snapshot_max_byte_span_));
  spec.output(array_geometry)
      .max_emits_per_compute(1U)
      .produces_tensor(pooled_tensor_output(
          kHostGeometry, 2U, geom_max_byte_span_));
  spec.output(is_valid_out)
      .produces_tensor(pooled_tensor_output(
          kDevice, 2U, is_valid_max_byte_span_));
  // [live] Exact overrides, Params order (schema and host Tensor).
  spec.input(array_geometry_in).optional().queue_depth(2U);
  spec.input(subarray)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(kHostIndexVector));
}

template <typename T>
Contract SubarraySelectionOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{snapshot});
  result.input(is_valid, Exact{}).on_missing(is_valid, SafeDefault{});
  add_live_override_to_contract(result, array_geometry_in);
  add_live_override_to_contract(result, subarray);
  return result;
}

template <typename T>
expected<void, Error> SubarraySelectionOp<T>::compute(
    ExecutionContext& context) {
  // Receive and classify the trigger tensor.
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));
  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_complex<T>(snap, "snapshot");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Merge Exact overrides into one tick-local config, then derive geometry.
  const SubarraySelectionParams& pinned =
      context.params<SubarraySelectionParams>();
  const int64_t live_subarray_bound =
      subarray_max_extent_ > 0
          ? subarray_max_extent_
          : (snap.shape().size() > 1U ? snap.shape()[1] : 0);
  HOLOSCAN_TRY(
      auto stage,
      resolve_params(array_geometry_in, subarray, pinned,
                     live_subarray_bound));
  HOLOSCAN_TRY(
      auto plan,
      build_subarray_stage_plan(
          snap, std::move(stage), max_items_, max_snapshots_));

  // Validate the optional mask against this tick's named axes.
  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  HOLOSCAN_TRY(validate_optional_mask(maybe_valid, plan));

  // Resolve the runtime stream and cached host/device selection artifacts.
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "SubarraySelectionOp: no CUDA compute stream"});
  }
  HOLOSCAN_TRY(
      const auto* selection,
      prepare_selection(*state_, plan.stage, subarray_max_extent_, stream));
  const ArrayGeometry& out_geometry =
      selection->reduced ? selection->reduced_geometry : plan.stage.geometry;
  const int64_t out_channels =
      selection->reduced ? selection->subarray_size : plan.num_channels;
  const std::vector<int64_t> out_shape =
      plan.has_snapshot_axis
          ? std::vector<int64_t>{
                plan.num_items, out_channels, plan.num_snapshots}
          : std::vector<int64_t>{plan.num_items, out_channels};

  // Prepare the required selected snapshot before any publication.
  HOLOSCAN_TRY(
      auto snap_loan,
      prepare_snapshot_output<T>(
          snapshot_out, context, snap, plan, *selection,
          state_->channel_index, out_shape, stream));

  // Publish the geometry sideband before the Tensor that triggers downstream
  // Exact joins. An unwired optional is absorbed before snapshot publication.
  if (array_geometry_out_.active()) {
    HOLOSCAN_TRY(
        auto prepared,
        prepare_wire_array_geometry(array_geometry, out_geometry));
    array_geometry_out_.emit_schema_tensor(
        array_geometry, context, std::move(prepared.loan),
        std::move(prepared.payload),
        "SubarraySelectionOp.array_geometry");
  }

  HOLOSCAN_TRY(
      emit_prepared_tensor(snapshot_out, context, std::move(snap_loan)));

  // is_valid_out is not a selector for snapshot_out and remains independent.
  if (maybe_valid.has_value() && is_valid_out_.active()) {
    HOLOSCAN_TRY(
        auto prepared,
        prepare_mask_output(is_valid_out, context, *maybe_valid, stream));
    is_valid_out_.emit(is_valid_out, context, std::move(prepared),
                       "SubarraySelectionOp.is_valid_out");
    // Keeps the D2D source alive until the copy enqueued above retires.
    retained_is_valid_ = std::move(*maybe_valid);
  } else if (!maybe_valid.has_value()) {
    retained_is_valid_ = {};
  }

  // Keep the trigger allocation alive until the next activation.
  retained_snapshot_ = std::move(snap);
  return {};
}

template class SubarraySelectionOp<float>;
template class SubarraySelectionOp<double>;

}  // namespace holoscan::radar
