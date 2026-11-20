// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace detail {

struct SnapshotGatherState {
  std::shared_ptr<holoscan::Tensor> leading_dims;
  std::shared_ptr<holoscan::Tensor> batch_linear;
  std::shared_ptr<holoscan::Tensor> error_flag;
  int64_t batch_linear_capacity{0};
};

}  // namespace detail

namespace {

constexpr const char* kOpName = "SnapshotGatherOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
// is_valid is emitted as uint8 (0/1), not kDLBool: Holoscan's DLPack message
// path cannot map kDLBool to a NumPy typestr (a bool port throws in Python).
// Every is_valid consumer accepts uint8 and treats nonzero as set.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

std::array<int64_t, 2> normalize_snapshot_window(const std::vector<int64_t>& window) {
  std::array<int64_t, 2> normalized{};
  if (window.empty()) {
    normalized = {0, 0};
  } else if (window.size() == 1) {
    normalized = {window[0], window[0]};
  } else if (window.size() == 2) {
    normalized = {window[0], window[1]};
  } else {
    throw std::invalid_argument(
        "SnapshotGatherOp: snapshot_window must have exactly 2 entries "
        "(doppler, range)");
  }
  if (normalized[0] < 0 || normalized[1] < 0) {
    throw std::invalid_argument("SnapshotGatherOp: snapshot_window half-extents must be >= 0");
  }
  return normalized;
}

struct SnapshotGatherStageConfig {
  int64_t doppler_half{0};
  int64_t range_half{0};
  SnapshotEdgeMode doppler_edge{SnapshotEdgeMode::kWrap};
  SnapshotEdgeMode range_edge{SnapshotEdgeMode::kShift};
};

struct SnapshotGatherStagePlan {
  SnapshotGatherStageConfig stage;
  int64_t num_detections{0};
  int64_t idx_cols{0};
  int64_t n_preserved{0};
  int64_t num_channels{0};
  int64_t num_doppler{0};
  int64_t num_range{0};
  int64_t num_snapshots{0};
  int64_t slice_elems{0};
  std::vector<int64_t> leading_dims;
};

expected<SnapshotGatherStageConfig, Error> resolve_params(
    Input<SnapshotWindowOverride>& window_port, const SnapshotGatherParams& pinned) {
  SnapshotWindowOverride seed;
  try {
    const auto normalized = normalize_snapshot_window(pinned.snapshot_window);
    seed = {normalized[0], normalized[1]};
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  HOLOSCAN_TRY(auto window, resolve_live_override(window_port, seed));
  try {
    const auto normalized = normalize_snapshot_window({window.doppler, window.range});
    return SnapshotGatherStageConfig{
        .doppler_half = normalized[0],
        .range_half = normalized[1],
        .doppler_edge = pinned.doppler_edge,
        .range_edge = pinned.range_edge,
    };
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
}

expected<SnapshotGatherStagePlan, Error> build_snapshot_gather_stage_plan(
    const Tensor& rd, const Tensor& idx, SnapshotGatherStageConfig stage, int64_t max_detections,
    int64_t max_channels, int64_t max_snapshots) {
  const auto& idx_shape = idx.shape();
  if (idx_shape.size() != 2U || idx_shape[1] < 2) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": detection_idx must have shape "
                                                        "(num_detections, 2 [+ n_batch])"});
  }
  const int64_t num_detections = idx_shape[0];
  if (max_detections > 0 && num_detections > max_detections) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": detection_idx rows (" +
                                     std::to_string(num_detections) +
                                     ") exceed MaxTensorExtents max_detections/max_items (" +
                                     std::to_string(max_detections) + ")"});
  }
  const int64_t idx_cols = idx_shape[1];
  const int64_t n_preserved = idx_cols - 2;
  if (n_preserved > detail::kMaxSnapshotPreservedBatchDims) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": too many preserved batch dims in detection_idx"});
  }

  const auto& rd_shape = rd.shape();
  if (rd_shape.size() < 3U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": rd_maps must end with "
                                                        "(num_channels, num_doppler, num_range)"});
  }
  const int64_t num_range = rd_shape[rd_shape.size() - 1U];
  const int64_t num_doppler = rd_shape[rd_shape.size() - 2U];
  const int64_t num_channels = rd_shape[rd_shape.size() - 3U];
  if (num_channels <= 0 || num_doppler <= 0 || num_range <= 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": rd_maps trailing axes must be positive"});
  }
  std::vector<int64_t> leading_dims(rd_shape.begin(), rd_shape.end() - 3);
  if (static_cast<int64_t>(leading_dims.size()) != n_preserved) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                               ": rd_maps leading rank must match detection_idx "
                                               "preserved batch columns"});
  }
  if (max_channels > 0 && num_channels > max_channels) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                                                  ": rd_maps num_channels exceeds "
                                                                  "MaxTensorExtents.max_channels"});
  }

  constexpr int64_t kMaxHalf = (std::numeric_limits<int64_t>::max() - 1) / 2;
  if (stage.doppler_half > kMaxHalf || stage.range_half > kMaxHalf) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": snapshot_window is too large"});
  }
  const int64_t doppler_len = 2 * stage.doppler_half + 1;
  const int64_t range_len = 2 * stage.range_half + 1;
  if ((stage.doppler_edge == SnapshotEdgeMode::kShift && doppler_len > num_doppler) ||
      (stage.range_edge == SnapshotEdgeMode::kShift && range_len > num_range)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": shift window length must not exceed its rd_maps axis"});
  }
  if (doppler_len > std::numeric_limits<int64_t>::max() / range_len) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": snapshot_window product is too large"});
  }
  const int64_t num_snapshots = doppler_len * range_len;
  if (max_snapshots > 0 && num_snapshots > max_snapshots) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 std::string(kOpName) + ": snapshot count exceeds "
                                                        "MaxTensorExtents.max_snapshots"});
  }
  return SnapshotGatherStagePlan{
      .stage = stage,
      .num_detections = num_detections,
      .idx_cols = idx_cols,
      .n_preserved = n_preserved,
      .num_channels = num_channels,
      .num_doppler = num_doppler,
      .num_range = num_range,
      .num_snapshots = num_snapshots,
      .slice_elems = num_channels * num_doppler * num_range,
      .leading_dims = std::move(leading_dims),
  };
}

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor, const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              std::string(kOpName) + ": " + what + " is not writable: " + mut.error().message});
  }
  return *mut;
}

expected<void, Error> prepare_gather_workspace(detail::SnapshotGatherState& state,
                                               const SnapshotGatherStagePlan& plan,
                                               const Tensor& idx, cudaStream_t stream) {
  if (state.batch_linear_capacity < plan.num_detections) {
    state.batch_linear = make_device_tensor({plan.num_detections}, kInt64Dtype);
    state.batch_linear_capacity = plan.num_detections;
  }
  if (plan.n_preserved > 0) {
    HOLOSCAN_TRY(auto leading_ptr, writable_ptr(*state.leading_dims, "leading_dims"));
    if (cudaMemcpyAsync(leading_ptr, plan.leading_dims.data(),
                        plan.leading_dims.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return fail_runtime("SnapshotGatherOp: leading-dims H2D copy failed");
    }
  }

  HOLOSCAN_TRY(auto batch_ptr, writable_ptr(*state.batch_linear, "batch_linear"));
  HOLOSCAN_TRY(auto error_ptr, writable_ptr(*state.error_flag, "error_flag"));
  if (cudaMemsetAsync(error_ptr, 0, sizeof(int), stream) != cudaSuccess) {
    return fail_runtime("SnapshotGatherOp: error-flag clear failed");
  }
  try {
    detail::launch_batch_linear(
        static_cast<const int64_t*>(idx.data()), plan.num_detections, plan.idx_cols,
        plan.n_preserved,
        plan.n_preserved > 0 ? static_cast<const int64_t*>(state.leading_dims->data()) : nullptr,
        plan.num_doppler, plan.num_range, static_cast<int64_t*>(batch_ptr),
        static_cast<int*>(error_ptr), stream);
    check_cuda("batch linear kernel", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  int host_error = 0;
  if (cudaMemcpyAsync(&host_error, state.error_flag->data(), sizeof(int), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess) {
    return fail_runtime("SnapshotGatherOp: error-flag D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return fail_runtime("SnapshotGatherOp: stream sync failed");
  }
  if (host_error == detail::kSnapshotErrBatchCoord) {
    return fail_invalid(
        "SnapshotGatherOp: detection_idx preserved-batch coordinate out of "
        "range for rd_maps leading dims");
  }
  if (host_error == detail::kSnapshotErrCellCoord) {
    return fail_invalid(
        "SnapshotGatherOp: detection_idx doppler/range coordinate out of "
        "range for rd_maps");
  }
  return {};
}

template <typename T>
expected<void, Error> run_snapshot_gather(const Tensor& rd, const Tensor& idx,
                                          const SnapshotGatherStagePlan& plan,
                                          const detail::SnapshotGatherState& state,
                                          void* snapshot_ptr, void* valid_ptr,
                                          std::size_t snapshot_bytes, cudaStream_t stream) {
  (void)snapshot_bytes;
  try {
    detail::launch_gather_snapshots<T>(
        static_cast<const cuda::std::complex<T>*>(rd.data()),
        static_cast<const int64_t*>(idx.data()), plan.num_detections, plan.idx_cols,
        plan.n_preserved, static_cast<const int64_t*>(state.batch_linear->data()),
        plan.num_channels, plan.num_doppler, plan.num_range, plan.stage.doppler_half,
        plan.stage.range_half, plan.stage.doppler_edge, plan.stage.range_edge, plan.num_snapshots,
        plan.slice_elems, static_cast<cuda::std::complex<T>*>(snapshot_ptr),
        static_cast<uint8_t*>(valid_ptr), stream);
    check_cuda("gather snapshots kernel", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }
  return {};
}

}  // namespace

template <typename T>
SnapshotGatherOp<T>::SnapshotGatherOp(const MaxTensorExtents& extents)
: state_(std::make_shared<detail::SnapshotGatherState>()) {  // NOLINT
  state_->leading_dims = make_device_tensor({detail::kMaxSnapshotPreservedBatchDims}, kInt64Dtype);
  state_->error_flag = make_device_tensor({1}, kInt32Dtype);

  const bool has_det = extents.max_detections.has_value() || extents.max_items.has_value();
  if (!has_det && !extents.max_channels.has_value() && !extents.max_snapshots.has_value()) {
    return;  // Python / non-pooled: leave byte spans at 0.
  }
  const char* who = "SnapshotGatherOp";
  const std::int64_t max_det = extents.max_detections.has_value()
                                   ? require_extent(extents.max_detections, "max_detections", who)
                                   : require_extent(extents.max_items, "max_items", who);
  const std::int64_t max_ch = require_extent(extents.max_channels, "max_channels", who);
  const std::int64_t max_snap = require_extent(extents.max_snapshots, "max_snapshots", who);
  max_detections_ = max_det;
  max_channels_ = max_ch;
  max_snapshots_ = max_snap;
  state_->batch_linear = make_device_tensor({max_det}, kInt64Dtype);
  state_->batch_linear_capacity = max_det;
  snapshot_max_byte_span_ = byte_span_complex_cube<T>({max_det, max_ch, max_snap}, who);
  is_valid_max_byte_span_ = byte_span_elems(product_extents({max_det, max_snap}, who), 1U, who);
}

template <typename T>
void SnapshotGatherOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,  // rank-agnostic batching
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(snapshot).produces_tensor(pooled_tensor_output(kDevice, 3U, snapshot_max_byte_span_));
  spec.output(is_valid).produces_tensor(pooled_tensor_output(kDevice, 2U, is_valid_max_byte_span_));

  // [live] Exact overrides, Params order.
  spec.input(snapshot_window).optional().queue_depth(2U);
}

template <typename T>
Contract SnapshotGatherOp<T>::contract() const {
  Contract result;
  result
      .trigger(
          OnAligned{.ports = ports(rd_maps, detection_idx), .window = std::chrono::nanoseconds{0}})
      .input(rd_maps, Exact{})
      .input(detection_idx, Exact{});
  add_live_override_to_contract(result, snapshot_window);
  return result;
}

template <typename T>
expected<void, Error> SnapshotGatherOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive and classify the two aligned data-plane tensors.
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kOpName));
  try {
    require_complex<T>(rd, "rd_maps");
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_cuda_accessible(idx, "detection_idx", kOpName);
    require_contiguous(idx, "detection_idx", kOpName);
    require_dtype(idx, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Merge the Exact live window onto Params, then validate this tick's
  // named axes and derive output geometry.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  HOLOSCAN_TRY(auto stage, resolve_params(snapshot_window, context.params<SnapshotGatherParams>()));
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan,
               build_snapshot_gather_stage_plan(rd, idx, std::move(stage), max_detections_,
                                                max_channels_, max_snapshots_));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "SnapshotGatherOp: no CUDA compute stream"});
  }

  const std::vector<int64_t> snap_shape = {plan.num_detections, plan.num_channels,
                                           plan.num_snapshots};
  const std::vector<int64_t> valid_shape = {plan.num_detections, plan.num_snapshots};
  const DLDataType snap_dtype = complex_dtype<T>();

  // Allocate exact-sized outputs before launching device work.
  if (plan.num_detections == 0) {
    // Prepare both loans before any publish so a late allocate/commit
    // failure cannot leave a sibling port already downstream.
    auto noop = [](void*, std::size_t) -> expected<void, Error> {
      return {};
    };
    HOLOSCAN_TRY(auto snap_loan,
                 prepare_device_tensor(snapshot, context, snap_shape, snap_dtype, noop));
    HOLOSCAN_TRY(auto valid_loan,
                 prepare_device_tensor(is_valid, context, valid_shape, kMaskDtype, noop));
    // Publish the optional sideband first so a downstream Exact selector sees
    // it when snapshot triggers. A declined optional emit remains harmless.
    // Publish coherent outputs in dependency order and retain borrowed inputs.
    (void)emit_prepared_tensor(is_valid, context, std::move(valid_loan));
    HOLOSCAN_TRY(emit_prepared_tensor(snapshot, context, std::move(snap_loan)));
    retained_rd_maps_ = std::move(rd);
    retained_detection_idx_ = std::move(idx);
    return {};
  }

  // Prepare reusable validation/gather workspace before borrowing outputs.
  HOLOSCAN_TRY(prepare_gather_workspace(*state_, plan, idx, stream));

  // One kernel fills the required snapshot and optional/always mask. Prepare
  // both loans before publication so downstream cannot observe half a pair.
  auto snap_loan =
      snapshot.allocate_tensor(TensorLoanRequest{.shape = snap_shape, .dtype = snap_dtype});
  if (!snap_loan) {
    return make_unexpected(std::move(snap_loan).error());
  }
  auto valid_loan =
      is_valid.allocate_tensor(TensorLoanRequest{.shape = valid_shape, .dtype = kMaskDtype});
  if (!valid_loan) {
    return make_unexpected(std::move(valid_loan).error());
  }

  auto snap_writer = snap_loan->write(context.cuda_stream());
  if (!snap_writer) {
    return make_unexpected(std::move(snap_writer).error());
  }
  auto valid_writer = valid_loan->write(context.cuda_stream());
  if (!valid_writer) {
    return make_unexpected(std::move(valid_writer).error());
  }

  HOLOSCAN_TRY(run_snapshot_gather<T>(rd, idx, plan, *state_, snap_writer->data(),
                                      valid_writer->data(), snap_writer->byte_size(), stream));

  auto snap_committed = std::move(*snap_writer).commit();
  if (!snap_committed) {
    return make_unexpected(std::move(snap_committed).error());
  }
  auto valid_committed = std::move(*valid_writer).commit();
  if (!valid_committed) {
    return make_unexpected(std::move(valid_committed).error());
  }

  // Publish the optional sideband before the triggering snapshot for Exact
  // downstream coherence. A declined optional emit cannot suppress snapshot.
  (void)emit_prepared_tensor(is_valid, context, std::move(*valid_loan));
  HOLOSCAN_TRY(emit_prepared_tensor(snapshot, context, std::move(*snap_loan)));

  // Retain source buffers until the next tick after asynchronous reads.
  retained_rd_maps_ = std::move(rd);
  retained_detection_idx_ = std::move(idx);
  return {};
}

template class SnapshotGatherOp<float>;
template class SnapshotGatherOp<double>;

}  // namespace holoscan::radar
