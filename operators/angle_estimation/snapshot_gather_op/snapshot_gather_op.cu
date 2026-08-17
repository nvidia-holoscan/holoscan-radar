// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_stage.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "SnapshotGatherOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
// is_valid is emitted as uint8 (0/1), not kDLBool: Holoscan's DLPack message
// path cannot map kDLBool to a NumPy typestr (a bool port throws in Python).
// The buffer holds 0/1 bytes (written as bool internally); uint8 is byte-
// identical, NumPy-visible, and every is_valid consumer treats nonzero as set.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

std::vector<int64_t> normalize_snapshot_window(
    const std::vector<int64_t>& window) {
  if (window.empty()) {
    return {0, 0};
  }
  if (window.size() == 1) {
    return {window[0], window[0]};
  }
  if (window.size() != 2) {
    throw std::invalid_argument(
        "SnapshotGatherOp: snapshot_window must have exactly 2 entries "
        "(doppler, range)");
  }
  if (window[0] < 0 || window[1] < 0) {
    throw std::invalid_argument(
        "SnapshotGatherOp: snapshot_window half-extents must be >= 0");
  }
  return window;
}

int64_t num_snapshots_from_window(const std::vector<int64_t>& window) {
  const auto w = normalize_snapshot_window(window);
  return (2 * w[0] + 1) * (2 * w[1] + 1);
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

}  // namespace

template <typename T>
SnapshotGatherOp<T>::SnapshotGatherOp(
    const MaxTensorExtents& extents)
    : extents_(extents) {
  const bool has_det = extents.max_detections.has_value() ||
                       extents.max_items.has_value();
  if (!has_det && !extents.max_channels.has_value() &&
      !extents.max_snapshots.has_value()) {
    return;  // Python / non-pooled: leave byte spans at 0.
  }
  const char* who = "SnapshotGatherOp";
  const std::int64_t max_det =
      extents.max_detections.has_value()
          ? require_extent(extents.max_detections,
                           "max_detections", who)
          : require_extent(extents.max_items,
                           "max_items", who);
  const std::int64_t max_ch =
      require_extent(extents.max_channels,
                     "max_channels", who);
  const std::int64_t max_snap =
      require_extent(extents.max_snapshots,
                     "max_snapshots", who);
  snapshot_max_byte_span_ = byte_span_complex_cube<T>(
      {max_det, max_ch, max_snap}, who);
  is_valid_max_byte_span_ = byte_span_elems(
      product_extents({max_det, max_snap}, who), 1U, who);
}

template <typename T>
void SnapshotGatherOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,  // rank-agnostic batching
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(kDevice);
  spec.input(detection_idx).queue_depth(2U).expects_tensor(kDevice);
  spec.output(snapshot)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = snapshot_max_rank_,
          .max_byte_span = snapshot_max_byte_span_});
  spec.output(is_valid)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = snapshot_max_rank_,
          .max_byte_span = is_valid_max_byte_span_});
  // [live] snapshot_window override port. Wired graphs may push window
  // samples; the operator caches the last value (sticky). Leaving the port
  // unconnected is legal — compute() seeds from Params.
  spec.input(snapshot_window).queue_depth(2U);
}

template <typename T>
Contract SnapshotGatherOp<T>::contract() const {
  // Trigger when both data tensors share an aligned event time. The [live]
  // snapshot_window input is intentionally omitted from the temporal contract
  // so an unconnected override port does not yield TEMPORAL_INPUT_UNCONNECTED,
  // and so a second OnClock source is not required (HSDK 5 EA multi-partition
  // device binding is still rough — sticky live-override coverage is
  // exercised via Params + operator sticky state when a sample does arrive).
  // [EA HSDK5 ONLY] see docs/planning/HSDK5_PORT.md §1.
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(rd_maps, detection_idx),
                         .window = std::chrono::nanoseconds{0}})
      .input(rd_maps, Exact{})
      .input(detection_idx, Exact{});
  return result;
}

template <typename T>
expected<void, Error> SnapshotGatherOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kOpName));

  const SnapshotGatherParams& pinned =
      context.params<SnapshotGatherParams>();
  // Normalize Params vector (empty→{0,0}, size1→isotropic) into the POD seed
  // for sticky resolve. Live samples arrive as SnapshotWindowOverride.
  SnapshotWindowOverride window_seed;
  try {
    const auto seeded = normalize_snapshot_window(pinned.snapshot_window);
    window_seed = SnapshotWindowOverride{seeded[0], seeded[1]};
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  // Best-effort sticky override: if the port is wired and a sample is waiting,
  // absorb it; otherwise keep sticky / fall back to Params.
  HOLOSCAN_TRY(auto window_pod,
               resolve_sticky_override(snapshot_window, window_seed,
                                       sticky_snapshot_window_));

  std::vector<int64_t> effective_window;
  try {
    effective_window =
        normalize_snapshot_window({window_pod.doppler, window_pod.range});
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const SnapshotEdgeMode d_edge = pinned.doppler_edge;
  const SnapshotEdgeMode r_edge = pinned.range_edge;
  const int64_t d_half = effective_window[0];
  const int64_t r_half = effective_window[1];
  const int64_t num_snapshots = num_snapshots_from_window(effective_window);

  try {
    classify_input_dtype<T>(rd, "rd_maps", /*allow_real=*/false);
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_dtype(rd, complex_dtype<T>(), "rd_maps", kOpName);

    require_cuda_accessible(idx, "detection_idx", kOpName);
    require_contiguous(idx, "detection_idx", kOpName);
    require_dtype(idx, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& idx_shape = idx.shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    return fail_invalid(
        "SnapshotGatherOp: detection_idx must have shape "
        "(num_detections, 2 [+ n_batch])");
  }
  const int64_t num_detections = idx_shape[0];
  const int64_t max_det_cap =
      extents_.max_detections.value_or(extents_.max_items.value_or(-1));
  if (max_det_cap >= 0 && num_detections > max_det_cap) {
    return fail_invalid(
        "SnapshotGatherOp: detection_idx rows (" +
        std::to_string(num_detections) +
        ") exceed MaxTensorExtents max_detections/max_items (" +
        std::to_string(max_det_cap) + ")");
  }
  const int64_t idx_cols = idx_shape[1];
  const int64_t n_preserved = idx_cols - 2;

  const auto& rshape = rd.shape();
  if (rshape.size() < 3) {
    return fail_invalid(
        "SnapshotGatherOp: rd_maps must end with "
        "(num_channels, num_doppler, num_range)");
  }
  const int64_t num_range = rshape[rshape.size() - 1];
  const int64_t num_doppler = rshape[rshape.size() - 2];
  const int64_t num_channels = rshape[rshape.size() - 3];
  if (num_channels <= 0 || num_doppler <= 0 || num_range <= 0) {
    return fail_invalid(
        "SnapshotGatherOp: rd_maps trailing axes must be positive");
  }

  std::vector<int64_t> leading_dims(rshape.begin(), rshape.end() - 3);
  if (static_cast<int64_t>(leading_dims.size()) != n_preserved) {
    return fail_invalid(
        "SnapshotGatherOp: rd_maps leading rank must match detection_idx "
        "preserved batch columns");
  }
  if (n_preserved > detail::kMaxSnapshotPreservedBatchDims) {
    return fail_invalid(
        "SnapshotGatherOp: too many preserved batch dims in detection_idx");
  }

  const int64_t slice_elems = num_channels * num_doppler * num_range;

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "SnapshotGatherOp: no CUDA compute stream"});
  }

  const std::vector<int64_t> snap_shape = {num_detections, num_channels,
                                           num_snapshots};
  const std::vector<int64_t> valid_shape = {num_detections, num_snapshots};
  const DLDataType snap_dtype = complex_dtype<T>();

  if (num_detections == 0) {
    // Prepare both loans before any publish so a late allocate/commit
    // failure cannot leave a sibling port already downstream.
    auto noop = [](void*, std::size_t) -> expected<void, Error> {
      return {};
    };
    HOLOSCAN_TRY(auto snap_loan,
                 prepare_device_tensor(snapshot, context, snap_shape,
                                      snap_dtype, noop));
    HOLOSCAN_TRY(auto valid_loan,
                 prepare_device_tensor(is_valid, context, valid_shape,
                                      kMaskDtype, noop));
    HOLOSCAN_TRY(
        emit_prepared_tensor(snapshot, context, std::move(snap_loan)));
    HOLOSCAN_TRY(
        emit_prepared_tensor(is_valid, context, std::move(valid_loan)));
    retained_rd_maps_ = std::move(rd);
    retained_detection_idx_ = std::move(idx);
    return {};
  }

  // Both outputs are filled by one gather kernel — allocate both loans,
  // write on the runtime stream, then commit/emit.
  auto snap_loan = snapshot.allocate_tensor(
      TensorLoanRequest{.shape = snap_shape, .dtype = snap_dtype});
  if (!snap_loan) {
    return make_unexpected(std::move(snap_loan).error());
  }
  auto valid_loan = is_valid.allocate_tensor(
      TensorLoanRequest{.shape = valid_shape, .dtype = kMaskDtype});
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

  if (!retained_leading_dims_ ||
      retained_leading_dims_->shape()[0] != n_preserved) {
    retained_leading_dims_ = make_device_tensor({n_preserved}, kInt64Dtype);
  }
  if (n_preserved > 0) {
    HOLOSCAN_TRY(auto lead_ptr,
                 writable_ptr(*retained_leading_dims_, "leading_dims"));
    if (cudaMemcpyAsync(lead_ptr, leading_dims.data(),
                        leading_dims.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return fail_runtime("SnapshotGatherOp: leading-dims H2D copy failed");
    }
  }

  auto batch_linear = make_device_tensor({num_detections}, kInt64Dtype);
  auto error_flag = make_device_tensor({1}, kInt32Dtype);
  HOLOSCAN_TRY(auto batch_ptr, writable_ptr(*batch_linear, "batch_linear"));
  HOLOSCAN_TRY(auto err_ptr, writable_ptr(*error_flag, "error_flag"));

  if (cudaMemsetAsync(err_ptr, 0, sizeof(int), stream) != cudaSuccess) {
    return fail_runtime("SnapshotGatherOp: error-flag clear failed");
  }

  try {
    detail::launch_batch_linear(
        static_cast<const int64_t*>(idx.data()), num_detections, idx_cols,
        n_preserved,
        n_preserved > 0
            ? static_cast<const int64_t*>(retained_leading_dims_->data())
            : nullptr,
        static_cast<int64_t*>(batch_ptr), static_cast<int*>(err_ptr), stream);
    check_cuda("batch linear kernel", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  // Host sync for the batch-coord error flag (validation correctness).
  int host_error = 0;
  if (cudaMemcpyAsync(&host_error, error_flag->data(), sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
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

  void* snap_ptr = snap_writer->data();
  void* valid_ptr = valid_writer->data();
  if (cudaMemsetAsync(snap_ptr, 0, snap_writer->byte_size(), stream) !=
      cudaSuccess) {
    return fail_runtime("SnapshotGatherOp: snapshot zero failed");
  }

  try {
    detail::launch_gather_snapshots<T>(
        static_cast<const cuda::std::complex<T>*>(rd.data()),
        static_cast<const int64_t*>(idx.data()), num_detections, idx_cols,
        n_preserved, static_cast<const int64_t*>(batch_linear->data()),
        num_channels, num_doppler, num_range, d_half, r_half, d_edge, r_edge,
        num_snapshots, slice_elems,
        static_cast<cuda::std::complex<T>*>(snap_ptr),
        static_cast<bool*>(valid_ptr), stream);
    check_cuda("gather snapshots kernel", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  auto snap_committed = std::move(*snap_writer).commit();
  if (!snap_committed) {
    return make_unexpected(std::move(snap_committed).error());
  }
  auto valid_committed = std::move(*valid_writer).commit();
  if (!valid_committed) {
    return make_unexpected(std::move(valid_committed).error());
  }

  auto snap_emitted = snapshot.emit(
      std::move(*snap_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!snap_emitted) {
    return make_unexpected(std::move(snap_emitted).error());
  }
  auto valid_emitted = is_valid.emit(
      std::move(*valid_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!valid_emitted) {
    return make_unexpected(std::move(valid_emitted).error());
  }

  retained_rd_maps_ = std::move(rd);
  retained_detection_idx_ = std::move(idx);
  return {};
}

template class SnapshotGatherOp<float>;
template class SnapshotGatherOp<double>;

}  // namespace holoscan::radar
