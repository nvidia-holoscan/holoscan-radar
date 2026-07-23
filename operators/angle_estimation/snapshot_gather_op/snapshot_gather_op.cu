// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "angle_estimation/snapshot_gather_op/snapshot_gather_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "SnapshotGatherOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
// is_valid is emitted as uint8 (0/1), not kDLBool: HSDK 4.3's DLPack message
// path cannot map kDLBool to a NumPy typestr (a bool port throws in Python).
// The buffer holds 0/1 bytes (written as bool internally); uint8 is byte-
// identical, NumPy-visible, and every is_valid consumer treats nonzero as set.
constexpr DLDataType kMaskDtype{kDLUInt, 8, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

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

}  // namespace

template <typename T>
void SnapshotGatherOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void SnapshotGatherOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.input<holoscan::TensorMap>("detection_idx");

  spec.output<holoscan::TensorMap>("snapshot");
  spec.output<holoscan::TensorMap>("is_valid");

  spec.param(snapshot_window_, "snapshot_window", "Snapshot window half-extent",
             "Per-axis (doppler, range) neighborhood half-size around each "
             "detection",
             {static_cast<int64_t>(0), static_cast<int64_t>(0)});
  add_override_port<std::vector<int64_t>>(spec, "snapshot_window");

  spec.param(doppler_edge_, "doppler_edge", "Doppler edge mode",
             "Neighborhood behavior at the Doppler-axis edges",
             SnapshotEdgeMode::kWrap);
  spec.param(range_edge_, "range_edge", "Range edge mode",
             "Neighborhood behavior at the range-axis edges",
             SnapshotEdgeMode::kShift);
}

template <typename T>
void SnapshotGatherOp<T>::compute(InputContext& op_input,
                                  OutputContext& op_output, ExecutionContext&) {
  if (!seeded_) {
    effective_snapshot_window_ =
        normalize_snapshot_window(snapshot_window_.get());
    seeded_ = true;
  }
  bool window_changed = false;
  effective_snapshot_window_ = resolve_override_param(
      op_input, "snapshot_window", effective_snapshot_window_, &window_changed);
  effective_snapshot_window_ =
      normalize_snapshot_window(effective_snapshot_window_);

  const SnapshotEdgeMode d_edge = doppler_edge_.get();
  const SnapshotEdgeMode r_edge = range_edge_.get();
  const int64_t d_half = effective_snapshot_window_[0];
  const int64_t r_half = effective_snapshot_window_[1];
  const int64_t num_snapshots =
      num_snapshots_from_window(effective_snapshot_window_);

  auto rd_maps = receive_single(op_input, "rd_maps", kOpName);
  require_cuda_accessible(*rd_maps, "rd_maps", kOpName);
  require_contiguous(*rd_maps, "rd_maps", kOpName);
  const InputPrecision precision =
      classify_input_dtype<T>(*rd_maps, "rd_maps", false);
  if (precision != InputPrecision::kComplex) {
    throw std::invalid_argument(
        "SnapshotGatherOp: rd_maps must be complex64/complex128");
  }
  require_dtype(*rd_maps, complex_dtype<T>(), "rd_maps", kOpName);

  auto detection_idx = receive_single(op_input, "detection_idx", kOpName);
  require_cuda_accessible(*detection_idx, "detection_idx", kOpName);
  require_contiguous(*detection_idx, "detection_idx", kOpName);
  require_dtype(*detection_idx, kInt64Dtype, "detection_idx", kOpName);

  const auto& idx_shape = detection_idx->shape();
  if (idx_shape.size() != 2 || idx_shape[1] < 2) {
    throw std::invalid_argument(
        "SnapshotGatherOp: detection_idx must have shape "
        "(num_detections, 2 [+ n_batch])");
  }
  const int64_t num_detections = idx_shape[0];
  const int64_t idx_cols = idx_shape[1];
  const int64_t n_preserved = idx_cols - 2;

  const auto& rshape = rd_maps->shape();
  if (rshape.size() < 3) {
    throw std::invalid_argument(
        "SnapshotGatherOp: rd_maps must end with "
        "(num_channels, num_doppler, num_range)");
  }
  const int64_t num_range = rshape[rshape.size() - 1];
  const int64_t num_doppler = rshape[rshape.size() - 2];
  const int64_t num_channels = rshape[rshape.size() - 3];
  if (num_channels <= 0 || num_doppler <= 0 || num_range <= 0) {
    throw std::invalid_argument(
        "SnapshotGatherOp: rd_maps trailing axes must be positive");
  }

  std::vector<int64_t> leading_dims(rshape.begin(), rshape.end() - 3);
  if (static_cast<int64_t>(leading_dims.size()) != n_preserved) {
    throw std::invalid_argument(
        "SnapshotGatherOp: rd_maps leading rank must match detection_idx "
        "preserved batch columns");
  }
  if (n_preserved > detail::kMaxSnapshotPreservedBatchDims) {
    throw std::invalid_argument(
        "SnapshotGatherOp: too many preserved batch dims in detection_idx");
  }

  int64_t batch_in = 1;
  for (int64_t d : leading_dims) {
    batch_in *= d;
  }
  const int64_t slice_elems = num_channels * num_doppler * num_range;

  auto stream = op_input.receive_cuda_stream("rd_maps");

  auto snapshot = make_device_tensor(
      {num_detections, num_channels, num_snapshots}, complex_dtype<T>());
  auto is_valid =
      make_device_tensor({num_detections, num_snapshots}, kMaskDtype);

  if (num_detections == 0) {
    if (stream != cudaStreamDefault) {
      op_output.set_cuda_stream(stream, "snapshot");
      op_output.set_cuda_stream(stream, "is_valid");
    }
    emit_single(op_output, "snapshot", snapshot);
    emit_single(op_output, "is_valid", is_valid);
    retained_rd_maps_ = std::move(rd_maps);
    retained_detection_idx_ = std::move(detection_idx);
    return;
  }

  if (!retained_leading_dims_ ||
      retained_leading_dims_->shape()[0] != n_preserved) {
    retained_leading_dims_ = make_device_tensor({n_preserved}, kInt64Dtype);
  }
  if (n_preserved > 0) {
    if (cudaMemcpyAsync(retained_leading_dims_->data(), leading_dims.data(),
                        leading_dims.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(
          "SnapshotGatherOp: leading-dims H2D copy failed");
    }
  }

  auto batch_linear = make_device_tensor({num_detections}, kInt64Dtype);
  auto error_flag = make_device_tensor({1}, kInt32Dtype);
  if (cudaMemsetAsync(error_flag->data(), 0, sizeof(int), stream) !=
      cudaSuccess) {
    throw std::runtime_error("SnapshotGatherOp: error-flag clear failed");
  }

  detail::launch_batch_linear(
      static_cast<const int64_t*>(detection_idx->data()), num_detections,
      idx_cols, n_preserved,
      n_preserved > 0
          ? static_cast<const int64_t*>(retained_leading_dims_->data())
          : nullptr,
      static_cast<int64_t*>(batch_linear->data()),
      static_cast<int*>(error_flag->data()), stream);
  check_cuda("batch linear kernel", kOpName);

  int host_error = 0;
  if (cudaMemcpyAsync(&host_error, error_flag->data(), sizeof(int),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("SnapshotGatherOp: error-flag D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("SnapshotGatherOp: stream sync failed");
  }
  if (host_error == detail::kSnapshotErrBatchCoord) {
    throw std::invalid_argument(
        "SnapshotGatherOp: detection_idx preserved-batch coordinate out of "
        "range for rd_maps leading dims");
  }

  if (cudaMemsetAsync(snapshot->data(), 0,
                      static_cast<size_t>(num_detections) *
                          static_cast<size_t>(num_channels) *
                          static_cast<size_t>(num_snapshots) *
                          sizeof(cuda::std::complex<T>),
                      stream) != cudaSuccess) {
    throw std::runtime_error("SnapshotGatherOp: snapshot zero failed");
  }

  detail::launch_gather_snapshots<T>(
      static_cast<const cuda::std::complex<T>*>(rd_maps->data()),
      static_cast<const int64_t*>(detection_idx->data()), num_detections,
      idx_cols, n_preserved,
      static_cast<const int64_t*>(batch_linear->data()), num_channels,
      num_doppler, num_range, d_half, r_half, d_edge, r_edge, num_snapshots,
      slice_elems, static_cast<cuda::std::complex<T>*>(snapshot->data()),
      static_cast<bool*>(is_valid->data()), stream);
  check_cuda("gather snapshots kernel", kOpName);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "snapshot");
    op_output.set_cuda_stream(stream, "is_valid");
  }
  emit_single(op_output, "snapshot", snapshot);
  emit_single(op_output, "is_valid", is_valid);

  retained_rd_maps_ = std::move(rd_maps);
  retained_detection_idx_ = std::move(detection_idx);
  (void)window_changed;
}

template class SnapshotGatherOp<float>;
template class SnapshotGatherOp<double>;

}  // namespace holoscan::radar
