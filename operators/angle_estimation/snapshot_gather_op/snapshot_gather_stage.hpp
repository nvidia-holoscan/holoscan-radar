// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar::detail {

constexpr int kSnapshotErrBatchCoord = 1;
constexpr int kSnapshotErrCellCoord = 2;
constexpr int kMaxSnapshotPreservedBatchDims = 8;

__host__ __device__ inline int edge_mode_id(SnapshotEdgeMode mode) {
  switch (mode) {
    case SnapshotEdgeMode::kWrap:
      return 0;
    case SnapshotEdgeMode::kClamp:
      return 1;
    case SnapshotEdgeMode::kShift:
      return 2;
  }
  return 0;
}

__device__ inline int64_t shift_window_start(int64_t center, int64_t half, int64_t n) {
  const int64_t win_len = 2 * half + 1;
  int64_t start = center - half;
  if (start < 0) {
    start = 0;
  }
  int64_t end = start + win_len - 1;
  if (end >= n) {
    end = n - 1;
    start = end - win_len + 1;
    if (start < 0) {
      start = 0;
    }
  }
  return start;
}

__device__ inline int64_t resolve_axis_index(int64_t center, int64_t offset, int64_t half,
                                             int64_t n, int edge_mode, bool* valid) {
  if (edge_mode == edge_mode_id(SnapshotEdgeMode::kWrap)) {
    int64_t idx = center + offset;
    idx = ((idx % n) + n) % n;
    *valid = true;
    return idx;
  }
  if (edge_mode == edge_mode_id(SnapshotEdgeMode::kShift)) {
    const int64_t start = shift_window_start(center, half, n);
    const int64_t k = offset + half;
    const int64_t idx = start + k;
    *valid = true;
    return idx;
  }
  const int64_t idx = center + offset;
  if (idx < 0 || idx >= n) {
    *valid = false;
    return 0;
  }
  *valid = true;
  return idx;
}

static __global__ void batch_linear_kernel(
    const int64_t* __restrict__ detection_idx, int64_t num_detections, int64_t idx_cols,
    int64_t n_preserved, const int64_t* __restrict__ leading_dims, int64_t num_doppler,
    int64_t num_range, int64_t* __restrict__ batch_linear, int* __restrict__ error_flag) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_detections) {
    return;
  }
  const int64_t* idx_row = detection_idx + row * idx_cols;
  const int64_t doppler = idx_row[n_preserved];
  const int64_t range = idx_row[n_preserved + 1];
  if (doppler < 0 || doppler >= num_doppler || range < 0 || range >= num_range) {
    atomicMax(error_flag, kSnapshotErrCellCoord);
    return;
  }
  int64_t linear = 0;
  for (int64_t a = 0; a < n_preserved; ++a) {
    const int64_t coord = idx_row[a];
    const int64_t dim = leading_dims[a];
    if (coord < 0 || coord >= dim) {
      atomicMax(error_flag, kSnapshotErrBatchCoord);
      return;
    }
    linear = linear * dim + coord;
  }
  batch_linear[row] = linear;
}

template <typename T>
static __global__ void gather_snapshots_kernel(
    const cuda::std::complex<T>* __restrict__ rd_maps, const int64_t* __restrict__ detection_idx,
    int64_t num_detections, int64_t idx_cols, int64_t n_preserved,
    const int64_t* __restrict__ batch_linear, int64_t num_channels, int64_t num_doppler,
    int64_t num_range, int64_t d_half, int64_t r_half, int doppler_edge, int range_edge,
    int64_t num_snapshots, int64_t slice_elems, cuda::std::complex<T>* __restrict__ snapshot,
    uint8_t* __restrict__ is_valid) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_detections) {
    return;
  }

  const int64_t* idx_row = detection_idx + row * idx_cols;
  const int64_t doppler_center = idx_row[n_preserved];
  const int64_t range_center = idx_row[n_preserved + 1];
  const int64_t batch_base = batch_linear[row] * slice_elems;

  const int64_t range_len = 2 * r_half + 1;
  for (int64_t snap_slot = static_cast<int64_t>(threadIdx.x); snap_slot < num_snapshots;
       snap_slot += static_cast<int64_t>(blockDim.x)) {
    const int64_t od = snap_slot / range_len - d_half;
    const int64_t orng = snap_slot % range_len - r_half;
    bool valid_d = true;
    bool valid_r = true;
    const int64_t d =
        resolve_axis_index(doppler_center, od, d_half, num_doppler, doppler_edge, &valid_d);
    const int64_t r =
        resolve_axis_index(range_center, orng, r_half, num_range, range_edge, &valid_r);
    const bool valid = valid_d && valid_r;
    if (is_valid != nullptr) {
      is_valid[row * num_snapshots + snap_slot] = static_cast<uint8_t>(valid);
    }
    const int64_t plane_offset = d * num_range + r;
    // snapshot is (num_detections, num_channels, num_snapshots) row-major
    // (channel-major, snapshot-minor), matching the estimators' covariance
    // indexing xi[c * num_snapshots + s]. Write channel c's slot at
    // c * num_snapshots + snap_slot — NOT snap_slot * num_channels + c, which
    // transposes to (num_snapshots, num_channels) and only agrees for K == 1.
    cuda::std::complex<T>* out = snapshot + row * num_channels * num_snapshots;
    if (valid) {
      for (int64_t c = 0; c < num_channels; ++c) {
        const int64_t src = batch_base + c * (num_doppler * num_range) + plane_offset;
        out[c * num_snapshots + snap_slot] = rd_maps[src];
      }
    }
  }
}

inline void launch_batch_linear(const int64_t* detection_idx, int64_t num_detections,
                                int64_t idx_cols, int64_t n_preserved, const int64_t* leading_dims,
                                int64_t num_doppler, int64_t num_range, int64_t* batch_linear,
                                int* error_flag, cudaStream_t stream) {
  if (num_detections <= 0) {
    return;
  }
  const int block = 256;
  const int grid = static_cast<int>((num_detections + block - 1) / static_cast<int64_t>(block));
  batch_linear_kernel<<<grid, block, 0, stream>>>(detection_idx, num_detections, idx_cols,
                                                  n_preserved, leading_dims, num_doppler, num_range,
                                                  batch_linear, error_flag);
}

template <typename T>
void launch_gather_snapshots(const cuda::std::complex<T>* rd_maps, const int64_t* detection_idx,
                             int64_t num_detections, int64_t idx_cols, int64_t n_preserved,
                             const int64_t* batch_linear, int64_t num_channels, int64_t num_doppler,
                             int64_t num_range, int64_t d_half, int64_t r_half,
                             SnapshotEdgeMode d_edge, SnapshotEdgeMode r_edge,
                             int64_t num_snapshots, int64_t slice_elems,
                             cuda::std::complex<T>* snapshot, uint8_t* is_valid,
                             cudaStream_t stream) {
  if (num_detections <= 0) {
    return;
  }
  const int grid = static_cast<int>(num_detections);
  const int block = num_snapshots == 1 ? 1 : (num_snapshots <= 32 ? 32 : 128);
  gather_snapshots_kernel<T><<<grid, block, 0, stream>>>(
      rd_maps, detection_idx, num_detections, idx_cols, n_preserved, batch_linear, num_channels,
      num_doppler, num_range, d_half, r_half, edge_mode_id(d_edge), edge_mode_id(r_edge),
      num_snapshots, slice_elems, snapshot, is_valid);
}

}  // namespace holoscan::radar::detail
