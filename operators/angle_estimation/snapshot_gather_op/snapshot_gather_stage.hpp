// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar::detail {

constexpr int kSnapshotErrBatchCoord = 1;
constexpr int kMaxSnapshotPreservedBatchDims = 8;

__host__ __device__ inline int edge_mode_id(SnapshotEdgeMode mode) {
  switch (mode) {
    case SnapshotEdgeMode::kWrap: return 0;
    case SnapshotEdgeMode::kClamp: return 1;
    case SnapshotEdgeMode::kShift: return 2;
  }
  return 0;
}

__device__ inline int64_t shift_window_start(int64_t center, int64_t half,
                                             int64_t n) {
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

__device__ inline int64_t resolve_axis_index(int64_t center, int64_t offset,
                                             int64_t half, int64_t n,
                                             int edge_mode, bool* valid) {
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

__global__ void batch_linear_kernel(const int64_t* __restrict__ detection_idx,
                                    int64_t num_detections, int64_t idx_cols,
                                    int64_t n_preserved,
                                    const int64_t* __restrict__ leading_dims,
                                    int64_t* __restrict__ batch_linear,
                                    int* __restrict__ error_flag) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (row >= num_detections) {
    return;
  }
  if (n_preserved == 0) {
    batch_linear[row] = 0;
    return;
  }
  const int64_t* idx_row = detection_idx + row * idx_cols;
  int64_t linear = 0;
  for (int64_t a = 0; a < n_preserved; ++a) {
    const int64_t coord = idx_row[a];
    const int64_t dim = leading_dims[a];
    if (coord < 0 || coord >= dim) {
      atomicExch(error_flag, kSnapshotErrBatchCoord);
      return;
    }
    linear = linear * dim + coord;
  }
  batch_linear[row] = linear;
}

template <typename T>
__global__ void gather_snapshots_kernel(
    const cuda::std::complex<T>* __restrict__ rd_maps,
    const int64_t* __restrict__ detection_idx, int64_t num_detections,
    int64_t idx_cols, int64_t n_preserved,
    const int64_t* __restrict__ batch_linear, int64_t num_channels,
    int64_t num_doppler, int64_t num_range, int64_t d_half, int64_t r_half,
    int doppler_edge, int range_edge, int64_t num_snapshots,
    int64_t slice_elems, cuda::std::complex<T>* __restrict__ snapshot,
    bool* __restrict__ is_valid) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_detections) {
    return;
  }

  const int64_t* idx_row = detection_idx + row * idx_cols;
  const int64_t doppler_center = idx_row[n_preserved];
  const int64_t range_center = idx_row[n_preserved + 1];
  const int64_t batch_base = batch_linear[row] * slice_elems;

  int64_t snap_slot = 0;
  for (int64_t od = -d_half; od <= d_half; ++od) {
    for (int64_t orng = -r_half; orng <= r_half; ++orng) {
      bool valid_d = true;
      bool valid_r = true;
      const int64_t d = resolve_axis_index(doppler_center, od, d_half,
                                           num_doppler, doppler_edge, &valid_d);
      const int64_t r = resolve_axis_index(range_center, orng, r_half,
                                           num_range, range_edge, &valid_r);
      const bool valid = valid_d && valid_r;
      if (is_valid != nullptr) {
        is_valid[row * num_snapshots + snap_slot] = valid;
      }
      const int64_t plane_offset = d * num_range + r;
      cuda::std::complex<T>* out =
          snapshot + (row * num_channels * num_snapshots + snap_slot * num_channels);
      if (valid) {
        for (int64_t c = 0; c < num_channels; ++c) {
          const int64_t src =
              batch_base + c * (num_doppler * num_range) + plane_offset;
          out[c] = rd_maps[src];
        }
      }
      ++snap_slot;
    }
  }
}

inline void launch_batch_linear(const int64_t* detection_idx,
                                int64_t num_detections, int64_t idx_cols,
                                int64_t n_preserved,
                                const int64_t* leading_dims,
                                int64_t* batch_linear, int* error_flag,
                                cudaStream_t stream) {
  if (num_detections <= 0) {
    return;
  }
  const int block = 256;
  const int grid = static_cast<int>(
      (num_detections + block - 1) / static_cast<int64_t>(block));
  batch_linear_kernel<<<grid, block, 0, stream>>>(
      detection_idx, num_detections, idx_cols, n_preserved, leading_dims,
      batch_linear, error_flag);
}

template <typename T>
void launch_gather_snapshots(
    const cuda::std::complex<T>* rd_maps, const int64_t* detection_idx,
    int64_t num_detections, int64_t idx_cols, int64_t n_preserved,
    const int64_t* batch_linear, int64_t num_channels, int64_t num_doppler,
    int64_t num_range, int64_t d_half, int64_t r_half, SnapshotEdgeMode d_edge,
    SnapshotEdgeMode r_edge, int64_t num_snapshots, int64_t slice_elems,
    cuda::std::complex<T>* snapshot, bool* is_valid, cudaStream_t stream) {
  if (num_detections <= 0) {
    return;
  }
  const int grid = static_cast<int>(num_detections);
  gather_snapshots_kernel<T><<<grid, 1, 0, stream>>>(
      rd_maps, detection_idx, num_detections, idx_cols, n_preserved,
      batch_linear, num_channels, num_doppler, num_range, d_half, r_half,
      edge_mode_id(d_edge), edge_mode_id(r_edge), num_snapshots, slice_elems,
      snapshot, is_valid);
}

}  // namespace holoscan::radar::detail
