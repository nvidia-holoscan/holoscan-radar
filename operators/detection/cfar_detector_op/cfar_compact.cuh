// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device-side detection_idx compaction for CFARDetectorOp.
//
// Two-pass, raster-stable:
//   1) Convert bool mask → uint32 flags, CUB DeviceReduce::Sum → count
//   2) CUB DeviceScan::ExclusiveSum on flags + scatter kernel writing
//      (preserved..., doppler, range) rows in batch-major raster order
//
// Include only from .cu translation units (pulls in CUB).

#pragma once

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace holoscan::radar::detail {

// Cached CUB temps + scan offsets. Owned by CFARDetectorOp; reused across
// ticks when out_elems / preserved rank are stable.
struct DetectionIdxCompactState {
  void* cub_temp{nullptr};
  size_t cub_temp_bytes{0};
  uint32_t* d_offsets{nullptr};
  int64_t offsets_capacity{0};
  int64_t* d_count{nullptr};
  int64_t* d_preserved_dims{nullptr};
  int64_t preserved_capacity{0};

  DetectionIdxCompactState() = default;
  DetectionIdxCompactState(const DetectionIdxCompactState&) = delete;
  DetectionIdxCompactState& operator=(const DetectionIdxCompactState&) =
      delete;

  ~DetectionIdxCompactState() {
    if (cub_temp != nullptr) {
      cudaFree(cub_temp);
    }
    if (d_offsets != nullptr) {
      cudaFree(d_offsets);
    }
    if (d_count != nullptr) {
      cudaFree(d_count);
    }
    if (d_preserved_dims != nullptr) {
      cudaFree(d_preserved_dims);
    }
  }

  void ensure_cub_temp(size_t bytes) {
    if (bytes <= cub_temp_bytes) {
      return;
    }
    if (cub_temp != nullptr) {
      cudaFree(cub_temp);
      cub_temp = nullptr;
      cub_temp_bytes = 0;
    }
    if (cudaMalloc(&cub_temp, bytes) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionIdxCompactState: cub temp cudaMalloc failed");
    }
    cub_temp_bytes = bytes;
  }

  void ensure_offsets(int64_t n) {
    if (n <= offsets_capacity) {
      return;
    }
    if (d_offsets != nullptr) {
      cudaFree(d_offsets);
      d_offsets = nullptr;
      offsets_capacity = 0;
    }
    if (cudaMalloc(&d_offsets, static_cast<size_t>(n) * sizeof(uint32_t)) !=
        cudaSuccess) {
      throw std::runtime_error(
          "DetectionIdxCompactState: offsets cudaMalloc failed");
    }
    offsets_capacity = n;
  }

  void ensure_count_scalar() {
    if (d_count != nullptr) {
      return;
    }
    if (cudaMalloc(&d_count, sizeof(int64_t)) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionIdxCompactState: count cudaMalloc failed");
    }
  }

  void ensure_preserved(int64_t n) {
    if (n <= 0) {
      return;
    }
    if (n <= preserved_capacity) {
      return;
    }
    if (d_preserved_dims != nullptr) {
      cudaFree(d_preserved_dims);
      d_preserved_dims = nullptr;
      preserved_capacity = 0;
    }
    if (cudaMalloc(&d_preserved_dims,
                   static_cast<size_t>(n) * sizeof(int64_t)) != cudaSuccess) {
      throw std::runtime_error(
          "DetectionIdxCompactState: preserved_dims cudaMalloc failed");
    }
    preserved_capacity = n;
  }
};

namespace compact_detail {

__global__ void mask_to_u32_kernel(const uint8_t* mask, uint32_t* flags,
                                   int64_t n) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    flags[i] = mask[i] != 0 ? 1u : 0u;
  }
}

inline void launch_mask_to_u32(const uint8_t* mask, uint32_t* flags, int64_t n,
                               cudaStream_t stream) {
  const int block = 256;
  const int grid =
      static_cast<int>((n + block - 1) / static_cast<int64_t>(block));
  mask_to_u32_kernel<<<grid, block, 0, stream>>>(mask, flags, n);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("mask_to_u32_kernel launch failed");
  }
}

inline void launch_count_u32(const uint32_t* flags, int64_t n, int64_t* d_count,
                             DetectionIdxCompactState* state,
                             cudaStream_t stream) {
  // Sum into a uint32 device scalar first (CUB Sum matches input value type),
  // then widen to int64 on the host via a tiny D2H of uint32 into the int64
  // slot's low bytes — cleaner: reduce to uint32 then cast kernel.
  size_t bytes = 0;
  uint32_t* d_count_u32 = reinterpret_cast<uint32_t*>(state->d_count);
  cub::DeviceReduce::Sum(nullptr, bytes, flags, d_count_u32,
                         static_cast<int>(n), stream);
  state->ensure_cub_temp(bytes);
  cub::DeviceReduce::Sum(state->cub_temp, state->cub_temp_bytes, flags,
                         d_count_u32, static_cast<int>(n), stream);
}

// After uint32 reduce into the first 4 bytes of d_count, zero-extend to int64.
__global__ void u32_count_to_i64_kernel(int64_t* d_count) {
  const uint32_t lo = *reinterpret_cast<uint32_t*>(d_count);
  *d_count = static_cast<int64_t>(lo);
}

inline void launch_exclusive_sum(uint32_t* flags_io, int64_t n,
                                 DetectionIdxCompactState* state,
                                 cudaStream_t stream) {
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, flags_io, flags_io,
                                static_cast<int>(n), stream);
  state->ensure_cub_temp(bytes);
  cub::DeviceScan::ExclusiveSum(state->cub_temp, state->cub_temp_bytes,
                                flags_io, flags_io, static_cast<int>(n),
                                stream);
}

// Decode linear index → (preserved..., d, r) and write one output row.
// Caps at max_emit so max_detections surplus is dropped while preserving
// raster order of the kept prefix. `offsets` is the exclusive-sum of flags;
// a cell is a detection iff offsets[i+1]-style — we still need the original
// mask because ExclusiveSum overwrites the flags buffer in place.
__global__ void scatter_detection_idx_kernel(
    const uint8_t* mask, const uint32_t* offsets, int64_t out_elems,
    int64_t num_doppler, int64_t num_range, int64_t n_preserved,
    const int64_t* preserved_dims, int64_t max_emit, int64_t idx_cols,
    int64_t* out_idx) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= out_elems) {
    return;
  }
  if (mask[i] == 0) {
    return;
  }
  const uint32_t slot = offsets[i];
  if (static_cast<int64_t>(slot) >= max_emit) {
    return;
  }

  const int64_t plane = num_doppler * num_range;
  const int64_t b_out = i / plane;
  const int64_t rem = i - b_out * plane;
  const int64_t d = rem / num_range;
  const int64_t r = rem - d * num_range;

  int64_t* row = out_idx + static_cast<int64_t>(slot) * idx_cols;
  int64_t linear = b_out;
  for (int64_t axis = n_preserved - 1; axis >= 0; --axis) {
    const int64_t dim = preserved_dims[axis];
    row[axis] = linear % dim;
    linear /= dim;
  }
  row[n_preserved] = d;
  row[n_preserved + 1] = r;
}

}  // namespace compact_detail

// Count detections in mask[0, out_elems). Returns the host-side count (one
// device sync on the count scalar only). Also leaves uint32 flags in
// state->d_offsets for a subsequent compact_detection_idx call on the same
// tick (compact reuses / overwrites that buffer with the exclusive scan).
inline int64_t count_mask_detections(const bool* mask, int64_t out_elems,
                                     DetectionIdxCompactState* state,
                                     cudaStream_t stream) {
  if (out_elems <= 0) {
    return 0;
  }
  if (out_elems > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "count_mask_detections: out_elems exceeds CUB 32-bit length limit");
  }
  state->ensure_count_scalar();
  state->ensure_offsets(out_elems);
  const uint8_t* mask_u8 = reinterpret_cast<const uint8_t*>(mask);
  compact_detail::launch_mask_to_u32(mask_u8, state->d_offsets, out_elems,
                                     stream);
  compact_detail::launch_count_u32(state->d_offsets, out_elems, state->d_count,
                                   state, stream);
  compact_detail::u32_count_to_i64_kernel<<<1, 1, 0, stream>>>(state->d_count);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("u32_count_to_i64_kernel launch failed");
  }
  int64_t host_count = 0;
  if (cudaMemcpyAsync(&host_count, state->d_count, sizeof(int64_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    throw std::runtime_error("count_mask_detections: count D2H failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("count_mask_detections: stream sync failed");
  }
  return host_count;
}

// Write the first max_emit detections (raster order) into out_idx of shape
// (max_emit, 2 + n_preserved). Requires count_mask_detections to have been
// called this tick (flags currently in state->d_offsets). No-op when
// max_emit == 0.
inline void compact_detection_idx(
    const bool* mask, int64_t out_elems, int64_t num_doppler, int64_t num_range,
    const std::vector<int64_t>& preserved_dims, int64_t max_emit,
    int64_t* out_idx, DetectionIdxCompactState* state, cudaStream_t stream) {
  if (max_emit <= 0 || out_elems <= 0) {
    return;
  }
  if (out_elems > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "compact_detection_idx: out_elems exceeds CUB 32-bit length limit");
  }
  if (state->d_offsets == nullptr || state->offsets_capacity < out_elems) {
    throw std::runtime_error(
        "compact_detection_idx: call count_mask_detections first this tick");
  }

  const int64_t n_preserved = static_cast<int64_t>(preserved_dims.size());
  const int64_t idx_cols = 2 + n_preserved;

  state->ensure_preserved(n_preserved);
  if (n_preserved > 0) {
    if (cudaMemcpyAsync(state->d_preserved_dims, preserved_dims.data(),
                        static_cast<size_t>(n_preserved) * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(
          "compact_detection_idx: preserved_dims H2D failed");
    }
  }

  // In-place exclusive sum: flags → offsets. Scatter still reads the original
  // bool mask to decide which cells write.
  compact_detail::launch_exclusive_sum(state->d_offsets, out_elems, state,
                                       stream);

  const int block = 256;
  const int grid =
      static_cast<int>((out_elems + block - 1) / static_cast<int64_t>(block));
  compact_detail::scatter_detection_idx_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(mask), state->d_offsets, out_elems,
      num_doppler, num_range, n_preserved, state->d_preserved_dims, max_emit,
      idx_cols, out_idx);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error(
        "compact_detection_idx: scatter kernel launch failed");
  }
}

}  // namespace holoscan::radar::detail
