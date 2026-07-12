// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public group-by-label reduction primitives.
// Non-noise labels must lie in [0, max_label] (DBSCAN-style).
//
// Host launch wrappers are declared here; CUDA kernels live only in
// group_by_label.cu (compiled into detection_reduction_op). Callers link that
// library — they must not re-define the kernels in another TU.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "radar_common/reduce/reduction_enums.hpp"

namespace holoscan::radar::detail {

// CUDA atomics require unsigned long long for 64-bit atomicMax.
using cuda_u64 = unsigned long long;  // NOLINT(runtime/int)

constexpr int64_t kNoiseLabel = -1;
constexpr int64_t kEmptyRow = std::numeric_limits<int64_t>::max();

inline int grid_for(int64_t n, int block = 256) {
  if (n <= 0) {
    return 1;
  }
  return static_cast<int>((n + block - 1) / block);
}

void launch_find_max_label(const int64_t* labels, int64_t n,
                           int64_t* max_label, int* any_cluster,
                           cudaStream_t stream);

void launch_init_i64(int64_t* data, int64_t n, int64_t value,
                     cudaStream_t stream);

void launch_init_u64(cuda_u64* data, int64_t n, cuda_u64 value,
                     cudaStream_t stream);

template <typename T>
void launch_init_t_zero(T* data, int64_t n, cudaStream_t stream);

void launch_select_first(const int64_t* labels, int64_t n, int64_t* best_rows,
                         cudaStream_t stream);

template <typename T>
void launch_peak_select(const int64_t* labels, const T* points,
                        const T* weights, int64_t n, int64_t num_features,
                        ScoreMethod method, cuda_u64* best_score_bits,
                        int64_t* best_rows, cudaStream_t stream);

template <typename T>
void launch_medoid_select(const int64_t* labels, const T* points,
                          const T* weights, T* sum_feat, int64_t* counts,
                          int64_t n, int64_t n_slots, int64_t num_features,
                          cuda_u64* best_score_bits, int64_t* best_rows,
                          cudaStream_t stream);

void launch_count_occupied(const int64_t* best_rows, int64_t n_slots,
                           int64_t* count, cudaStream_t stream);

void launch_compact_and_gather(const int64_t* best_rows,
                               const int64_t* detection_idx, int64_t n_slots,
                               int64_t idx_cols, int64_t* rep_idx,
                               cudaStream_t stream);

#if defined(__CUDACC__)
// Template custom-peak path (ScoreFn is a compile-time device functor).
// Safe in a header: each ScoreFn instantiation has a unique mangled symbol.

template <typename T>
__device__ inline cuda_u64 ordered_bits_t(T v);

template <>
__device__ inline cuda_u64 ordered_bits_t<float>(float v) {
  unsigned int bits = __float_as_uint(v);
  if (bits & 0x80000000u) {
    bits = ~bits;
  } else {
    bits ^= 0x80000000u;
  }
  return static_cast<cuda_u64>(bits);
}

template <>
__device__ inline cuda_u64 ordered_bits_t<double>(double v) {
  cuda_u64 bits = static_cast<cuda_u64>(__double_as_longlong(v));
  if (bits & 0x8000000000000000ull) {
    bits = ~bits;
  } else {
    bits ^= 0x8000000000000000ull;
  }
  return bits;
}

template <typename T, typename ScoreFn>
__global__ void peak_max_score_custom_kernel(const int64_t* labels,
                                             const T* points, int64_t n,
                                             int64_t num_features, ScoreFn fn,
                                             cuda_u64* best_score_bits) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = fn(points + i * num_features, num_features);
  atomicMax(best_score_bits + lab, ordered_bits_t(score));
}

template <typename T, typename ScoreFn>
__global__ void peak_min_row_custom_kernel(
    const int64_t* labels, const T* points, int64_t n, int64_t num_features,
    ScoreFn fn, const cuda_u64* best_score_bits, int64_t* best_rows) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = fn(points + i * num_features, num_features);
  if (ordered_bits_t(score) != best_score_bits[lab]) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab),
            static_cast<cuda_u64>(i));
}

template <typename T, typename ScoreFn>
void launch_peak_select_custom(const int64_t* labels, const T* points,
                               int64_t n, int64_t num_features, ScoreFn fn,
                               cuda_u64* best_score_bits, int64_t* best_rows,
                               cudaStream_t stream) {
  peak_max_score_custom_kernel<T, ScoreFn>
      <<<grid_for(n), 256, 0, stream>>>(labels, points, n, num_features, fn,
                                       best_score_bits);
  peak_min_row_custom_kernel<T, ScoreFn>
      <<<grid_for(n), 256, 0, stream>>>(labels, points, n, num_features, fn,
                                       best_score_bits, best_rows);
}
#endif  // __CUDACC__

}  // namespace holoscan::radar::detail
