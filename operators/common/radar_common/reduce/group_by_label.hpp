// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public group-by-label reduction primitives.
// Non-noise labels must lie in [0, max_label] (DBSCAN-style).
//
// Host launch wrappers are declared here; non-template CUDA kernels live in
// group_by_label.cu and are compiled into holoscan::radar_common_reduce.
// Link that library — do not recompile the .cu into another target (ODR).
// Template custom-peak launches are defined below under __CUDACC__.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
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

template <typename T>
void launch_compact_points(const int64_t* best_rows, const T* points,
                           const T* slot_points, int64_t n_slots,
                           int64_t num_features, bool use_slot_points,
                           T* reduced_points, cudaStream_t stream);

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

template <typename T>
struct BuiltInRowScore {
  const T* weights;
  ScoreMethod method;
  bool uniform;

  __device__ T operator()(const T* row, int64_t num_features) const {
    if (uniform) {
      return T{1};
    }
    switch (method) {
      case ScoreMethod::kWeightedSum: {
        T score = T{0};
        for (int64_t f = 0; f < num_features; ++f) {
          score += weights[f] * row[f];
        }
        return score;
      }
      case ScoreMethod::kWeightedL2: {
        T score = T{0};
        for (int64_t f = 0; f < num_features; ++f) {
          const T value = weights[f] * row[f];
          score += value * value;
        }
        return sqrt(score);
      }
      case ScoreMethod::kWeightedMax: {
        T score = weights[0] * row[0];
        for (int64_t f = 1; f < num_features; ++f) {
          const T value = weights[f] * row[f];
          score = value > score ? value : score;
        }
        return score;
      }
      default:
        return T{0};
    }
  }
};

template <typename T, typename WeightFn>
__global__ void weighted_mean_accumulate_kernel(
    const int64_t* labels, const int64_t* detection_idx, const T* points,
    int64_t n, int64_t idx_cols, int64_t num_features, WeightFn weight_fn,
    T* weight_sums, T* point_sums, T* index_sums, int64_t* counts,
    int* invalid_weight) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T* point = points + i * num_features;
  const T alpha = weight_fn(point, num_features);
  if (!isfinite(alpha) || alpha < T{0}) {
    atomicExch(invalid_weight, 1);
    return;
  }
  atomicAdd(reinterpret_cast<cuda_u64*>(counts + lab), 1ull);
  atomicAdd(weight_sums + lab, alpha);
  T* point_dest = point_sums + lab * num_features;
  for (int64_t f = 0; f < num_features; ++f) {
    atomicAdd(point_dest + f, alpha * point[f]);
  }
  T* index_dest = index_sums + lab * idx_cols;
  const int64_t* index = detection_idx + i * idx_cols;
  for (int64_t c = 0; c < idx_cols; ++c) {
    atomicAdd(index_dest + c, alpha * static_cast<T>(index[c]));
  }
}

template <typename T>
__global__ void weighted_mean_finalize_kernel(
    T* weight_sums, T* point_sums, T* index_sums, const int64_t* counts,
    int64_t n_slots, int64_t idx_cols, int64_t num_features,
    int* invalid_weight) {
  const int64_t lab =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots || counts[lab] <= 0) {
    return;
  }
  const T total = weight_sums[lab];
  if (!isfinite(total) || total <= T{0}) {
    atomicExch(invalid_weight, 1);
    return;
  }
  const T inv = T{1} / total;
  for (int64_t f = 0; f < num_features; ++f) {
    point_sums[lab * num_features + f] *= inv;
  }
  for (int64_t c = 0; c < idx_cols; ++c) {
    index_sums[lab * idx_cols + c] *= inv;
  }
}

template <typename T>
__device__ T weighted_index_neg_dist2(const int64_t* index,
                                      const T* centroid, int64_t idx_cols) {
  T distance = T{0};
  for (int64_t c = 0; c < idx_cols; ++c) {
    const T delta = static_cast<T>(index[c]) - centroid[c];
    distance += delta * delta;
  }
  return -distance;
}

template <typename T>
__global__ void weighted_mean_max_negdist_kernel(
    const int64_t* labels, const int64_t* detection_idx,
    const T* index_centroids, int64_t n, int64_t idx_cols,
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
  const T score = weighted_index_neg_dist2(
      detection_idx + i * idx_cols, index_centroids + lab * idx_cols,
      idx_cols);
  atomicMax(best_score_bits + lab, ordered_bits_t(score));
}

template <typename T>
__global__ void weighted_mean_min_row_kernel(
    const int64_t* labels, const int64_t* detection_idx,
    const T* index_centroids, int64_t n, int64_t idx_cols,
    const cuda_u64* best_score_bits, int64_t* best_rows) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = weighted_index_neg_dist2(
      detection_idx + i * idx_cols, index_centroids + lab * idx_cols,
      idx_cols);
  if (ordered_bits_t(score) != best_score_bits[lab]) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab),
            static_cast<cuda_u64>(i));
}

template <typename T, typename WeightFn>
void launch_weighted_mean_select(
    const int64_t* labels, const int64_t* detection_idx, const T* points,
    int64_t n, int64_t n_slots, int64_t idx_cols, int64_t num_features,
    WeightFn weight_fn, T* weight_sums, T* point_means, T* index_means,
    int64_t* counts, int* invalid_weight, cuda_u64* best_score_bits,
    int64_t* best_rows, cudaStream_t stream) {
  weighted_mean_accumulate_kernel<T, WeightFn>
      <<<grid_for(n), 256, 0, stream>>>(
          labels, detection_idx, points, n, idx_cols, num_features, weight_fn,
          weight_sums, point_means, index_means, counts, invalid_weight);
  weighted_mean_finalize_kernel<T><<<grid_for(n_slots), 256, 0, stream>>>(
      weight_sums, point_means, index_means, counts, n_slots, idx_cols,
      num_features, invalid_weight);
  weighted_mean_max_negdist_kernel<T><<<grid_for(n), 256, 0, stream>>>(
      labels, detection_idx, index_means, n, idx_cols, best_score_bits);
  weighted_mean_min_row_kernel<T><<<grid_for(n), 256, 0, stream>>>(
      labels, detection_idx, index_means, n, idx_cols, best_score_bits,
      best_rows);
}
#endif  // __CUDACC__

}  // namespace holoscan::radar::detail
