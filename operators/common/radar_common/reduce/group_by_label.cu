// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/reduce/group_by_label.hpp"

#include <cub/block/block_scan.cuh>

#include <cmath>
#include <cstdint>

namespace holoscan::radar::detail {

namespace {

template <typename T>
__device__ T peak_score(const T* row, const T* weights, int64_t num_features, ScoreMethod method) {
  switch (method) {
    case ScoreMethod::kWeightedSum: {
      T s = T{0};
      for (int64_t f = 0; f < num_features; ++f) {
        s += weights[f] * row[f];
      }
      return s;
    }
    case ScoreMethod::kWeightedL2: {
      T s = T{0};
      for (int64_t f = 0; f < num_features; ++f) {
        const T v = weights[f] * row[f];
        s += v * v;
      }
      return sqrt(s);
    }
    case ScoreMethod::kWeightedMax: {
      if (num_features <= 0) {
        return T{0};
      }
      T s = weights[0] * row[0];
      for (int64_t f = 1; f < num_features; ++f) {
        const T v = weights[f] * row[f];
        s = v > s ? v : s;
      }
      return s;
    }
    default:
      return T{0};
  }
}

template <typename T>
__device__ T medoid_neg_dist2(const T* row, const T* weights, const T* cen, int64_t num_features) {
  T dist2 = T{0};
  for (int64_t f = 0; f < num_features; ++f) {
    const T d = weights[f] * row[f] - cen[f];
    dist2 += d * d;
  }
  return -dist2;
}

__global__ void find_max_label_kernel(const int64_t* labels, int64_t n, int64_t* max_label,
                                      int* any_cluster) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab > kNoiseLabel) {
    atomicMax(reinterpret_cast<cuda_u64*>(max_label), static_cast<cuda_u64>(lab));
    atomicOr(any_cluster, 1);
  }
}

__global__ void init_i64_kernel(int64_t* data, int64_t n, int64_t value) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = value;
  }
}

__global__ void init_u64_kernel(cuda_u64* data, int64_t n, cuda_u64 value) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = value;
  }
}

template <typename T>
__global__ void init_t_zero_kernel(T* data, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = T{0};
  }
}

__global__ void select_first_kernel(const int64_t* labels, int64_t n, int64_t* best_rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab), static_cast<cuda_u64>(i));
}

template <typename T>
__global__ void peak_max_score_kernel(const int64_t* labels, const T* points, const T* weights,
                                      int64_t n, int64_t num_features, ScoreMethod method,
                                      cuda_u64* best_score_bits) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = peak_score(points + i * num_features, weights, num_features, method);
  atomicMax(best_score_bits + lab, ordered_bits_t(score));
}

template <typename T>
__global__ void peak_min_row_kernel(const int64_t* labels, const T* points, const T* weights,
                                    int64_t n, int64_t num_features, ScoreMethod method,
                                    const cuda_u64* best_score_bits, int64_t* best_rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = peak_score(points + i * num_features, weights, num_features, method);
  if (ordered_bits_t(score) != best_score_bits[lab]) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab), static_cast<cuda_u64>(i));
}

template <typename T>
__global__ void accumulate_centroid_kernel(const int64_t* labels, const T* points, const T* weights,
                                           int64_t n, int64_t num_features, T* sum_feat,
                                           int64_t* counts) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  atomicAdd(reinterpret_cast<cuda_u64*>(counts + lab), 1ull);
  const T* row = points + i * num_features;
  T* dest = sum_feat + lab * num_features;
  for (int64_t f = 0; f < num_features; ++f) {
    atomicAdd(dest + f, weights[f] * row[f]);
  }
}

template <typename T>
__global__ void finalize_centroid_kernel(T* sum_feat, const int64_t* counts, int64_t n_slots,
                                         int64_t num_features) {
  const int64_t lab = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots || counts[lab] <= 0) {
    return;
  }
  const T inv = T{1} / static_cast<T>(counts[lab]);
  T* row = sum_feat + lab * num_features;
  for (int64_t f = 0; f < num_features; ++f) {
    row[f] *= inv;
  }
}

template <typename T>
__global__ void medoid_max_negdist_kernel(const int64_t* labels, const T* points, const T* weights,
                                          const T* centroids, int64_t n, int64_t num_features,
                                          cuda_u64* best_score_bits) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = medoid_neg_dist2(points + i * num_features, weights,
                                   centroids + lab * num_features, num_features);
  atomicMax(best_score_bits + lab, ordered_bits_t(score));
}

template <typename T>
__global__ void medoid_min_row_kernel(const int64_t* labels, const T* points, const T* weights,
                                      const T* centroids, int64_t n, int64_t num_features,
                                      const cuda_u64* best_score_bits, int64_t* best_rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T score = medoid_neg_dist2(points + i * num_features, weights,
                                   centroids + lab * num_features, num_features);
  if (ordered_bits_t(score) != best_score_bits[lab]) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab), static_cast<cuda_u64>(i));
}

__global__ void count_occupied_rows_kernel(const int64_t* best_rows, int64_t n_slots,
                                           int64_t* count) {
  const int64_t lab = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots || best_rows[lab] == kEmptyRow) {
    return;
  }
  atomicAdd(reinterpret_cast<cuda_u64*>(count), 1ull);
}

__global__ void compact_and_gather_kernel(const int64_t* best_rows, const int64_t* detection_idx,
                                          int64_t n_slots, int64_t idx_cols, int64_t* rep_idx) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int64_t base;
  if (threadIdx.x == 0)
    base = 0;
  __syncthreads();
  for (int64_t tile = 0; tile < n_slots; tile += 256) {
    const int64_t lab = tile + threadIdx.x;
    const bool occupied = lab < n_slots && best_rows[lab] != kEmptyRow;
    int offset = 0;
    int total = 0;
    Scan(scan).ExclusiveSum(occupied ? 1 : 0, offset, total);
    if (occupied) {
      const int64_t row = best_rows[lab];
      for (int64_t c = 0; c < idx_cols; ++c) {
        rep_idx[(base + offset) * idx_cols + c] = detection_idx[row * idx_cols + c];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
      base += total;
    __syncthreads();
  }
}

template <typename T>
__global__ void compact_points_kernel(const int64_t* best_rows, const T* points,
                                      const T* slot_points, int64_t n_slots, int64_t num_features,
                                      bool use_slot_points, T* reduced_points) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int64_t base;
  if (threadIdx.x == 0)
    base = 0;
  __syncthreads();
  for (int64_t tile = 0; tile < n_slots; tile += 256) {
    const int64_t lab = tile + threadIdx.x;
    const bool occupied = lab < n_slots && best_rows[lab] != kEmptyRow;
    int offset = 0;
    int total = 0;
    Scan(scan).ExclusiveSum(occupied ? 1 : 0, offset, total);
    if (occupied) {
      const T* row = use_slot_points ? slot_points + lab * num_features
                                     : points + best_rows[lab] * num_features;
      for (int64_t f = 0; f < num_features; ++f) {
        reduced_points[(base + offset) * num_features + f] = row[f];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
      base += total;
    __syncthreads();
  }
}

}  // namespace

void launch_find_max_label(const int64_t* labels, int64_t n, int64_t* max_label, int* any_cluster,
                           cudaStream_t stream) {
  find_max_label_kernel<<<grid_for(n), 256, 0, stream>>>(labels, n, max_label, any_cluster);
}

void launch_init_i64(int64_t* data, int64_t n, int64_t value, cudaStream_t stream) {
  init_i64_kernel<<<grid_for(n), 256, 0, stream>>>(data, n, value);
}

void launch_init_u64(cuda_u64* data, int64_t n, cuda_u64 value, cudaStream_t stream) {
  init_u64_kernel<<<grid_for(n), 256, 0, stream>>>(data, n, value);
}

template <typename T>
void launch_init_t_zero(T* data, int64_t n, cudaStream_t stream) {
  init_t_zero_kernel<T><<<grid_for(n), 256, 0, stream>>>(data, n);
}

void launch_select_first(const int64_t* labels, int64_t n, int64_t* best_rows,
                         cudaStream_t stream) {
  select_first_kernel<<<grid_for(n), 256, 0, stream>>>(labels, n, best_rows);
}

template <typename T>
void launch_peak_select(const int64_t* labels, const T* points, const T* weights, int64_t n,
                        int64_t num_features, ScoreMethod method, cuda_u64* best_score_bits,
                        int64_t* best_rows, cudaStream_t stream) {
  // Caller initializes best_score_bits to 0 and best_rows to kEmptyRow over
  // the slot count (not n). This launcher only walks the point axis.
  if (method == ScoreMethod::kCustom) {
    return;
  }
  peak_max_score_kernel<T><<<grid_for(n), 256, 0, stream>>>(labels, points, weights, n,
                                                            num_features, method, best_score_bits);
  peak_min_row_kernel<T><<<grid_for(n), 256, 0, stream>>>(labels, points, weights, n, num_features,
                                                          method, best_score_bits, best_rows);
}

template <typename T>
void launch_medoid_select(const int64_t* labels, const T* points, const T* weights, T* sum_feat,
                          int64_t* counts, int64_t n, int64_t n_slots, int64_t num_features,
                          cuda_u64* best_score_bits, int64_t* best_rows, cudaStream_t stream) {
  // Caller zeros counts/sum_feat/best_score_bits and sets best_rows to
  // kEmptyRow over n_slots.
  accumulate_centroid_kernel<T>
      <<<grid_for(n), 256, 0, stream>>>(labels, points, weights, n, num_features, sum_feat, counts);
  finalize_centroid_kernel<T>
      <<<grid_for(n_slots), 256, 0, stream>>>(sum_feat, counts, n_slots, num_features);
  medoid_max_negdist_kernel<T><<<grid_for(n), 256, 0, stream>>>(labels, points, weights, sum_feat,
                                                                n, num_features, best_score_bits);
  medoid_min_row_kernel<T><<<grid_for(n), 256, 0, stream>>>(
      labels, points, weights, sum_feat, n, num_features, best_score_bits, best_rows);
}

void launch_count_occupied(const int64_t* best_rows, int64_t n_slots, int64_t* count,
                           cudaStream_t stream) {
  count_occupied_rows_kernel<<<grid_for(n_slots), 256, 0, stream>>>(best_rows, n_slots, count);
}

void launch_compact_and_gather(const int64_t* best_rows, const int64_t* detection_idx,
                               int64_t n_slots, int64_t idx_cols, int64_t* rep_idx,
                               cudaStream_t stream) {
  compact_and_gather_kernel<<<1, 256, 0, stream>>>(best_rows, detection_idx, n_slots, idx_cols,
                                                   rep_idx);
}

template <typename T>
void launch_compact_points(const int64_t* best_rows, const T* points, const T* slot_points,
                           int64_t n_slots, int64_t num_features, bool use_slot_points,
                           T* reduced_points, cudaStream_t stream) {
  compact_points_kernel<T><<<1, 256, 0, stream>>>(best_rows, points, slot_points, n_slots,
                                                  num_features, use_slot_points, reduced_points);
}

template void launch_init_t_zero<float>(float*, int64_t, cudaStream_t);
template void launch_init_t_zero<double>(double*, int64_t, cudaStream_t);

template void launch_peak_select<float>(const int64_t*, const float*, const float*, int64_t,
                                        int64_t, ScoreMethod, cuda_u64*, int64_t*, cudaStream_t);
template void launch_peak_select<double>(const int64_t*, const double*, const double*, int64_t,
                                         int64_t, ScoreMethod, cuda_u64*, int64_t*, cudaStream_t);

template void launch_medoid_select<float>(const int64_t*, const float*, const float*, float*,
                                          int64_t*, int64_t, int64_t, int64_t, cuda_u64*, int64_t*,
                                          cudaStream_t);
template void launch_medoid_select<double>(const int64_t*, const double*, const double*, double*,
                                           int64_t*, int64_t, int64_t, int64_t, cuda_u64*, int64_t*,
                                           cudaStream_t);

template void launch_compact_points<float>(const int64_t*, const float*, const float*, int64_t,
                                           int64_t, bool, float*, cudaStream_t);
template void launch_compact_points<double>(const int64_t*, const double*, const double*, int64_t,
                                            int64_t, bool, double*, cudaStream_t);

}  // namespace holoscan::radar::detail
