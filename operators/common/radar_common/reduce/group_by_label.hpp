// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public group-by-label reduction primitives (CUDA-TU only).
// Non-noise labels must lie in [0, max_label] (DBSCAN-style).

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "radar_common/reduce/reduction_enums.hpp"

namespace holoscan::radar::detail {

// CUDA 64-bit atomics require this exact type (not uint64_t).
using cuda_u64 = unsigned long long;  // NOLINT(runtime/int)

constexpr int64_t kNoiseLabel = -1;
constexpr int64_t kEmptyRow = std::numeric_limits<int64_t>::max();

// Labels are non-negative cluster ids. Tracks max id and whether any
// non-noise label was seen (unsigned atomicMax cannot seed from -1 —
// that bit pattern is UINT64_MAX and would swallow all real labels).
__global__ void find_max_label_kernel(const int64_t* labels, int64_t n,
                                      int64_t* max_label, int* any_cluster) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab > kNoiseLabel) {
    atomicMax(reinterpret_cast<cuda_u64*>(max_label),
              static_cast<cuda_u64>(lab));
    atomicOr(any_cluster, 1);
  }
}

__global__ void init_i64_kernel(int64_t* data, int64_t n, int64_t value) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = value;
  }
}

__global__ void init_float_zero_kernel(float* data, int64_t n) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = 0.f;
  }
}

__global__ void select_first_kernel(const int64_t* labels, int64_t n,
                                    int64_t* best_rows) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  atomicMin(reinterpret_cast<cuda_u64*>(best_rows + lab),
            static_cast<cuda_u64>(i));
}

template <typename T>
__device__ T peak_score(const T* row, const T* weights, int64_t num_features,
                        ScoreMethod method) {
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
    case ScoreMethod::kCustom:
      return T{0};
  }
  return T{0};
}

__device__ cuda_u64 pack_score_row(float score, int64_t row) {
  unsigned int score_bits = __float_as_uint(score);
  if (score_bits & 0x80000000u) {
    score_bits = ~score_bits;
  } else {
    score_bits ^= 0x80000000u;
  }
  return (static_cast<cuda_u64>(score_bits) << 32) |
         static_cast<cuda_u64>(static_cast<uint32_t>(row));
}

__device__ int64_t unpack_row(cuda_u64 packed) {
  return static_cast<int64_t>(static_cast<uint32_t>(packed));
}

template <typename T>
__global__ void select_peak_kernel(const int64_t* labels, const T* points,
                                   const T* weights, int64_t n,
                                   int64_t num_features, ScoreMethod method,
                                   cuda_u64* best_packed, int* occupied) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T* row = points + i * num_features;
  const float score =
      static_cast<float>(peak_score(row, weights, num_features, method));
  atomicMax(best_packed + lab, pack_score_row(score, i));
  atomicOr(occupied + lab, 1);
}

template <typename T>
__global__ void accumulate_centroid_kernel(const int64_t* labels,
                                           const T* points, const T* weights,
                                           int64_t n, int64_t num_features,
                                           float* sum_feat, int64_t* counts) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  atomicAdd(reinterpret_cast<cuda_u64*>(counts + lab), 1ull);
  const T* row = points + i * num_features;
  float* dest = sum_feat + lab * num_features;
  for (int64_t f = 0; f < num_features; ++f) {
    atomicAdd(dest + f, static_cast<float>(weights[f] * row[f]));
  }
}

__global__ void finalize_centroid_kernel(float* sum_feat,
                                         const int64_t* counts, int64_t n_slots,
                                         int64_t num_features) {
  const int64_t lab =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots || counts[lab] <= 0) {
    return;
  }
  const float inv = 1.f / static_cast<float>(counts[lab]);
  float* row = sum_feat + lab * num_features;
  for (int64_t f = 0; f < num_features; ++f) {
    row[f] *= inv;
  }
}

template <typename T>
__global__ void select_medoid_kernel(const int64_t* labels, const T* points,
                                     const T* weights, const float* centroids,
                                     int64_t n, int64_t num_features,
                                     cuda_u64* best_packed, int* occupied) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t lab = labels[i];
  if (lab <= kNoiseLabel) {
    return;
  }
  const T* row = points + i * num_features;
  const float* cen = centroids + lab * num_features;
  float dist2 = 0.f;
  for (int64_t f = 0; f < num_features; ++f) {
    const float d =
        static_cast<float>(weights[f] * row[f]) - cen[f];
    dist2 += d * d;
  }
  atomicMax(best_packed + lab, pack_score_row(-dist2, i));
  atomicOr(occupied + lab, 1);
}

__global__ void packed_to_rows_kernel(const cuda_u64* best_packed,
                                      const int* occupied, int64_t* best_rows,
                                      int64_t n_slots) {
  const int64_t lab =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots) {
    return;
  }
  if (!occupied[lab]) {
    best_rows[lab] = kEmptyRow;
    return;
  }
  best_rows[lab] = unpack_row(best_packed[lab]);
}

__global__ void count_occupied_rows_kernel(const int64_t* best_rows,
                                           int64_t n_slots, int64_t* count) {
  const int64_t lab =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lab >= n_slots || best_rows[lab] == kEmptyRow) {
    return;
  }
  atomicAdd(reinterpret_cast<cuda_u64*>(count), 1ull);
}

__global__ void compact_and_gather_kernel(const int64_t* best_rows,
                                          const int64_t* detection_idx,
                                          int64_t n_slots, int64_t idx_cols,
                                          int64_t* rep_idx) {
  // Single-threaded ordered compact so rep_idx rows follow ascending label.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t out = 0;
  for (int64_t lab = 0; lab < n_slots; ++lab) {
    if (best_rows[lab] == kEmptyRow) {
      continue;
    }
    const int64_t row = best_rows[lab];
    const int64_t* src = detection_idx + row * idx_cols;
    int64_t* dst = rep_idx + out * idx_cols;
    for (int64_t c = 0; c < idx_cols; ++c) {
      dst[c] = src[c];
    }
    ++out;
  }
}

inline int grid_for(int64_t n, int block = 256) {
  if (n <= 0) {
    return 1;
  }
  return static_cast<int>((n + block - 1) / block);
}

}  // namespace holoscan::radar::detail
