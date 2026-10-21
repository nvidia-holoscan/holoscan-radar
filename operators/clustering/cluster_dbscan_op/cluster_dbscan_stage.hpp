// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CUDA-TU stage for ClusterDBSCANOp. Brute-force O(num_points²) DBSCAN:
//
//   1. mark_core        — per point, count neighbors (all points) → is_core;
//                         seed the connected-components label (core → own row,
//                         non-core → noise sentinel).
//   2. cc_step (loop)    — iterative min-label propagation with a pointer-jump,
//                         over core–core edges, until no label changes. Reads a
//                         stable snapshot (ping-pong buffers) so there is no
//                         intra-launch data race; converges to the min row
//                         index of each core connected component (the root).
//   3. assign_border    — each non-core point joins the min-root core neighbor
//                         (if any); otherwise stays noise.
//   4. root_flag / scan  — mark component roots, CUB exclusive-sum → dense
//                         cluster ids in ascending-root order.
//   5. finalize_labels   — map each point's root to its dense id (noise → -1).
//   6. renumber          — reorder ids by descending cluster size (tie-break
//                         ascending root) so cluster 0 is the largest; the host
//                         argsorts the per-id counts and remap_labels applies
//                         the permutation. This also makes the max_clusters cap
//                         (drop highest ids) keep the largest clusters.
//   7. histogram         — per-cluster point counts → sizes (descending).
//
// Include from a .cu that explicitly instantiates the needed specializations.

#pragma once

#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::radar::cluster_dbscan {

constexpr int64_t kNoiseRoot = -1;

inline int grid_for(int64_t n, int block = 256) {
  if (n <= 0) {
    return 1;
  }
  return static_cast<int>((n + block - 1) / block);
}

// q is a neighbor of p when Σ_a ((p_a − q_a) · eps_inv2_a) ≤ 1. A zero
// eps_inv2 entry excludes that feature entirely; skip it before reading the
// point values so ignored non-finite pass-through features cannot poison d2.
template <typename T>
__device__ inline bool dbscan_neighbor(const T* __restrict__ pi, const T* __restrict__ pj,
                                       const T* __restrict__ eps_inv2, int64_t f) {
  T d2 = static_cast<T>(0);
  for (int64_t a = 0; a < f; ++a) {
    if (eps_inv2[a] == static_cast<T>(0)) {
      continue;
    }
    const T diff = pi[a] - pj[a];
    d2 += diff * diff * eps_inv2[a];
    if (d2 > static_cast<T>(1)) {
      return false;  // early out; distance is monotonically accumulating
    }
  }
  return true;
}

// Count neighbors of each point (self included) over all points → is_core when
// the count reaches min_points. Seed cc: core → its own row index (a candidate
// component root), non-core → the noise sentinel.
template <typename T>
__global__ void mark_core_kernel(const T* __restrict__ points, const T* __restrict__ eps_inv2,
                                 int64_t n, int64_t f, int64_t min_points,
                                 uint8_t* __restrict__ is_core, int64_t* __restrict__ cc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const T* pi = points + i * f;
  int64_t count = 0;
  for (int64_t j = 0; j < n; ++j) {
    if (dbscan_neighbor<T>(pi, points + j * f, eps_inv2, f)) {
      ++count;
      // Only the core predicate is consumed; later neighbors cannot change it.
      if (count >= min_points) {
        break;
      }
    }
  }
  const bool core = count >= min_points;
  is_core[i] = core ? uint8_t{1} : uint8_t{0};
  cc[i] = core ? i : kNoiseRoot;
}

// One connected-components relaxation step over core–core edges. Reads the
// stable snapshot `cc_in`, writes `cc_out` (thread i owns row i, so no atomics
// and no aliasing races). Combines min-of-neighbors with a single pointer jump
// (cc_in[cc_in[i]]) for logarithmic convergence. Sets *changed when any label
// drops. Non-core rows are copied through unchanged.
template <typename T>
__global__ void cc_step_kernel(const T* __restrict__ points, const T* __restrict__ eps_inv2,
                               const uint8_t* __restrict__ is_core,
                               const int64_t* __restrict__ cc_in, int64_t* __restrict__ cc_out,
                               int64_t n, int64_t f, int* __restrict__ changed) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (is_core[i] == 0) {
    cc_out[i] = cc_in[i];
    return;
  }
  const int64_t self = cc_in[i];
  int64_t m = self;
  const int64_t jumped = cc_in[self];  // self is a valid core row index
  if (jumped < m) {
    m = jumped;
  }
  const T* pi = points + i * f;
  for (int64_t j = 0; j < n; ++j) {
    if (j == i || is_core[j] == 0) {
      continue;
    }
    if (dbscan_neighbor<T>(pi, points + j * f, eps_inv2, f)) {
      const int64_t lj = cc_in[j];
      if (lj < m) {
        m = lj;
      }
    }
  }
  cc_out[i] = m;
  if (m != self) {
    atomicExch(changed, 1);
  }
}

// Each non-core point joins the cluster of its lowest-root core neighbor (if
// any); points with no core neighbor stay noise. Operates in place: thread i
// writes only its own (non-core) row and reads finalized core roots.
template <typename T>
__global__ void assign_border_kernel(const T* __restrict__ points, const T* __restrict__ eps_inv2,
                                     const uint8_t* __restrict__ is_core, int64_t* __restrict__ cc,
                                     int64_t n, int64_t f) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n || is_core[i] != 0) {
    return;
  }
  const T* pi = points + i * f;
  int64_t best_root = kNoiseRoot;
  for (int64_t j = 0; j < n; ++j) {
    if (is_core[j] == 0) {
      continue;
    }
    if (dbscan_neighbor<T>(pi, points + j * f, eps_inv2, f)) {
      const int64_t root = cc[j];
      if (best_root == kNoiseRoot || root < best_root) {
        best_root = root;
      }
    }
  }
  cc[i] = best_root;
}

// Flag component roots: a core row whose label points to itself. 1 → root.
__global__ inline void root_flag_kernel(const uint8_t* __restrict__ is_core,
                                        const int64_t* __restrict__ cc,
                                        uint32_t* __restrict__ root_flag, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  root_flag[i] = (is_core[i] != 0 && cc[i] == i) ? 1u : 0u;
}

// Map each point's root to its dense cluster id (exclusive scan of root_flag at
// the root row); noise rows → -1.
__global__ inline void finalize_labels_kernel(const int64_t* __restrict__ cc,
                                              const uint32_t* __restrict__ dense_of_root,
                                              int64_t* __restrict__ labels, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t root = cc[i];
  labels[i] = (root < 0) ? kNoiseRoot : static_cast<int64_t>(dense_of_root[root]);
}

// Renumber cluster ids through a permutation (remap[old id] = new id); noise
// rows (-1) pass through unchanged. Used to reorder ids by descending size.
__global__ inline void remap_labels_kernel(int64_t* __restrict__ labels,
                                           const int64_t* __restrict__ remap, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t l = labels[i];
  if (l >= 0) {
    labels[i] = remap[l];
  }
}

// Drop clusters whose dense id is at or beyond the cap → relabel to noise.
__global__ inline void cap_labels_kernel(int64_t* __restrict__ labels, int64_t cap, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (labels[i] >= cap) {
    labels[i] = kNoiseRoot;
  }
}

// Per-cluster point counts (sizes[label] += 1) over the non-noise rows.
__global__ inline void histogram_kernel(const int64_t* __restrict__ labels,
                                        int64_t* __restrict__ sizes, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t l = labels[i];
  if (l >= 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(sizes) + l,  // NOLINT
              1ull);
  }
}

// ---- Host launch wrappers ------------------------------------------------

template <typename T>
void launch_mark_core(const T* points, const T* eps_inv2, int64_t n, int64_t f, int64_t min_points,
                      uint8_t* is_core, int64_t* cc, cudaStream_t stream) {
  mark_core_kernel<T>
      <<<grid_for(n), 256, 0, stream>>>(points, eps_inv2, n, f, min_points, is_core, cc);
}

template <typename T>
void launch_cc_step(const T* points, const T* eps_inv2, const uint8_t* is_core,
                    const int64_t* cc_in, int64_t* cc_out, int64_t n, int64_t f, int* changed,
                    cudaStream_t stream) {
  cc_step_kernel<T>
      <<<grid_for(n), 256, 0, stream>>>(points, eps_inv2, is_core, cc_in, cc_out, n, f, changed);
}

template <typename T>
void launch_assign_border(const T* points, const T* eps_inv2, const uint8_t* is_core, int64_t* cc,
                          int64_t n, int64_t f, cudaStream_t stream) {
  assign_border_kernel<T><<<grid_for(n), 256, 0, stream>>>(points, eps_inv2, is_core, cc, n, f);
}

inline void launch_root_flag(const uint8_t* is_core, const int64_t* cc, uint32_t* root_flag,
                             int64_t n, cudaStream_t stream) {
  root_flag_kernel<<<grid_for(n), 256, 0, stream>>>(is_core, cc, root_flag, n);
}

inline void launch_finalize_labels(const int64_t* cc, const uint32_t* dense_of_root,
                                   int64_t* labels, int64_t n, cudaStream_t stream) {
  finalize_labels_kernel<<<grid_for(n), 256, 0, stream>>>(cc, dense_of_root, labels, n);
}

inline void launch_remap_labels(int64_t* labels, const int64_t* remap, int64_t n,
                                cudaStream_t stream) {
  remap_labels_kernel<<<grid_for(n), 256, 0, stream>>>(labels, remap, n);
}

inline void launch_cap_labels(int64_t* labels, int64_t cap, int64_t n, cudaStream_t stream) {
  cap_labels_kernel<<<grid_for(n), 256, 0, stream>>>(labels, cap, n);
}

inline void launch_histogram(const int64_t* labels, int64_t* sizes, int64_t n,
                             cudaStream_t stream) {
  histogram_kernel<<<grid_for(n), 256, 0, stream>>>(labels, sizes, n);
}

// CUB exclusive scan of the uint32 root flags into dense cluster ids. Caller
// provides temp storage sized by a prior nullptr query (cub_scan_temp_bytes).
inline size_t cub_scan_temp_bytes(int64_t n) {
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                static_cast<uint32_t*>(nullptr), static_cast<int>(n));
  return bytes;
}

inline void launch_scan_root_ids(void* temp, size_t temp_bytes, const uint32_t* root_flag,
                                 uint32_t* dense, int64_t n, cudaStream_t stream) {
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, root_flag, dense, static_cast<int>(n), stream);
}

}  // namespace holoscan::radar::cluster_dbscan
