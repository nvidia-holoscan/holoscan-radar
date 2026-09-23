// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <holoscan/core/domain/tensor.hpp>

namespace holoscan::radar::detail {

// Device-side feature token ids (match AssemblyFeature enum order).
constexpr int kFeatRange = 0;
constexpr int kFeatVelocity = 1;
constexpr int kFeatScore = 2;
constexpr int kFeatAz = 3;
constexpr int kFeatEl = 4;
constexpr int kFeatAzEstimate = 5;
constexpr int kFeatElEstimate = 6;
constexpr int kFeatDopplerBin = 7;
constexpr int kFeatRangeBin = 8;
constexpr int kFeatAzBin = 9;
constexpr int kFeatElBin = 10;

// Cap on preserved batch rank for score-plane indexing (CFAR n_preserved).
constexpr int kMaxPreservedBatchDims = 8;

// Contract input is exactly rank 2. Extra leading axes are not a module
// batch (the module named-axis invariant). num_points == 0 is legal.
struct DetectionIdxLayout {
  int64_t num_points{0};
  int64_t idx_cols{0};
  int64_t n_preserved{0};
};

inline DetectionIdxLayout parse_detection_idx_layout(
    const holoscan::Tensor& tensor, const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() != 2 || shape[1] < 2) {
    throw std::invalid_argument(
        std::string("FeatureAssemblyOp: ") + port_name +
        " must have shape (num_points, 2 [+ n_batch])");
  }
  DetectionIdxLayout layout;
  layout.num_points = shape[0];
  layout.idx_cols = shape[1];
  layout.n_preserved = layout.idx_cols - 2;
  if (layout.num_points < 0) {
    throw std::invalid_argument(std::string("FeatureAssemblyOp: ") +
                                port_name + " row count must be >= 0");
  }
  return layout;
}

// Error codes written to *error_flag (0 = ok). Host throws after the kernel.
constexpr int kErrBatchCoord = 1;
constexpr int kErrRangeBin = 2;
constexpr int kErrDopplerBin = 3;
constexpr int kErrScoreCoord = 4;
constexpr int kErrAzBin = 5;
constexpr int kErrElBin = 6;
constexpr int kErrRangeSource = 7;
constexpr int kErrVelocitySource = 8;

// Per-row element offset into a compact (…, D, R) score map from the
// preserved batch columns of detection_idx. Out-of-range batch coords set
// *error_flag and leave offsets[row] unspecified.
// static: header is included from the op .cu and (optionally) stage tests;
// each TU gets a private kernel copy to avoid ODR link errors.
static __global__ void score_plane_offsets_kernel(
    const int64_t* __restrict__ detection_idx, int64_t num_points,
    int64_t idx_cols, int64_t n_preserved,
    const int64_t* __restrict__ leading_dims, int64_t plane_elems,
    int64_t* __restrict__ offsets, int* __restrict__ error_flag) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (row >= num_points) {
    return;
  }
  const int64_t* idx_row = detection_idx + row * idx_cols;
  int64_t linear = 0;
  for (int64_t a = 0; a < n_preserved; ++a) {
    const int64_t coord = idx_row[a];
    const int64_t dim = leading_dims[a];
    if (coord < 0 || coord >= dim) {
      atomicExch(error_flag, kErrBatchCoord);
      return;
    }
    linear = linear * dim + coord;
  }
  offsets[row] = linear * plane_elems;
}

inline void launch_score_plane_offsets(const int64_t* detection_idx,
                                       int64_t num_points, int64_t idx_cols,
                                       int64_t n_preserved,
                                       const int64_t* leading_dims,
                                       int64_t plane_elems, int64_t* offsets,
                                       int* error_flag, cudaStream_t stream) {
  if (num_points <= 0 || n_preserved <= 0) {
    return;
  }
  const int block = 256;
  const int grid =
      static_cast<int>((num_points + block - 1) / static_cast<int64_t>(block));
  score_plane_offsets_kernel<<<grid, block, 0, stream>>>(
      detection_idx, num_points, idx_cols, n_preserved, leading_dims,
      plane_elems, offsets, error_flag);
}

// Assemble `points[row, f]` from detection_idx and optional sources.
// detection_idx columns: (…preserved_batch, doppler_bin, range_bin).
// score_plane_offsets[row] is the element offset to the start of that row's
// (D, R) plane within score_map (nullptr ⇒ plane base 0).
// Physical axis/grid lookups and score gathers are bounds-checked. Raw-bin
// features emit their detection_idx coordinates directly.
template <typename T>
static __global__ void assemble_features_kernel(
    const int64_t* __restrict__ detection_idx, int64_t num_points,
    int64_t idx_cols, int64_t n_preserved, const T* __restrict__ range_bins,
    int64_t num_range, bool have_range_bins,
    const T* __restrict__ velocity_bins, int64_t num_doppler,
    bool have_velocity_bins, const T* __restrict__ score_map,
    int64_t num_score_doppler, int64_t num_score_range,
    const int64_t* __restrict__ score_plane_offsets, bool have_score,
    const T* __restrict__ aoa, bool have_aoa,
    const T* __restrict__ az_grid, int64_t num_az,
    const T* __restrict__ el_grid, int64_t num_el,
    const int* __restrict__ feature_ids, int64_t num_features,
    T* __restrict__ points, int* __restrict__ error_flag) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (row >= num_points) {
    return;
  }
  // Prior kernel (score plane offsets) may have already flagged an error;
  // do not read uninitialized offsets or gather further.
  if (*error_flag != 0) {
    return;
  }

  const int64_t* idx_row = detection_idx + row * idx_cols;
  const int64_t doppler = idx_row[n_preserved];
  const int64_t range = idx_row[n_preserved + 1];

  for (int64_t f = 0; f < num_features; ++f) {
    T value = static_cast<T>(0);
    switch (feature_ids[f]) {
      case kFeatDopplerBin:
        value = static_cast<T>(doppler);
        break;
      case kFeatRangeBin:
        value = static_cast<T>(range);
        break;
      case kFeatRange:
        if (!have_range_bins) {
          atomicExch(error_flag, kErrRangeSource);
          return;
        }
        if (range < 0 || range >= num_range) {
          atomicExch(error_flag, kErrRangeBin);
          return;
        }
        value = range_bins[range];
        break;
      case kFeatVelocity:
        if (!have_velocity_bins) {
          atomicExch(error_flag, kErrVelocitySource);
          return;
        }
        if (doppler < 0 || doppler >= num_doppler) {
          atomicExch(error_flag, kErrDopplerBin);
          return;
        }
        value = velocity_bins[doppler];
        break;
      case kFeatScore:
        if (have_score) {
          if (doppler < 0 || doppler >= num_score_doppler || range < 0 ||
              range >= num_score_range) {
            atomicExch(error_flag, kErrScoreCoord);
            return;
          }
          const int64_t plane = score_plane_offsets != nullptr
                                    ? score_plane_offsets[row]
                                    : int64_t{0};
          value = score_map[plane + doppler * num_score_range + range];
        }
        break;
      case kFeatAz:
        // Dense path: az is the preserved-batch column just before the
        // (doppler_bin, range_bin) pair; look it up in the azimuth scan grid.
        if (az_grid != nullptr) {
          const int64_t az_bin = idx_row[n_preserved - 2];
          if (az_bin < 0 || az_bin >= num_az) {
            atomicExch(error_flag, kErrAzBin);
            return;
          }
          value = az_grid[az_bin];
        }
        break;
      case kFeatEl:
        if (el_grid != nullptr) {
          const int64_t el_bin = idx_row[n_preserved - 1];
          if (el_bin < 0 || el_bin >= num_el) {
            atomicExch(error_flag, kErrElBin);
            return;
          }
          value = el_grid[el_bin];
        }
        break;
      case kFeatAzEstimate:
        if (have_aoa) {
          value = aoa[row * 2];
        }
        break;
      case kFeatElEstimate:
        if (have_aoa) {
          value = aoa[row * 2 + 1];
        }
        break;
      case kFeatAzBin:
        value = static_cast<T>(idx_row[n_preserved - 2]);
        break;
      case kFeatElBin:
        value = static_cast<T>(idx_row[n_preserved - 1]);
        break;
      default:
        break;
    }
    points[row * num_features + f] = value;
  }
}

template <typename T>
void launch_assemble_features(
    const int64_t* detection_idx, int64_t num_points, int64_t idx_cols,
    int64_t n_preserved, const T* range_bins, int64_t num_range,
    bool have_range_bins, const T* velocity_bins, int64_t num_doppler,
    bool have_velocity_bins, const T* score_map, int64_t num_score_doppler,
    int64_t num_score_range,
    const int64_t* score_plane_offsets, bool have_score, const T* aoa,
    bool have_aoa, const T* az_grid, int64_t num_az, const T* el_grid,
    int64_t num_el, const int* feature_ids, int64_t num_features, T* points,
    int* error_flag, cudaStream_t stream) {
  if (num_points <= 0 || num_features <= 0) {
    return;
  }
  const int block = 256;
  const int grid =
      static_cast<int>((num_points + block - 1) / static_cast<int64_t>(block));
  assemble_features_kernel<T><<<grid, block, 0, stream>>>(
      detection_idx, num_points, idx_cols, n_preserved, range_bins, num_range,
      have_range_bins, velocity_bins, num_doppler, have_velocity_bins,
      score_map, num_score_doppler, num_score_range, score_plane_offsets,
      have_score, aoa, have_aoa, az_grid, num_az, el_grid, num_el,
      feature_ids, num_features, points, error_flag);
}

}  // namespace holoscan::radar::detail
