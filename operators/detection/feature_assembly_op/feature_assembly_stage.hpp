// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::radar::detail {

// Device-side feature token ids (match AssemblyFeature enum order).
constexpr int kFeatRange = 0;
constexpr int kFeatVelocity = 1;
constexpr int kFeatScore = 2;
constexpr int kFeatAz = 3;
constexpr int kFeatEl = 4;
constexpr int kFeatDopplerBin = 5;
constexpr int kFeatRangeBin = 6;

// Assemble `points[row, f]` from detection_idx and optional sources.
// detection_idx columns: (…preserved_batch, doppler_bin, range_bin).
// score_plane_offsets[row] is the element offset to the start of that row's
// (D, R) plane within score_map (0 when there are no preserved batch dims).
template <typename T>
__global__ void assemble_features_kernel(
    const int64_t* __restrict__ detection_idx, int64_t num_points,
    int64_t idx_cols, int64_t n_preserved, const T* __restrict__ range_bins,
    int64_t num_range, bool have_range_bins,
    const T* __restrict__ velocity_bins, int64_t num_doppler,
    bool have_velocity_bins, const T* __restrict__ score_map,
    int64_t num_score_range, const int64_t* __restrict__ score_plane_offsets,
    bool have_score, const T* __restrict__ aoa, bool have_aoa,
    const int* __restrict__ feature_ids, int64_t num_features,
    T* __restrict__ points) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (row >= num_points) {
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
        if (have_range_bins && range >= 0 && range < num_range) {
          value = range_bins[range];
        } else {
          value = static_cast<T>(range);
        }
        break;
      case kFeatVelocity:
        if (have_velocity_bins && doppler >= 0 && doppler < num_doppler) {
          value = velocity_bins[doppler];
        } else {
          value = static_cast<T>(doppler);
        }
        break;
      case kFeatScore:
        if (have_score) {
          const int64_t plane = score_plane_offsets != nullptr
                                    ? score_plane_offsets[row]
                                    : int64_t{0};
          value = score_map[plane + doppler * num_score_range + range];
        }
        break;
      case kFeatAz:
        if (have_aoa) {
          value = aoa[row * 2];
        }
        break;
      case kFeatEl:
        if (have_aoa) {
          value = aoa[row * 2 + 1];
        }
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
    bool have_velocity_bins, const T* score_map, int64_t num_score_range,
    const int64_t* score_plane_offsets, bool have_score, const T* aoa,
    bool have_aoa, const int* feature_ids, int64_t num_features, T* points,
    cudaStream_t stream) {
  if (num_points <= 0 || num_features <= 0) {
    return;
  }
  const int block = 256;
  const int grid =
      static_cast<int>((num_points + block - 1) / static_cast<int64_t>(block));
  assemble_features_kernel<T><<<grid, block, 0, stream>>>(
      detection_idx, num_points, idx_cols, n_preserved, range_bins, num_range,
      have_range_bins, velocity_bins, num_doppler, have_velocity_bins,
      score_map, num_score_range, score_plane_offsets, have_score, aoa,
      have_aoa, feature_ids, num_features, points);
}

}  // namespace holoscan::radar::detail
