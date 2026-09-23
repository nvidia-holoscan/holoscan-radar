// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CFAR detection stage: the kernels behind CFARDetectorOp.
//
// Design split (validated by benchmark, see PR discussion): MatX forms the
// square-law power map (matx::abs2 — a natural element-wise fit), then a
// single fused CUDA stencil computes the windowed noise estimate, adaptive
// threshold, score, and mask in one launch. CFAR's noise estimate is a
// windowed order/sum statistic, not a signal convolution; expressing it as
// matx::conv2d measured 2.5-13x slower (extra passes, scratch, and a
// redundant count convolution) than the fused stencil, and the OS variant
// (per-cell selection) cannot be expressed in MatX at all — so the stencil
// is the uniform home for all four variants.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "detection/cfar_detector_op/cfar_alpha.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Fully resolved kernel and output policy for one operator tick. Fields follow
// CFARParams order; live values have already been merged over pinned Params.
struct CfarStageConfig {
  CfarVariant cfar_variant;
  int64_t os_rank;
  std::vector<int64_t> guard_cells;
  std::vector<int64_t> training_cells;
  double pfa;
  std::vector<double> alpha_lut;
  double threshold_scale;
  std::vector<int64_t> nci_axes;
  int64_t preintegrated_looks;
  int64_t max_detections;
  bool blocking_alpha_lut_rebuild;
  CfarPeakMode peak_mode;
  std::vector<int64_t> peak_radius;
};

// Sum power along integrated leading axes using a host-built lookup table.
// in_power is (batch_in, doppler, range); out_power is (out_batch, doppler,
// range).
template <typename T>
__global__ void nci_sum_kernel(const T* in_power, T* out_power,
                               const int64_t* batch_in_table, int64_t out_batch,
                               int64_t fold, int64_t num_doppler,
                               int64_t num_range) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t d = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t b_out = blockIdx.z;
  if (r >= num_range || d >= num_doppler || b_out >= out_batch) {
    return;
  }

  const int64_t plane = num_doppler * num_range;
  const int64_t out_idx = b_out * plane + d * num_range + r;
  T sum = T{0};
  const int64_t base = b_out * fold;
  for (int64_t k = 0; k < fold; ++k) {
    const int64_t b_in = batch_in_table[base + k];
    sum += in_power[b_in * plane + d * num_range + r];
  }
  out_power[out_idx] = sum;
}

// CA-CFAR over the (doppler, range) plane for each batch slice, fully fused:
// one thread per cell computes the training-cell sum and (border-correct)
// count, the noise estimate, the Pfa-derived multiplier alpha, the score,
// and the boolean mask in a single launch.
//
//   Pfa = I_{1/(1 + alpha/N)}(N*K, K)  [regularized incomplete beta]
//   alpha(N) is precomputed on the host and passed as alpha_lut[N].
//   threshold = threshold_scale * alpha * noise
//   score = power / threshold ; mask = score > 1
template <typename T>
__global__ void cfar_ca_kernel(const T* power, int64_t batch,
                               int64_t num_doppler, int64_t num_range,
                               int guard_d, int guard_r, int train_d,
                               int train_r, T threshold_scale,
                               const T* alpha_lut, int64_t alpha_lut_size,
                               T* score, T* noise, bool* mask) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t d = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t b = blockIdx.z;
  if (r >= num_range || d >= num_doppler || b >= batch) {
    return;
  }

  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  const int64_t plane = num_doppler * num_range;
  const T* p = power + b * plane;

  T train_sum = T{0};
  int64_t train_cnt = 0;
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const int64_t nd = d + dd;
    if (nd < 0 || nd >= num_doppler) {
      continue;
    }
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      const int64_t nr = r + rr;
      if (nr < 0 || nr >= num_range) {
        continue;
      }
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      train_sum += p[nd * num_range + nr];
      ++train_cnt;
    }
  }

  const int64_t idx = b * plane + d * num_range + r;
  if (train_cnt == 0) {
    noise[idx] = T{0};
    score[idx] = T{0};
    mask[idx] = false;
    return;
  }
  const T n = static_cast<T>(train_cnt);
  const T noise_est = train_sum / n;
  // train_cnt > 0 here (the train_cnt == 0 case returned above).
  const T alpha =
      train_cnt < alpha_lut_size ? alpha_lut[train_cnt] : T{0};
  const T threshold = threshold_scale * alpha * noise_est;
  noise[idx] = noise_est;
  const T s =
      threshold > T{0} ? p[d * num_range + r] / threshold : T{0};
  score[idx] = s;
  mask[idx] = s > T{1};
}

// GO-CFAR (2-D GOGO): split training cells by range sign (rr) and Doppler
// sign (dd) relative to the CUT, form range_go = max(mean_neg, mean_pos) and
// doppler_go = max(mean_neg, mean_pos) on each axis (single-sided at borders),
// then noise_estimate = max(range_go, doppler_go). Alpha LUT is indexed by
// go_lut_index(range_neg, range_pos, doppler_neg, doppler_pos).
__device__ int64_t go_lut_side_index_dev(int64_t neg_count, int64_t pos_count) {
  if (neg_count > 0 && pos_count > 0) {
    return neg_count < pos_count ? neg_count : pos_count;
  }
  return neg_count > pos_count ? neg_count : pos_count;
}

__device__ int64_t go_lut_index_dev(int64_t range_neg, int64_t range_pos,
                                    int64_t doppler_neg, int64_t doppler_pos) {
  const int64_t nr = go_lut_side_index_dev(range_neg, range_pos);
  const int64_t nd = go_lut_side_index_dev(doppler_neg, doppler_pos);
  if (nr > 0 && nd > 0) {
    return nr < nd ? nr : nd;
  }
  return nr > nd ? nr : nd;
}

template <typename T, bool kGreatestOf>
__device__ T goso_side_statistic(T neg_sum, int64_t neg_cnt, T pos_sum,
                                 int64_t pos_cnt) {
  if (neg_cnt > 0 && pos_cnt > 0) {
    const T neg_mean = neg_sum / static_cast<T>(neg_cnt);
    const T pos_mean = pos_sum / static_cast<T>(pos_cnt);
    if (kGreatestOf) {
      return neg_mean > pos_mean ? neg_mean : pos_mean;
    }
    return neg_mean < pos_mean ? neg_mean : pos_mean;
  }
  if (neg_cnt > 0) {
    return neg_sum / static_cast<T>(neg_cnt);
  }
  if (pos_cnt > 0) {
    return pos_sum / static_cast<T>(pos_cnt);
  }
  return T{0};
}

template <typename T, bool kGreatestOf>
__global__ void cfar_goso_kernel(const T* power, int64_t batch,
                                 int64_t num_doppler, int64_t num_range,
                                 int guard_d, int guard_r, int train_d,
                                 int train_r, T threshold_scale,
                                 const T* alpha_lut, int64_t alpha_lut_size,
                                 T* score, T* noise, bool* mask) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t d = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t b = blockIdx.z;
  if (r >= num_range || d >= num_doppler || b >= batch) {
    return;
  }

  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  const int64_t plane = num_doppler * num_range;
  const T* p = power + b * plane;

  T range_neg_sum = T{0};
  T range_pos_sum = T{0};
  T doppler_neg_sum = T{0};
  T doppler_pos_sum = T{0};
  int64_t range_neg_cnt = 0;
  int64_t range_pos_cnt = 0;
  int64_t doppler_neg_cnt = 0;
  int64_t doppler_pos_cnt = 0;

  for (int dd = -half_d; dd <= half_d; ++dd) {
    const int64_t nd = d + dd;
    if (nd < 0 || nd >= num_doppler) {
      continue;
    }
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      const int64_t nr = r + rr;
      if (nr < 0 || nr >= num_range) {
        continue;
      }
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      const T cell = p[nd * num_range + nr];
      if (rr < 0) {
        range_neg_sum += cell;
        ++range_neg_cnt;
      } else if (rr > 0) {
        range_pos_sum += cell;
        ++range_pos_cnt;
      }
      if (dd < 0) {
        doppler_neg_sum += cell;
        ++doppler_neg_cnt;
      } else if (dd > 0) {
        doppler_pos_sum += cell;
        ++doppler_pos_cnt;
      }
    }
  }

  const int64_t idx = b * plane + d * num_range + r;
  const int64_t lut_idx = go_lut_index_dev(range_neg_cnt, range_pos_cnt,
                                           doppler_neg_cnt, doppler_pos_cnt);
  if (lut_idx == 0) {
    noise[idx] = T{0};
    score[idx] = T{0};
    mask[idx] = false;
    return;
  }

  const T range_stat =
      goso_side_statistic<T, kGreatestOf>(range_neg_sum, range_neg_cnt,
                                          range_pos_sum, range_pos_cnt);
  const T doppler_stat =
      goso_side_statistic<T, kGreatestOf>(doppler_neg_sum, doppler_neg_cnt,
                                          doppler_pos_sum, doppler_pos_cnt);
  const T noise_est = kGreatestOf ? (range_stat > doppler_stat ? range_stat
                                                               : doppler_stat)
                                  : (range_stat < doppler_stat ? range_stat
                                                               : doppler_stat);
  const T alpha = lut_idx < alpha_lut_size ? alpha_lut[lut_idx] : T{0};
  const T threshold = threshold_scale * alpha * noise_est;
  noise[idx] = noise_est;
  const T s =
      threshold > T{0} ? p[d * num_range + r] / threshold : T{0};
  score[idx] = s;
  mask[idx] = s > T{1};
}

// OS-CFAR: noise_estimate is the os_rank-th order statistic (0-based,
// ascending) of training-cell powers in the CA window footprint. When
// train_cnt <= os_rank, the maximum available rank (train_cnt - 1) is used.

template <typename T>
__device__ T os_select_rank(T* samples, int count, int64_t os_rank) {
  if (count <= 0) {
    return T{0};
  }
  const int rank =
      os_rank < static_cast<int64_t>(count) ? static_cast<int>(os_rank)
                                           : count - 1;
  for (int i = 1; i < count; ++i) {
    const T key = samples[i];
    int j = i - 1;
    while (j >= 0 && samples[j] > key) {
      samples[j + 1] = samples[j];
      --j;
    }
    samples[j + 1] = key;
  }
  return samples[rank];
}

template <typename T>
__global__ void cfar_os_kernel(const T* power, int64_t batch,
                               int64_t num_doppler, int64_t num_range,
                               int guard_d, int guard_r, int train_d,
                               int train_r, int64_t os_rank, T threshold_scale,
                               const T* alpha_lut, int64_t alpha_lut_size,
                               T* score, T* noise, bool* mask) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t d = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t b = blockIdx.z;
  if (r >= num_range || d >= num_doppler || b >= batch) {
    return;
  }

  const int half_d = guard_d + train_d;
  const int half_r = guard_r + train_r;
  const int64_t plane = num_doppler * num_range;
  const T* p = power + b * plane;

  T train_samples[kMaxOsTrainingCells];
  int train_cnt = 0;
  for (int dd = -half_d; dd <= half_d; ++dd) {
    const int64_t nd = d + dd;
    if (nd < 0 || nd >= num_doppler) {
      continue;
    }
    const bool guard_row = (dd >= -guard_d && dd <= guard_d);
    for (int rr = -half_r; rr <= half_r; ++rr) {
      const int64_t nr = r + rr;
      if (nr < 0 || nr >= num_range) {
        continue;
      }
      if (guard_row && rr >= -guard_r && rr <= guard_r) {
        continue;
      }
      // Defensive bound: compute() rejects footprints whose training-cell
      // count exceeds kMaxOsTrainingCells, so this is belt-and-suspenders
      // against a stack-buffer overrun if that guard is ever bypassed.
      if (train_cnt >= kMaxOsTrainingCells) {
        continue;
      }
      train_samples[train_cnt++] = p[nd * num_range + nr];
    }
  }

  const int64_t idx = b * plane + d * num_range + r;
  if (train_cnt == 0) {
    noise[idx] = T{0};
    score[idx] = T{0};
    mask[idx] = false;
    return;
  }

  const T noise_est = os_select_rank(train_samples, train_cnt, os_rank);
  const T alpha = train_cnt < alpha_lut_size ? alpha_lut[train_cnt] : T{0};
  const T threshold = threshold_scale * alpha * noise_est;
  noise[idx] = noise_est;
  const T s =
      threshold > T{0} ? p[d * num_range + r] / threshold : T{0};
  score[idx] = s;
  mask[idx] = s > T{1};
}

// Peak-grouping (non-maximum suppression) over the (doppler, range) plane.
//
// Refines the raw CFAR mask into a "one point per contiguous above-threshold
// blob" mask: a detected cell survives only if its CFAR statistic (power) is a
// local maximum over its neighborhood, restricted to *detected* neighbors (the
// contiguous above-threshold region). Undetected neighbors are ignored, so a
// separate weaker target nearby cannot suppress a real peak.
//
//   full8 = true  → 8-connected neighborhood (the 8 surrounding cells)
//   full8 = false → 4-connected (2 range + 2 Doppler orthogonal neighbors)
//
// Ties are broken by raster order: on exactly-equal power, the lower
// within-plane linear index (d*num_range + r) wins, so a flat plateau collapses
// to its lowest-index cell. (Pathological non-convex equal-power plateaus may
// retain more than one cell; exact ties on a real-valued RD map are effectively
// measure-zero.)
//
// power / mask / peak_mask share the (batch, num_doppler, num_range) contiguous
// layout; peak_mask must be a distinct buffer from mask (the kernel reads
// neighbor mask bits, so an in-place refinement would race).
template <typename T>
__global__ void cfar_peak_suppress_kernel(const T* power, const bool* mask,
                                          bool* peak_mask, int64_t batch,
                                          int64_t num_doppler,
                                          int64_t num_range,
                                          int64_t radius_doppler,
                                          int64_t radius_range) {
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t d = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t b = blockIdx.z;
  if (r >= num_range || d >= num_doppler || b >= batch) {
    return;
  }

  const int64_t plane = num_doppler * num_range;
  const int64_t lin_c = d * num_range + r;
  const int64_t idx = b * plane + lin_c;
  if (!mask[idx]) {
    peak_mask[idx] = false;
    return;
  }

  const T pc = power[idx];
  bool survive = true;
  const int64_t dd_min = -(radius_doppler < d ? radius_doppler : d);
  const int64_t doppler_right = num_doppler - 1 - d;
  const int64_t dd_max =
      radius_doppler < doppler_right ? radius_doppler : doppler_right;
  const int64_t rr_min = -(radius_range < r ? radius_range : r);
  const int64_t range_right = num_range - 1 - r;
  const int64_t rr_max =
      radius_range < range_right ? radius_range : range_right;
  for (int64_t dd = dd_min; dd <= dd_max && survive; ++dd) {
    const int64_t nd = d + dd;
    for (int64_t rr = rr_min; rr <= rr_max; ++rr) {
      if (dd == 0 && rr == 0) {
        continue;
      }
      const int64_t nr = r + rr;
      const int64_t neighbor_lin = nd * num_range + nr;
      const int64_t neighbor_idx = b * plane + neighbor_lin;
      if (!mask[neighbor_idx]) {
        continue;  // only compare against detected neighbors.
      }
      const T pn = power[neighbor_idx];
      // Cell loses to a strictly greater
      // neighbor, or to an equal-power neighbor
      // with a lower raster index (raster tiebreak → one survivor per plateau).
      if (pn > pc || (pn == pc && neighbor_lin < lin_c)) {
        survive = false;
        break;
      }
    }
  }
  peak_mask[idx] = survive;
}

// Configurable hyperrectangular N-D NMS for rank 2-4 output tensors.
template <typename T, int Rank>
__global__ void cfar_peak_suppress_all_kernel(
    const T* power, const bool* mask, bool* peak_mask, int64_t total,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t radius0, int64_t radius1, int64_t radius2, int64_t radius3) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  if (!mask[index]) {
    peak_mask[index] = false;
    return;
  }

  const int64_t dims[4] = {dim0, dim1, dim2, dim3};
  const int64_t radii[4] = {radius0, radius1, radius2, radius3};
  int64_t coordinates[4] = {};
  int64_t remainder = index;
#pragma unroll
  for (int axis = Rank - 1; axis >= 0; --axis) {
    coordinates[axis] = remainder % dims[axis];
    remainder /= dims[axis];
  }

  int64_t offset_min[4] = {};
  int64_t widths[4] = {};
  int64_t neighbor_count = 1;
#pragma unroll
  for (int axis = 0; axis < Rank; ++axis) {
    const int64_t left =
        radii[axis] < coordinates[axis] ? radii[axis] : coordinates[axis];
    const int64_t right_space = dims[axis] - 1 - coordinates[axis];
    const int64_t right =
        radii[axis] < right_space ? radii[axis] : right_space;
    offset_min[axis] = -left;
    widths[axis] = left + right + 1;
    neighbor_count *= widths[axis];
  }
  const T center_power = power[index];
  bool survive = true;
  for (int64_t code = 0; code < neighbor_count && survive; ++code) {
    int64_t remaining_code = code;
    bool is_center = true;
    int64_t neighbor_index = 0;
    int64_t offsets[4] = {};
#pragma unroll
    for (int axis = Rank - 1; axis >= 0; --axis) {
      offsets[axis] =
          offset_min[axis] + remaining_code % widths[axis];
      remaining_code /= widths[axis];
      is_center = is_center && offsets[axis] == 0;
    }
    if (is_center) {
      continue;
    }
#pragma unroll
    for (int axis = 0; axis < Rank; ++axis) {
      neighbor_index =
          neighbor_index * dims[axis] + coordinates[axis] + offsets[axis];
    }
    if (!mask[neighbor_index]) {
      continue;
    }
    const T neighbor_power = power[neighbor_index];
    if (neighbor_power > center_power ||
        (neighbor_power == center_power && neighbor_index < index)) {
      survive = false;
    }
  }
  peak_mask[index] = survive;
}

// Run the CFAR stage over a collapsed batch of (doppler, range) planes.
//
//   in     — device pointer, (batch_in, num_doppler, num_range).
//            input_is_complex → cuda::std::complex<T> (square-law power is
//            formed internally via MatX); else T (already a power map).
//   score  — device pointer, T,     (out_batch, num_doppler, num_range).
//   noise  — device pointer, T,     (out_batch, num_doppler, num_range).
//   mask   — device pointer, bool,  (out_batch, num_doppler, num_range).
//   power  — device scratch, T,     (batch_in, num_doppler, num_range).
//   integrated_power — scratch, T,  (out_batch, num_doppler, num_range).
//
// When fold > 1, batch_in_table_dev (a device pointer owned and cached by the
// caller) maps each (out_batch, fold) pair to the corresponding batch_in slice
// before CFAR runs on integrated_power.
//
// All outputs are preallocated by the caller. Work is enqueued on `stream`;
// no synchronization here.
//
// A non-none peak mode refines the raw mask into peak_mask (a distinct
// caller-preallocated bool buffer). range_doppler compares within each trailing
// plane; all compares the configured N-D hyperrectangle. Dense power, score,
// and noise outputs are unaffected.
//
// v1 implements CfarVariant::kCA, kGO, kSO, and kOS.
template <typename T>
void run_cfar_stage(const CfarStageConfig& config, const void* in,
                    bool input_is_complex, int64_t batch_in,
                    int64_t out_batch, int64_t num_doppler, int64_t num_range,
                    const std::vector<int64_t>& output_shape,
                    int64_t nci_fold, const int64_t* batch_in_table_dev,
                    const void* alpha_lut, int64_t alpha_lut_size, void* power,
                    void* integrated_power, void* score, void* noise,
                    void* mask, void* peak_mask, cudaStream_t stream) {
  const CfarVariant variant = config.cfar_variant;
  const int guard_d = static_cast<int>(config.guard_cells[0]);
  const int guard_r = static_cast<int>(config.guard_cells[1]);
  const int train_d = static_cast<int>(config.training_cells[0]);
  const int train_r = static_cast<int>(config.training_cells[1]);
  if (variant != CfarVariant::kCA && variant != CfarVariant::kGO &&
      variant != CfarVariant::kSO && variant != CfarVariant::kOS) {
    throw std::invalid_argument(
        std::string("CFAR stage: variant '") + std::string(to_string(variant)) +
        "' is not implemented; instantiate a supported detector or extend "
        "cfar_stage.hpp");
  }

  const int64_t n = batch_in * num_doppler * num_range;
  if (n == 0) {
    return;
  }

  matx::cudaExecutor exec{stream};
  auto power_view = make_real_view3<T>(power, batch_in, num_doppler, num_range);
  if (input_is_complex) {
    auto in_view =
        make_const_complex_view3<T>(in, batch_in, num_doppler, num_range);
    (power_view = matx::abs2(in_view)).run(exec);
  } else {
    const size_t bytes =
        static_cast<size_t>(batch_in * num_doppler * num_range) * sizeof(T);
    if (cudaMemcpyAsync(power, in, bytes, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess) {
      throw std::runtime_error("run_cfar_stage: power D2D copy failed");
    }
  }

  T* cfar_power = static_cast<T*>(power);
  if (nci_fold > 1) {
    const dim3 block(16, 16, 1);
    const dim3 grid(
        static_cast<unsigned>((num_range + block.x - 1) / block.x),
        static_cast<unsigned>((num_doppler + block.y - 1) / block.y),
        static_cast<unsigned>(out_batch));
    nci_sum_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T*>(power), static_cast<T*>(integrated_power),
        batch_in_table_dev, out_batch, nci_fold, num_doppler, num_range);
    cfar_power = static_cast<T*>(integrated_power);
  } else if (out_batch != batch_in) {
    throw std::invalid_argument(
        "run_cfar_stage: out_batch != batch_in requires nci_fold > 1");
  }

  const T* alpha_lut_t = static_cast<const T*>(alpha_lut);
  const dim3 block(16, 16, 1);
  const dim3 grid(static_cast<unsigned>((num_range + block.x - 1) / block.x),
                  static_cast<unsigned>((num_doppler + block.y - 1) / block.y),
                  1);
  for (int64_t b = 0; b < out_batch; ++b) {
    const int64_t plane = num_doppler * num_range;
    if (variant == CfarVariant::kCA) {
      cfar_ca_kernel<T><<<grid, block, 0, stream>>>(
          cfar_power + b * plane, 1, num_doppler, num_range, guard_d, guard_r,
          train_d, train_r, static_cast<T>(config.threshold_scale), alpha_lut_t,
          alpha_lut_size, static_cast<T*>(score) + b * plane,
          static_cast<T*>(noise) + b * plane,
          static_cast<bool*>(mask) + b * plane);
    } else if (variant == CfarVariant::kGO) {
      cfar_goso_kernel<T, true><<<grid, block, 0, stream>>>(
          cfar_power + b * plane, 1, num_doppler, num_range, guard_d, guard_r,
          train_d, train_r, static_cast<T>(config.threshold_scale), alpha_lut_t,
          alpha_lut_size, static_cast<T*>(score) + b * plane,
          static_cast<T*>(noise) + b * plane,
          static_cast<bool*>(mask) + b * plane);
    } else if (variant == CfarVariant::kSO) {
      cfar_goso_kernel<T, false><<<grid, block, 0, stream>>>(
          cfar_power + b * plane, 1, num_doppler, num_range, guard_d, guard_r,
          train_d, train_r, static_cast<T>(config.threshold_scale), alpha_lut_t,
          alpha_lut_size, static_cast<T*>(score) + b * plane,
          static_cast<T*>(noise) + b * plane,
          static_cast<bool*>(mask) + b * plane);
    } else {
      cfar_os_kernel<T><<<grid, block, 0, stream>>>(
          cfar_power + b * plane, 1, num_doppler, num_range, guard_d, guard_r,
          train_d, train_r, config.os_rank,
          static_cast<T>(config.threshold_scale),
          alpha_lut_t, alpha_lut_size, static_cast<T*>(score) + b * plane,
          static_cast<T*>(noise) + b * plane,
          static_cast<bool*>(mask) + b * plane);
    }
  }

  if (config.peak_mode == CfarPeakMode::kRangeDoppler) {
    // Non-maximum suppression over cfar_power (the map that was thresholded),
    // reading the raw mask and writing the refined peak_mask. One launch over
    // all out_batch planes; peak_mask must be distinct from mask (neighbor
    // reads would otherwise race).
    const dim3 peak_grid(
        static_cast<unsigned>((num_range + block.x - 1) / block.x),
        static_cast<unsigned>((num_doppler + block.y - 1) / block.y),
        static_cast<unsigned>(out_batch));
    cfar_peak_suppress_kernel<T><<<peak_grid, block, 0, stream>>>(
        cfar_power, static_cast<const bool*>(mask),
        static_cast<bool*>(peak_mask), out_batch, num_doppler, num_range,
        config.peak_radius[config.peak_radius.size() - 2],
        config.peak_radius.back());
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("run_cfar_stage: peak-suppress kernel launch "
                               "failed");
    }
  } else if (config.peak_mode == CfarPeakMode::kAll) {
    constexpr int kPeakBlock = 256;
    const int64_t total = out_batch * num_doppler * num_range;
    const int peak_grid = static_cast<int>(
        (total + kPeakBlock - 1) / kPeakBlock);
    if (output_shape.size() < 2U || output_shape.size() > 4U) {
      throw std::invalid_argument(
          "run_cfar_stage: all peak mode requires output rank 2-4");
    }
    const int64_t dim0 = output_shape[0];
    const int64_t dim1 = output_shape[1];
    const int64_t dim2 = output_shape.size() > 2U ? output_shape[2] : 1;
    const int64_t dim3 = output_shape.size() > 3U ? output_shape[3] : 1;
    const int64_t radius0 = config.peak_radius[0];
    const int64_t radius1 = config.peak_radius[1];
    const int64_t radius2 =
        output_shape.size() > 2U ? config.peak_radius[2] : 0;
    const int64_t radius3 =
        output_shape.size() > 3U ? config.peak_radius[3] : 0;
    if (output_shape.size() == 2U) {
      cfar_peak_suppress_all_kernel<T, 2>
          <<<peak_grid, kPeakBlock, 0, stream>>>(
              cfar_power, static_cast<const bool*>(mask),
              static_cast<bool*>(peak_mask), total, dim0, dim1, dim2, dim3,
              radius0, radius1, radius2, radius3);
    } else if (output_shape.size() == 3U) {
      cfar_peak_suppress_all_kernel<T, 3>
          <<<peak_grid, kPeakBlock, 0, stream>>>(
              cfar_power, static_cast<const bool*>(mask),
              static_cast<bool*>(peak_mask), total, dim0, dim1, dim2, dim3,
              radius0, radius1, radius2, radius3);
    } else if (output_shape.size() == 4U) {
      cfar_peak_suppress_all_kernel<T, 4>
          <<<peak_grid, kPeakBlock, 0, stream>>>(
              cfar_power, static_cast<const bool*>(mask),
              static_cast<bool*>(peak_mask), total, dim0, dim1, dim2, dim3,
              radius0, radius1, radius2, radius3);
    }
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error(
          "run_cfar_stage: all-axis peak-suppress kernel launch failed");
    }
  }
}

}  // namespace holoscan::radar::detail
