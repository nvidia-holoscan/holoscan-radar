// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Device pow(pfa, exp) at the instantiation precision.
template <typename T>
__device__ inline T cfar_pow(T base, T exp) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return powf(base, exp);
  } else {
    return pow(base, exp);
  }
}

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
//   alpha(N, K) = N * (pfa^(-1/(N*K)) - 1)   [CA, Gamma(K) cell statistic]
//   threshold = threshold_scale * alpha * noise
//   score = power / threshold ; mask = score > 1
//
// K = 1 recovers the exponential-cell CA formula used when nci_axes = [].
template <typename T>
__global__ void cfar_ca_kernel(const T* power, int64_t batch,
                               int64_t num_doppler, int64_t num_range,
                               int guard_d, int guard_r, int train_d,
                               int train_r, T pfa, T threshold_scale,
                               int64_t nci_fold, T* score, T* noise,
                               bool* mask) {
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
  const T fold = static_cast<T>(nci_fold);
  const T noise_est = train_sum / n;
  const T alpha =
      n * (cfar_pow<T>(pfa, T{-1} / (n * fold)) - T{1});
  const T threshold = threshold_scale * alpha * noise_est;
  noise[idx] = noise_est;
  const T s = threshold > T{0} ? p[b * plane + d * num_range + r] / threshold
                               : T{0};
  score[idx] = s;
  mask[idx] = s > T{1};
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
// v1 implements CfarVariant::kCA only; GO/SO/OS throw std::invalid_argument
// (they are static per instance, so this is a clean per-instance boundary).
template <typename T>
void run_cfar_stage(CfarVariant variant, const void* in, bool input_is_complex,
                    int64_t batch_in, int64_t out_batch, int64_t num_doppler,
                    int64_t num_range, int guard_d, int guard_r, int train_d,
                    int train_r, double pfa, double threshold_scale,
                    int64_t nci_fold, const int64_t* batch_in_table_dev,
                    void* power, void* integrated_power, void* score,
                    void* noise, void* mask, cudaStream_t stream) {
  if (variant != CfarVariant::kCA) {
    throw std::invalid_argument(
        std::string("CFAR stage: variant '") + std::string(to_string(variant)) +
        "' is not implemented yet (v1 implements CA); instantiate a CA "
        "detector or extend cfar_stage.hpp");
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

  const dim3 block(16, 16, 1);
  const dim3 grid(static_cast<unsigned>((num_range + block.x - 1) / block.x),
                  static_cast<unsigned>((num_doppler + block.y - 1) / block.y),
                  1);
  for (int64_t b = 0; b < out_batch; ++b) {
    const int64_t plane = num_doppler * num_range;
    cfar_ca_kernel<T><<<grid, block, 0, stream>>>(
        cfar_power + b * plane, 1, num_doppler, num_range, guard_d, guard_r,
        train_d, train_r, static_cast<T>(pfa), static_cast<T>(threshold_scale),
        nci_fold, static_cast<T*>(score) + b * plane,
        static_cast<T*>(noise) + b * plane,
        static_cast<bool*>(mask) + b * plane);
  }
}

}  // namespace holoscan::radar::detail
