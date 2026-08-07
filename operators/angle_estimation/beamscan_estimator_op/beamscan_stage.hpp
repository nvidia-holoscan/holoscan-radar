// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Beamscan (conventional / Bartlett) stage.
//
// The Bartlett power for detection i, beam b is
//   P[i, b] = | Σ_c conj(w[b, c]) · x[i, c] |²
// i.e. the elementwise magnitude² of the matrix product X · Wᶜ, where
//   X  = snapshots           [num_items,    num_channels]
//   Wᶜ = conj(w)ᵀ            [num_channels, num_beams]   (built host-side)
//   Y  = X · Wᶜ              [num_items,    num_beams]
// Casting the beamformer as one complex GEMM (cuBLAS, via MatX) is what makes
// this fast: the steering table is streamed once and reused across every
// detection through the GEMM's tiling, instead of being re-read per block.
// A light second kernel then turns Y into the |·|² spectrum (in the
// (num_az, num_el) layout), and picks the per-item peak (argmax + optional
// parabolic sub-grid refinement).
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

__host__ __device__ inline int peak_interp_id(PeakInterpolation mode) {
  switch (mode) {
    case PeakInterpolation::kNone: return 0;
    case PeakInterpolation::kParabolic: return 1;
  }
  return 0;
}

template <typename T>
__device__ inline T parabolic_offset(T left, T center, T right) {
  const T denom = left - T{2} * center + right;
  if (fabs(static_cast<double>(denom)) < 1e-12) {
    return T{0};
  }
  return T{0.5} * (left - right) / denom;
}

// spectrum is stored (num_az, num_el) per item; beams are enumerated
// el-major (beam = ie * num_az + ia) to match the host steering-table build.
template <typename T>
__device__ inline T& spectrum_at(T* spec, int64_t ia, int64_t ie,
                                 int64_t num_el) {
  return spec[ia * num_el + ie];
}

// One block per detection item. Threads cooperate across the beam grid: each
// reads its slice of the GEMM output Y[item, beam], writes |Y|² into the
// (num_az, num_el) spectrum layout, and keeps a running argmax; a block-level
// reduction (lowest beam index wins ties, matching an ascending serial scan)
// then selects the peak and (optionally) refines it parabolically.
// Dynamic shared layout: [power T[blockDim]] then [index int64_t[blockDim]].
template <typename T>
__global__ void beamscan_reduce_kernel(
    const cuda::std::complex<T>* __restrict__ beamformed, int64_t num_items,
    int64_t num_beams, int64_t num_az, int64_t num_el,
    const T* __restrict__ az_grid, const T* __restrict__ el_grid,
    int peak_interp, T* __restrict__ spectrum, T* __restrict__ aoa) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  if (item >= num_items) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem[];
  auto* red_pow = reinterpret_cast<T*>(smem);
  auto* red_idx =
      reinterpret_cast<int64_t*>(red_pow + static_cast<int64_t>(blockDim.x));

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  const cuda::std::complex<T>* y = beamformed + item * num_beams;
  T* spec = spectrum + item * num_beams;

  T best_power = T{0};
  int64_t best_beam = -1;
  for (int64_t beam = tid; beam < num_beams; beam += nthreads) {
    const cuda::std::complex<T> v = y[beam];
    const T power = v.real() * v.real() + v.imag() * v.imag();
    const int64_t ia = beam % num_az;
    const int64_t ie = beam / num_az;
    spectrum_at(spec, ia, ie, num_el) = power;
    if (power > best_power) {
      best_power = power;
      best_beam = beam;
    }
  }

  red_pow[tid] = best_power;
  red_idx[tid] = best_beam;
  __syncthreads();

  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const T other_pow = red_pow[tid + stride];
      const int64_t other_idx = red_idx[tid + stride];
      const bool take = other_pow > red_pow[tid] ||
                        (other_pow == red_pow[tid] && other_idx >= 0 &&
                         (red_idx[tid] < 0 || other_idx < red_idx[tid]));
      if (take) {
        red_pow[tid] = other_pow;
        red_idx[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  if (tid != 0) {
    return;
  }

  best_power = red_pow[0];
  best_beam = red_idx[0];

  T out0 = T{0};
  T out1 = T{0};
  if (best_beam < 0 || best_power <= T{0}) {
    out0 = T{0} / T{0};
    out1 = out0;
  } else {
    const int64_t ia = best_beam % num_az;
    const int64_t ie = best_beam / num_az;
    T az = az_grid[ia];
    T el = el_grid[ie];
    if (peak_interp == peak_interp_id(PeakInterpolation::kParabolic) &&
        num_az >= 3 && num_el >= 3 && ia > 0 && ia + 1 < num_az && ie > 0 &&
        ie + 1 < num_el) {
      const T az_step_left = az_grid[ia] - az_grid[ia - 1];
      const T az_step_right = az_grid[ia + 1] - az_grid[ia];
      const T el_step_left = el_grid[ie] - el_grid[ie - 1];
      const T el_step_right = el_grid[ie + 1] - el_grid[ie];
      const T off_az =
          parabolic_offset(spectrum_at(spec, ia - 1, ie, num_el),
                           spectrum_at(spec, ia, ie, num_el),
                           spectrum_at(spec, ia + 1, ie, num_el));
      const T off_el =
          parabolic_offset(spectrum_at(spec, ia, ie - 1, num_el),
                           spectrum_at(spec, ia, ie, num_el),
                           spectrum_at(spec, ia, ie + 1, num_el));
      az += off_az * (az_step_left + az_step_right) * T{0.5};
      el += off_el * (el_step_left + el_step_right) * T{0.5};
    }
    out0 = az;
    out1 = el;
  }
  aoa[item * 2] = out0;
  aoa[item * 2 + 1] = out1;
}

// Run the full beamscan for one tick: the X · Wᶜ GEMM (into the caller's
// `beamformed` scratch) followed by the power/argmax/refine kernel. All work
// is enqueued on `stream`; no synchronization.
//
//   snapshots   — device ptr, [num_items, num_channels] cuda::std::complex<T>
//                 (a 3-D [num_items, num_channels, 1] input is contiguous and
//                 aliases the same 2-D layout).
//   weights_ct  — device ptr, [num_channels, num_beams] cuda::std::complex<T>,
//                 the conjugate-transposed steering table (invalid beams are
//                 zero, so their power is zero without a separate mask).
//   beamformed  — device ptr, [num_items, num_beams] cuda::std::complex<T>
//                 scratch for the GEMM output.
template <typename T>
inline void run_beamscan_stage(const cuda::std::complex<T>* snapshots,
                               int64_t num_items, int64_t num_channels,
                               const cuda::std::complex<T>* weights_ct,
                               cuda::std::complex<T>* beamformed,
                               int64_t num_az, int64_t num_el, const T* az_grid,
                               const T* el_grid, PeakInterpolation peak_interp,
                               T* spectrum, T* aoa, cudaStream_t stream) {
  if (num_items <= 0) {
    return;
  }
  const int64_t num_beams = num_az * num_el;

  auto x_view = matx::make_tensor<cuda::std::complex<T>, 2>(
      const_cast<cuda::std::complex<T>*>(snapshots), {num_items, num_channels});
  auto w_view = matx::make_tensor<cuda::std::complex<T>, 2>(
      const_cast<cuda::std::complex<T>*>(weights_ct),
      {num_channels, num_beams});
  auto y_view =
      matx::make_tensor<cuda::std::complex<T>, 2>(beamformed,
                                                  {num_items, num_beams});
  matx::cudaExecutor exec{stream};
  (y_view = matx::matmul(x_view, w_view)).run(exec);

  // Fixed power-of-two block: the argmax tree reduction halves the stride, so
  // a non-power-of-two width would drop lanes. 256 keeps enough threads
  // resident to hide latency; the small waste on tiny grids is cheap.
  constexpr int block = 256;
  const int grid = static_cast<int>(num_items);
  const size_t smem =
      static_cast<size_t>(block) * (sizeof(T) + sizeof(int64_t));
  beamscan_reduce_kernel<T><<<grid, block, smem, stream>>>(
      beamformed, num_items, num_beams, num_az, num_el, az_grid, el_grid,
      peak_interp_id(peak_interp), spectrum, aoa);
}

}  // namespace holoscan::radar::detail
