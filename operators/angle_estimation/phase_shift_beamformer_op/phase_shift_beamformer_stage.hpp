// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Dense phase-shift (conventional / Bartlett) beamforming stage.
//
// For batch b, beam grid point (ia, ie), and range-Doppler cell n the cube is
//   Y[b, beam, n] = Σ_c conj(w[beam, c]) · X[b, c, n]
// i.e. a batched complex GEMM Y = Wᶜ · X per batch slice, where
//   Wᶜ = conj(w)            [num_beams,    num_channels]  (built on device,
//                                                          beam az-major)
//   X  = rd_maps            [batch, num_channels, num_cells]
//   Y  = rda_cube           [batch, num_beams,    num_cells]
// The weights are identical across batch, so Wᶜ is broadcast (clone with a
// zero batch stride) into the batched matmul instead of being materialized per
// slice. Casting the beamformer as one GEMM lets cuBLAS (via MatX) stream the
// steering table once and reuse it across every cell.
//
// num_beams = num_az * num_el (az-major: beam = ia * num_el + ie) and
// num_cells = num_doppler * num_range_bins, so the contiguous Y buffer aliases
// the (..., num_az, num_el, num_doppler, num_range_bins) output cube directly —
// no post-permute.
//
// Includes <matx.h> (via matx_utils.hpp): include only from .cu files.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// Enqueue the beamforming GEMM for one tick on `stream`; no synchronization.
//
//   rd_maps    — device ptr, [batch, num_channels, num_cells]
//                cuda::std::complex<T>.
//   weights_c  — device ptr, [num_beams, num_channels] cuda::std::complex<T>,
//                the conjugated steering table, beam az-major (invalid beams
//                are zero, so their cube power is zero without a mask).
//   rda_cube   — device ptr, [batch, num_beams, num_cells]
//                cuda::std::complex<T> output buffer (aliases the
//                (num_az, num_el, D, R) cube).
template <typename T>
inline void run_phase_shift_beamform_stage(
    const cuda::std::complex<T>* rd_maps, cuda::std::complex<T>* rda_cube,
    int64_t batch, int64_t num_channels, int64_t num_cells,
    const cuda::std::complex<T>* weights_c, int64_t num_beams,
    cudaStream_t stream) {
  if (batch <= 0 || num_cells <= 0 || num_beams <= 0 || num_channels <= 0) {
    return;
  }

  auto w_view = matx::make_tensor<cuda::std::complex<T>, 2>(
      const_cast<cuda::std::complex<T>*>(weights_c),
      {num_beams, num_channels});
  auto x_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      const_cast<cuda::std::complex<T>*>(rd_maps),
      {batch, num_channels, num_cells});
  auto y_view = matx::make_tensor<cuda::std::complex<T>, 3>(
      rda_cube, {batch, num_beams, num_cells});

  // Broadcast the (constant-across-batch) steering table into the batched
  // GEMM: clone adds a leading batch dim with stride 0 (no copy).
  auto w_bcast = matx::clone<3>(
      w_view, {batch, matx::matxKeepDim, matx::matxKeepDim});

  matx::cudaExecutor exec{stream};
  (y_view = matx::matmul(w_bcast, x_view)).run(exec);
}

}  // namespace holoscan::radar::detail
