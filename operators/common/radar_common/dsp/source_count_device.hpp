// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device (batched, per-item) counterpart of the host EFT reference in
// source_count.hpp — see that header for the algorithm and its fidelity
// note (closed-form profile prediction, Gaussian-approximated threshold,
// not the paper's Monte-Carlo-calibrated one). Lets the subspace angle
// estimators (MUSIC / Root-MUSIC / ESPRIT) resolve `num_sources: auto` per
// item directly from the eigenvalues matx::eig already produced on the GPU,
// with no host round trip and no calibration cache. This target owns the
// .cu (radar_common stays CUDA-free).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::radar {

// One source-count decision per item, from its ascending eigenvalue
// spectrum and its own valid-snapshot count. `eigenvalues` is device
// (num_items, num_channels) T, ascending per item (matx::eig's convention).
// `k_counts` is device (num_items,) int64 - the per-item valid-snapshot
// count K feeding the covariance (e.g. the covariance kernel's own
// `is_valid`-aware tally); a uniform snapshot count with no `is_valid`
// masking is just a constant array. `source_count` is device (num_items,)
// int32. Enqueued on `stream`; no synchronization.
//
// Applies the "rank-deficient covariance (K < num_channels) falls back to 1"
// contract clause internally (K taken per item from `k_counts`); otherwise
// clamps to [1, max_sources]. Callers still own the "and warns" half (a
// per-tick device kernel has no clean way to surface a one-shot host log).
template <typename T>
void run_source_count_device(const T* eigenvalues, const int64_t* k_counts,
                             int64_t num_items, int64_t num_channels,
                             double pfa, int64_t max_sources,
                             int32_t* source_count, cudaStream_t stream);

}  // namespace holoscan::radar
