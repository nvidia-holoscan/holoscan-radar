// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device gather that materializes spatial-smoothing sub-array "looks" from the
// host-planned index tables (radar_common/dsp/spatial_smoothing.hpp).
//
// Maps snapshot (num_items, C, K) → ss_snapshot (num_items, L, M) where each
// column is one sub-array's view of one snapshot; L = subarray_size,
// P = num_subarrays, and M = P·K (forward) or 2·P·K (forward-backward). The
// backward columns are conj + J-reversed (fb_flip). Feeding the plain
// covariance kernel `M` columns yields the forward (resp. forward-backward)
// covariance — no change to the estimator's covariance / eig / inverse / scan.
//
// Column layout: forward block [0, P·K) then (if forward_backward) backward
// block [P·K, 2·P·K); within a block, column = p·K + k. This target owns the
// .cu (radar_common stays CUDA-free); index tables + output buffers are
// caller-owned device memory (the operator caches the index tables per config).

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::radar {

// Launch the gather. `subarray_index` is device (num_subarrays·subarray_size)
// int32; `fb_flip` is device (subarray_size) int32, used only when
// `forward_backward` (may be nullptr otherwise). `mask` / `ss_mask` are the
// optional per-snapshot / per-column uint8 validity (both nullptr → all valid).
// `ss_snapshot` is (num_items, subarray_size, M). All work is enqueued on
// `stream`; no synchronization.
template <typename T>
void run_spatial_smoothing_gather(
    const cuda::std::complex<T>* snapshot, const unsigned char* mask,
    int64_t num_items, int64_t num_channels, int64_t num_snapshots,
    const int32_t* subarray_index, const int32_t* fb_flip,
    int64_t num_subarrays, int64_t subarray_size, bool forward_backward,
    cuda::std::complex<T>* ss_snapshot, unsigned char* ss_mask,
    cudaStream_t stream);

}  // namespace holoscan::radar
