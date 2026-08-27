// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::radar {

// Decode stored CF8 (big-endian complex64) on the device. d_in is
// nsamp * 8 bytes of file bytes; d_out is nsamp complex samples.
// For float, d_in may alias d_out (in-place after the H2D). d_amp is
// per-pulse AmpSF (length nsamp / num_samples) or nullptr. Empty
// nsamp is a no-op (cudaSuccess).
cudaError_t cphd_decode_cf8_f32(const void* d_in, void* d_out, size_t nsamp,
                                int64_t num_samples, const double* d_amp,
                                cudaStream_t stream);
cudaError_t cphd_decode_cf8_f64(const void* d_in, void* d_out, size_t nsamp,
                                int64_t num_samples, const double* d_amp,
                                cudaStream_t stream);

}  // namespace holoscan::radar
