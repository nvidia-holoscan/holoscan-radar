// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// CUDA magnitude and percentile helpers for the HSDK 5 SarImageVizOp.
// The display windows its downsampled positive-magnitude grid.

void sar_image_magnitude_c64(const void* in, float* out, size_t count, cudaStream_t stream);
void sar_image_magnitude_c128(const void* in, float* out, size_t count, cudaStream_t stream);

// Mean of each factor×factor block. out is (in_v/factor) by
// (in_u/factor). factor must be >= 1. Empty (non-positive)
// samples are omitted so a sparse mosaic does not dilute.
void sar_image_box_average(const float* in, float* out, int64_t in_v, int64_t in_u, int64_t factor,
                           cudaStream_t stream);

// out is (in_u, in_v): row-major transpose of (in_v, in_u).
void sar_image_transpose(const float* in, float* out, int64_t in_v, int64_t in_u,
                         cudaStream_t stream);

// Compact finite positive samples, radix-sort them, return
// nearest-rank percentiles. low_q <= 0 skips *low_out. n_valid 0
// means the grid is empty of usable samples (cudaSuccess).
cudaError_t sar_image_percentiles(const float* in, size_t count, double high_q, double low_q,
                                  float* high_out, float* low_out, size_t* n_valid,
                                  cudaStream_t stream);

// dB relative to high, clipped to [floor, 0], scaled 0–255.
// floor is negative dB.
void sar_image_apply_lut(const float* mag, std::uint8_t* out, size_t count, float high, float floor,
                         cudaStream_t stream);
