// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device implementation of RdMapToImageOp's range-Doppler -> LUT-image
// mapping. Split out of the header-only viz layer because it needs nvcc: the
// rest of radar_app is INTERFACE (host-compiled by each app's TU), so this is
// the one place under applications/ that owns CUDA kernels.
//
// Why on device at all: the mapping needs a per-range-column median, a global
// median and a global max. Doing that on the host cost a D2H of the plane, a
// full cudaDeviceSynchronize (a device-wide barrier that stalled the whole
// pipeline, not just this branch), and an H2D of the image, every frame.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::radar::app {

// Bytes of CUB scratch the sort in rd_map_to_lut_image needs for a plane of
// `count` floats. Query once per shape; the caller owns the allocation so no
// per-frame cudaMalloc happens on the render path.
size_t rd_lut_image_sort_scratch_bytes(int64_t count);

// (num_doppler, num_range) RD plane -> float32 image in [0, lut_max], laid out
// identically and mapped identically to the former host code:
//   1. magnitude per cell (|z| for complex input; the value itself for a real
//      power surface such as CFARDetectorOp's cfar_power),
//   2. per-range-column median, then db_factor * log10(v / median),
//   3. renormalize between the image's global median (floor) and max (peak)
//      onto [0, lut_max].
// db_factor is 20 for amplitude (complex input) and 10 for power (real input) —
// the same dB surface either way.
//
// `image` is both the output and the working buffer (count = num_doppler *
// num_range floats). The caller also supplies:
//   col_median   — num_range floats
//   sorted       — count floats (sorted copy of the dB image; supplies the
//                  global median and max without a host round trip)
//   sort_scratch — rd_lut_image_sort_scratch_bytes(count) bytes
// All work is enqueued on `stream`; this function never synchronizes.
//
// num_doppler must be <= kRdLutMaxDoppler (the per-column median stages a
// column in shared memory).
void rd_map_to_lut_image(const void* rd_map, bool input_is_complex,
                         int64_t num_doppler, int64_t num_range,
                         float db_factor, float lut_max, float* image,
                         float* col_median, float* sorted, void* sort_scratch,
                         size_t sort_scratch_bytes, cudaStream_t stream);

// Shared-memory bound on the column length the median kernel can stage.
constexpr int64_t kRdLutMaxDoppler = 8192;

// Max-project a (num_planes, num_doppler, num_range) cube down its leading
// axes, emitting the (num_doppler, num_range) plane of per-cell maxima as
// float32 power. `num_planes` is the product of every leading axis, so this
// covers a dense path's (num_az, num_el, D, R) as one flattened reduction.
//
// Complex input is reduced on |z|^2 (power, so the result is directly
// comparable to CFARDetectorOp's real cfar_power surface and both feed
// rd_map_to_lut_image with db_factor = 10). Real input is already power and is
// reduced as-is.
//
// One block per output cell, strided over planes then warp/block-reduced, so
// cost is one pass over the cube with no scratch. Enqueued on `stream`; never
// synchronizes.
void rd_map_max_over_planes(const void* cube, bool input_is_complex,
                            int64_t num_planes, int64_t num_doppler,
                            int64_t num_range, float* out,
                            cudaStream_t stream);

}  // namespace holoscan::radar::app
