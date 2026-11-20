// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace holoscan::radar::app::sar {

// Multi-look decimate the sub-rectangle anchored at (v0, u0) of a row-major
// complex image into `dst`. `src_pitch` is the source row stride in elements,
// `bits` the complex element width (64 or 128), and `factor` the decimation
// on both axes. Each output sample is the incoherent average of several
// source samples spread across its box, written as amplitude with zero
// phase -- subsampling speckle instead would be far noisier than the source,
// and at the decimations a strip needs it looks like static.
cudaError_t sample_mosaic_rect(const void* src, int64_t src_pitch, int64_t v0, int64_t u0,
                               void* dst, int64_t out_v, int64_t out_u, int64_t factor,
                               uint32_t bits, cudaStream_t stream);

}  // namespace holoscan::radar::app::sar
