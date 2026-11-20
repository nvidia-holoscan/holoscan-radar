// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "radar_app/sar/mosaic_sample.hpp"

namespace holoscan::radar::app::sar {
namespace {

constexpr int kThreads = 256;

// Looks taken per axis inside each decimation box, so a box contributes at
// most this squared. Speckle falls as 1/sqrt(looks), so 16 looks is already
// most of the available improvement; averaging all 1296 samples of a 36x
// box would read the whole accumulator every image for a fraction of a dB.
constexpr int64_t kMaxLooksPerAxis = 4;

template <typename T>
__global__ void multilook_rect(const T* __restrict__ src, int64_t src_pitch, int64_t v0, int64_t u0,
                               T* __restrict__ dst, int64_t out_v, int64_t out_u, int64_t factor,
                               int64_t looks, int64_t step) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= out_v * out_u) {
    return;
  }
  const int64_t row = index / out_u;
  const int64_t column = index - row * out_u;
  const int64_t base = (v0 + row * factor) * src_pitch + (u0 + column * factor);
  double power = 0.0;
  for (int64_t i = 0; i < looks; ++i) {
    for (int64_t j = 0; j < looks; ++j) {
      const T z = src[base + i * step * src_pitch + j * step];
      power += static_cast<double>(z.x) * z.x + static_cast<double>(z.y) * z.y;
    }
  }
  // Incoherent average: the box's mean intensity, carried as amplitude. A
  // multi-looked sample has no meaningful phase, so the imaginary part is
  // zero rather than a coherent mean that would cancel across the box.
  T out;
  out.x = static_cast<decltype(out.x)>(sqrt(power / static_cast<double>(looks * looks)));
  out.y = static_cast<decltype(out.y)>(0);
  dst[index] = out;
}

}  // namespace

cudaError_t sample_mosaic_rect(const void* src, int64_t src_pitch, int64_t v0, int64_t u0,
                               void* dst, int64_t out_v, int64_t out_u, int64_t factor,
                               uint32_t bits, cudaStream_t stream) {
  const int64_t count = out_v * out_u;
  if (src == nullptr || dst == nullptr || count <= 0 || factor < 1 || v0 < 0 || u0 < 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t looks = factor < kMaxLooksPerAxis ? factor : kMaxLooksPerAxis;
  const int64_t step = factor / looks;
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (bits == 128U) {
    multilook_rect<<<blocks, kThreads, 0, stream>>>(static_cast<const double2*>(src), src_pitch, v0,
                                                    u0, static_cast<double2*>(dst), out_v, out_u,
                                                    factor, looks, step);
  } else if (bits == 64U) {
    multilook_rect<<<blocks, kThreads, 0, stream>>>(static_cast<const float2*>(src), src_pitch, v0,
                                                    u0, static_cast<float2*>(dst), out_v, out_u,
                                                    factor, looks, step);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

}  // namespace holoscan::radar::app::sar
