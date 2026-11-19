// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "radar_common/math/checked_sizes.hpp"

namespace holoscan::radar::detail {

template <typename T>
inline int64_t profile_scalar_count(int64_t pulses, int64_t bins) {
  return checked_element_count<T>({pulses, bins, 2}, "BackprojectionOp profile conversion");
}

inline int conversion_blocks(int64_t count) {
  // Bounded grids keep launch geometry representable even for very large buffers.
  return static_cast<int>(std::min<int64_t>(4096, 1 + (count - 1) / 256));
}

template <typename Out, typename In>
__global__ void convert_backprojection_kernel(const In* src, Out* dst, int64_t count) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    dst[i] = static_cast<Out>(src[i]);
  }
}

template <typename Out, typename In>
void convert_backprojection(const In* src, Out* dst, int64_t count, cudaStream_t stream) {
  convert_backprojection_kernel<<<conversion_blocks(count), 256, 0, stream>>>(src, dst, count);
}

}  // namespace holoscan::radar::detail
