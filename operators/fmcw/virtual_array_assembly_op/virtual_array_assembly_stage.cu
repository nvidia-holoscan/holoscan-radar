// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;
constexpr int64_t kCellsPerAverageTile = 4096;

template <typename T>
using Complex = cuda::std::complex<T>;

template <typename T>
__global__ void gather_single_source_kernel(const Complex<T>* input, Complex<T>* output,
                                            int64_t num_doppler, int64_t num_range,
                                            const int64_t* src_channel) {
  const int64_t plane = num_doppler * num_range;
  const int64_t v = static_cast<int64_t>(blockIdx.x);
  const int64_t src = src_channel[v];
  const int64_t in_base = src * plane;
  const int64_t out_base = v * plane;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < plane; i += blockDim.x) {
    output[out_base + i] = input[in_base + i];
  }
}

template <typename T, bool Tiled>
__global__ void gather_average_kernel(const Complex<T>* input, Complex<T>* output,
                                      int64_t num_doppler, int64_t num_range,
                                      const int64_t* member_offsets,
                                      const int64_t* member_indices) {
  const int64_t plane = num_doppler * num_range;
  const int64_t v = static_cast<int64_t>(blockIdx.x);
  const int64_t begin = member_offsets[v];
  const int64_t end = member_offsets[v + 1];
  const int64_t count = end - begin;
  if (count <= 0) {
    return;
  }

  const int64_t out_base = v * plane;
  const int64_t first = static_cast<int64_t>(threadIdx.x) +
                        (Tiled ? static_cast<int64_t>(blockIdx.y) * blockDim.x : 0);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * (Tiled ? gridDim.y : 1);
  for (int64_t i = first; i < plane; i += stride) {
    Complex<T> sum{};
    for (int64_t m = begin; m < end; ++m) {
      const int64_t src = member_indices[m];
      sum += input[src * plane + i];
    }
    output[out_base + i] = sum / static_cast<T>(count);
  }
}

}  // namespace

RdMapsLayout parse_rd_maps_layout(const holoscan::Tensor& tensor, const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument(std::string(port_name) +
                                ": expected rank 4 (M_tx, N_rx, num_doppler, num_range), got " +
                                std::to_string(shape.size()));
  }
  RdMapsLayout layout;
  layout.num_tx = shape[0];
  layout.num_rx = shape[1];
  layout.num_doppler = shape[2];
  layout.num_range = shape[3];
  if (layout.num_tx <= 0 || layout.num_rx <= 0 || layout.num_doppler <= 0 ||
      layout.num_range <= 0) {
    throw std::invalid_argument(std::string(port_name) + ": trailing axes must be positive");
  }
  return layout;
}

std::vector<int64_t> assembled_rd_shape(const RdMapsLayout& in, int64_t num_virtual) {
  return {num_virtual, in.num_doppler, in.num_range};
}

template <typename T>
void run_virtual_assembly_stage(const void* input, void* output, const RdMapsLayout& layout,
                                const VirtualAssemblyPlan& plan, const int64_t* d_src_channel,
                                const int64_t* d_member_offsets, const int64_t* d_member_indices,
                                cudaStream_t stream) {
  const int grid = static_cast<int>(plan.num_virtual);
  const auto* in = static_cast<const Complex<T>*>(input);
  auto* out = static_cast<Complex<T>*>(output);

  if (plan.policy == OverlapPolicy::kAverage) {
    const int64_t plane = layout.num_doppler * layout.num_range;
    if (plane > kCellsPerAverageTile) {
      // Tiles cover disjoint stripes without changing a cell's member order.
      // The stride still covers larger planes when grid.y reaches its limit.
      const auto tiles = std::min<int64_t>(65535, 1 + (plane - 1) / kCellsPerAverageTile);
      const dim3 tiled_grid(static_cast<unsigned int>(grid), static_cast<unsigned int>(tiles));
      gather_average_kernel<T, true><<<tiled_grid, kBlock, 0, stream>>>(
          in, out, layout.num_doppler, layout.num_range, d_member_offsets, d_member_indices);
    } else {
      gather_average_kernel<T, false><<<grid, kBlock, 0, stream>>>(
          in, out, layout.num_doppler, layout.num_range, d_member_offsets, d_member_indices);
    }
  } else {
    gather_single_source_kernel<T>
        <<<grid, kBlock, 0, stream>>>(in, out, layout.num_doppler, layout.num_range, d_src_channel);
  }
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("virtual assembly stage: kernel launch failed");
  }
}

template void run_virtual_assembly_stage<float>(
    const void* input, void* output, const RdMapsLayout& layout, const VirtualAssemblyPlan& plan,
    const int64_t* d_src_channel, const int64_t* d_member_offsets, const int64_t* d_member_indices,
    cudaStream_t stream);
template void run_virtual_assembly_stage<double>(
    const void* input, void* output, const RdMapsLayout& layout, const VirtualAssemblyPlan& plan,
    const int64_t* d_src_channel, const int64_t* d_member_offsets, const int64_t* d_member_indices,
    cudaStream_t stream);

}  // namespace holoscan::radar::detail
