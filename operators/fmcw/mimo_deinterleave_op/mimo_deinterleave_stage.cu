// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_stage.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;

__global__ void deinterleave_kernel(const char* input, char* output,
                                    int64_t batch, int64_t num_rx,
                                    int64_t slow_time, int64_t samples,
                                    int64_t num_tx, int64_t chirps_per_tx,
                                    const int64_t* gather_map,
                                    size_t elem_bytes) {
  (void)slow_time;
  const int64_t in_plane = slow_time * samples;
  const int64_t out_plane = chirps_per_tx * samples;
  const int64_t in_rx_stride = in_plane;
  const int64_t out_rx_stride = out_plane;
  const int64_t in_batch_stride = num_rx * in_rx_stride;
  const int64_t out_batch_stride = num_tx * num_rx * out_rx_stride;

  const int64_t total =
      batch * num_tx * num_rx * chirps_per_tx * samples;
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  int64_t rem = tid;
  const int64_t s = rem % samples;
  rem /= samples;
  const int64_t l = rem % chirps_per_tx;
  rem /= chirps_per_tx;
  const int64_t rx = rem % num_rx;
  rem /= num_rx;
  const int64_t tx = rem % num_tx;
  rem /= num_tx;
  const int64_t b = rem;

  const int64_t in_slow = gather_map[tx * chirps_per_tx + l];
  const size_t in_off = static_cast<size_t>(
      (b * in_batch_stride + rx * in_rx_stride + in_slow * samples + s) *
      elem_bytes);
  const size_t out_off = static_cast<size_t>(
      (b * out_batch_stride + tx * num_rx * out_rx_stride + rx * out_rx_stride +
       l * samples + s) *
      elem_bytes);
  const char* src = input + in_off;
  char* dst = output + out_off;
  for (size_t i = 0; i < elem_bytes; ++i) {
    dst[i] = src[i];
  }
}

}  // namespace

AdcCubeLayout parse_adc_cube_layout(const holoscan::Tensor& tensor,
                                    const char* port_name) {
  const auto& shape = tensor.shape();
  if (shape.size() < 3) {
    throw std::invalid_argument(std::string(port_name) +
                                ": expected rank >= 3 "
                                "(..., N_rx, slow_time, samples)");
  }
  AdcCubeLayout layout;
  layout.samples = shape.back();
  layout.slow_time = shape[shape.size() - 2];
  layout.num_rx = shape[shape.size() - 3];
  if (layout.num_rx <= 0 || layout.slow_time <= 0 || layout.samples <= 0) {
    throw std::invalid_argument(std::string(port_name) +
                                ": trailing axes must be positive");
  }
  layout.leading.assign(shape.begin(), shape.end() - 3);
  layout.batch = 1;
  for (int64_t d : layout.leading) {
    if (d <= 0) {
      throw std::invalid_argument(std::string(port_name) +
                                  ": leading batch axes must be positive");
    }
    layout.batch *= d;
  }
  return layout;
}

std::vector<int64_t> deinterleaved_adc_shape(const AdcCubeLayout& in,
                                             int64_t num_tx,
                                             int64_t chirps_per_tx) {
  std::vector<int64_t> shape = in.leading;
  shape.push_back(num_tx);
  shape.push_back(in.num_rx);
  shape.push_back(chirps_per_tx);
  shape.push_back(in.samples);
  return shape;
}

void run_deinterleave_stage(const void* input, void* output,
                            const AdcCubeLayout& layout, int64_t num_tx,
                            int64_t chirps_per_tx, const int64_t* d_gather_map,
                            size_t elem_bytes, cudaStream_t stream) {
  const int64_t total = layout.batch * num_tx * layout.num_rx * chirps_per_tx *
                        layout.samples;
  const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
  deinterleave_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const char*>(input), static_cast<char*>(output), layout.batch,
      layout.num_rx, layout.slow_time, layout.samples, num_tx, chirps_per_tx,
      d_gather_map, elem_bytes);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("deinterleave stage: kernel launch failed");
  }
}

}  // namespace holoscan::radar::detail
