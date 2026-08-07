// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_stage.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::radar::detail {

namespace {

constexpr int64_t kBlock = 256;

// The de-interleave is a pure per-chirp-row permutation. The innermost
// `samples` axis is copied 1:1 and stays contiguous; the gather map only
// selects which interleaved slow-time row feeds each (tx, l) output row.
// Launching one CUDA block per output row avoids per-sample div/mod work and
// lets the block cooperatively copy the stride-1 fast-time vector.
template <typename Word>
__global__ void deinterleave_row_copy_kernel(
    const Word* input, Word* output, int64_t num_rx, int64_t slow_time,
    int64_t samples, int64_t num_tx, int64_t chirps_per_tx,
    const int64_t* gather_map) {
  const int64_t in_plane = slow_time * samples;
  const int64_t out_plane = chirps_per_tx * samples;
  const int64_t in_rx_stride = in_plane;
  const int64_t out_rx_stride = out_plane;
  const int64_t in_batch_stride = num_rx * in_rx_stride;
  const int64_t out_batch_stride = num_tx * num_rx * out_rx_stride;

  int64_t rem = static_cast<int64_t>(blockIdx.x);
  const int64_t l = rem % chirps_per_tx;
  rem /= chirps_per_tx;
  const int64_t rx = rem % num_rx;
  rem /= num_rx;
  const int64_t tx = rem % num_tx;
  rem /= num_tx;
  const int64_t b = rem;

  const int64_t in_slow = gather_map[tx * chirps_per_tx + l];
  const int64_t in_base =
      b * in_batch_stride + rx * in_rx_stride + in_slow * samples;
  const int64_t out_base = b * out_batch_stride +
                           tx * num_rx * out_rx_stride + rx * out_rx_stride +
                           l * samples;
  for (int64_t s = static_cast<int64_t>(threadIdx.x); s < samples;
       s += blockDim.x) {
    output[out_base + s] = input[in_base + s];
  }
}

template <typename Word>
void launch_typed(const void* input, void* output, const AdcCubeLayout& layout,
                  int64_t num_tx, int64_t chirps_per_tx,
                  const int64_t* d_gather_map, cudaStream_t stream) {
  const int64_t rows = layout.batch * num_tx * layout.num_rx * chirps_per_tx;
  const int grid = static_cast<int>(rows);
  deinterleave_row_copy_kernel<Word><<<grid, kBlock, 0, stream>>>(
      static_cast<const Word*>(input), static_cast<Word*>(output),
      layout.num_rx, layout.slow_time, layout.samples, num_tx, chirps_per_tx,
      d_gather_map);
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
  // One thread copies one sample element as a single word. The four supported
  // dtypes (complex64/128, float32/64) map onto 4-, 8-, and 16-byte words;
  // device tensors from make_device_tensor are cudaMalloc-aligned and every
  // offset here is an element multiple, so the wide loads stay aligned.
  switch (elem_bytes) {
    case 4:
      launch_typed<uint32_t>(input, output, layout, num_tx, chirps_per_tx,
                             d_gather_map, stream);
      break;
    case 8:
      launch_typed<uint64_t>(input, output, layout, num_tx, chirps_per_tx,
                             d_gather_map, stream);
      break;
    case 16:
      launch_typed<uint4>(input, output, layout, num_tx, chirps_per_tx,
                          d_gather_map, stream);
      break;
    default:
      throw std::invalid_argument(
          "deinterleave stage: unsupported element size " +
          std::to_string(elem_bytes) + " bytes");
  }
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("deinterleave stage: kernel launch failed");
  }
}

}  // namespace holoscan::radar::detail
