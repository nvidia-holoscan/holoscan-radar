// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "sar_image_lut_kernels.hpp"  // NOLINT(build/include_subdir)

namespace {

constexpr int kThreads = 256;

struct AutoFree {
  void* p{nullptr};
  ~AutoFree() { cudaFree(p); }
};

struct PositiveFinite {
  __device__ __forceinline__ bool operator()(float x) const {
    return isfinite(x) && x > 0.0F;
  }
};

__global__ void magnitude_c64_kernel(const float2* in, float* out,
                                     size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= count) {
    return;
  }
  const float2 z = in[i];
  out[i] = hypotf(z.x, z.y);
}

__global__ void magnitude_c128_kernel(const double2* in, float* out,
                                      size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= count) {
    return;
  }
  const double2 z = in[i];
  out[i] = static_cast<float>(hypot(z.x, z.y));
}

__global__ void box_average_kernel(const float* in, float* out,
                                   int64_t in_u, int64_t factor,
                                   int64_t out_v, int64_t out_u) {
  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y +
                    threadIdx.y;
  if (x >= out_u || y >= out_v) {
    return;
  }
  float sum = 0.0F;
  for (int64_t dy = 0; dy < factor; ++dy) {
    const int64_t iy = y * factor + dy;
    for (int64_t dx = 0; dx < factor; ++dx) {
      const int64_t ix = x * factor + dx;
      sum += in[iy * in_u + ix];
    }
  }
  out[y * out_u + x] =
      sum / static_cast<float>(factor * factor);
}

__global__ void apply_lut_kernel(const float* mag, std::uint8_t* out,
                                 size_t count, float high, float floor) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= count) {
    return;
  }
  const float db =
      20.0F * log10f(mag[i] / high + 1.0e-12F);
  const float clipped = fminf(0.0F, fmaxf(floor, db));
  const float scaled =
      (clipped - floor) / (-floor) * 255.0F;
  out[i] = static_cast<std::uint8_t>(
      fminf(255.0F, fmaxf(0.0F, roundf(scaled))));
}

int blocks_for(size_t count) {
  return static_cast<int>(
      (count + static_cast<size_t>(kThreads) - 1) /
      static_cast<size_t>(kThreads));
}

int64_t percentile_index(int64_t n, double q) {
  auto idx = static_cast<int64_t>(
      std::floor(static_cast<double>(n - 1) * (q / 100.0)));
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= n) {
    idx = n - 1;
  }
  return idx;
}

}  // namespace

void sar_image_magnitude_c64(const void* in, float* out, size_t count,
                             cudaStream_t stream) {
  magnitude_c64_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const float2*>(in), out, count);
}

void sar_image_magnitude_c128(const void* in, float* out, size_t count,
                              cudaStream_t stream) {
  magnitude_c128_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const double2*>(in), out, count);
}

void sar_image_box_average(const float* in, float* out, int64_t in_v,
                           int64_t in_u, int64_t factor,
                           cudaStream_t stream) {
  const int64_t out_v = in_v / factor;
  const int64_t out_u = in_u / factor;
  constexpr int kBx = 16;
  constexpr int kBy = 16;
  const dim3 block(kBx, kBy);
  const dim3 grid(
      static_cast<unsigned int>((out_u + kBx - 1) / kBx),
      static_cast<unsigned int>((out_v + kBy - 1) / kBy));
  box_average_kernel<<<grid, block, 0, stream>>>(in, out, in_u,
                                                 factor, out_v, out_u);
}

cudaError_t sar_image_percentiles(const float* in, size_t count,
                                  double high_q, double low_q,
                                  float* high_out, float* low_out,
                                  size_t* n_valid,
                                  cudaStream_t stream) {
  *n_valid = 0;
  if (count == 0) {
    return cudaSuccess;
  }
  const auto n = static_cast<int64_t>(count);
  AutoFree compact;
  AutoFree sorted;
  AutoFree nsel;
  AutoFree cub_select;
  AutoFree cub_sort;
  void* mem = nullptr;
  cudaError_t err = cudaMalloc(&mem, count * sizeof(float));
  if (err != cudaSuccess) {
    return err;
  }
  compact.p = mem;
  mem = nullptr;
  err = cudaMalloc(&mem, count * sizeof(float));
  if (err != cudaSuccess) {
    return err;
  }
  sorted.p = mem;
  mem = nullptr;
  err = cudaMalloc(&mem, sizeof(int64_t));
  if (err != cudaSuccess) {
    return err;
  }
  nsel.p = mem;

  PositiveFinite pred;
  auto* d_compact = static_cast<float*>(compact.p);
  auto* d_sorted = static_cast<float*>(sorted.p);
  auto* d_nsel = static_cast<int64_t*>(nsel.p);

  size_t temp_bytes = 0;
  err = cub::DeviceSelect::If(nullptr, temp_bytes, in, d_compact,
                              d_nsel, n, pred, stream);
  if (err != cudaSuccess) {
    return err;
  }
  if (temp_bytes > 0) {
    mem = nullptr;
    err = cudaMalloc(&mem, temp_bytes);
    if (err != cudaSuccess) {
      return err;
    }
    cub_select.p = mem;
  }
  err = cub::DeviceSelect::If(cub_select.p, temp_bytes, in,
                              d_compact, d_nsel, n, pred, stream);
  if (err != cudaSuccess) {
    return err;
  }

  int64_t h_n = 0;
  err = cudaMemcpyAsync(&h_n, d_nsel, sizeof(h_n),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return err;
  }
  if (h_n <= 0) {
    return cudaSuccess;
  }
  *n_valid = static_cast<size_t>(h_n);

  temp_bytes = 0;
  err = cub::DeviceRadixSort::SortKeys(
      nullptr, temp_bytes, d_compact, d_sorted, h_n, 0,
      static_cast<int>(sizeof(float) * 8), stream);
  if (err != cudaSuccess) {
    return err;
  }
  if (temp_bytes > 0) {
    mem = nullptr;
    err = cudaMalloc(&mem, temp_bytes);
    if (err != cudaSuccess) {
      return err;
    }
    cub_sort.p = mem;
  }
  err = cub::DeviceRadixSort::SortKeys(
      cub_sort.p, temp_bytes, d_compact, d_sorted, h_n, 0,
      static_cast<int>(sizeof(float) * 8), stream);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaMemcpyAsync(high_out,
                        d_sorted + percentile_index(h_n, high_q),
                        sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) {
    return err;
  }
  if (low_q > 0.0 && low_out != nullptr) {
    err = cudaMemcpyAsync(low_out,
                          d_sorted + percentile_index(h_n, low_q),
                          sizeof(float), cudaMemcpyDeviceToHost,
                          stream);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaStreamSynchronize(stream);
}

void sar_image_apply_lut(const float* mag, std::uint8_t* out,
                         size_t count, float high, float floor,
                         cudaStream_t stream) {
  apply_lut_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      mag, out, count, high, floor);
}
