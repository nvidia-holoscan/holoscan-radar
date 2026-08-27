// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_cf8_decode.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::radar {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t bswap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24) | ((value & 0x0000ff00U) << 8) |
         ((value & 0x00ff0000U) >> 8) | ((value & 0xff000000U) >> 24);
}

int blocks_for(size_t n) {
  if (n == 0) {
    return 1;
  }
  return static_cast<int>((n + static_cast<size_t>(kThreads) - 1) /
                          static_cast<size_t>(kThreads));
}

__global__ void decode_cf8_f32_kernel(const uint32_t* in, float2* out,
                                      size_t nsamp, int64_t num_samples,
                                      const double* amp) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= nsamp) {
    return;
  }
  float re = __uint_as_float(bswap32(in[2 * i]));
  float im = __uint_as_float(bswap32(in[2 * i + 1]));
  if (amp != nullptr) {
    const float scale = static_cast<float>(
        amp[i / static_cast<size_t>(num_samples)]);
    re *= scale;
    im *= scale;
  }
  out[i] = make_float2(re, im);
}

__global__ void decode_cf8_f64_kernel(const uint32_t* in, double2* out,
                                      size_t nsamp, int64_t num_samples,
                                      const double* amp) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= nsamp) {
    return;
  }
  const double re = static_cast<double>(
      __uint_as_float(bswap32(in[2 * i])));
  const double im = static_cast<double>(
      __uint_as_float(bswap32(in[2 * i + 1])));
  if (amp != nullptr) {
    const double scale = amp[i / static_cast<size_t>(num_samples)];
    out[i] = make_double2(re * scale, im * scale);
    return;
  }
  out[i] = make_double2(re, im);
}

}  // namespace

cudaError_t cphd_decode_cf8_f32(const void* d_in, void* d_out, size_t nsamp,
                                int64_t num_samples, const double* d_amp,
                                cudaStream_t stream) {
  if (nsamp == 0) {
    return cudaSuccess;
  }
  decode_cf8_f32_kernel<<<blocks_for(nsamp), kThreads, 0, stream>>>(
      static_cast<const uint32_t*>(d_in), static_cast<float2*>(d_out), nsamp,
      num_samples, d_amp);
  return cudaGetLastError();
}

cudaError_t cphd_decode_cf8_f64(const void* d_in, void* d_out, size_t nsamp,
                                int64_t num_samples, const double* d_amp,
                                cudaStream_t stream) {
  if (nsamp == 0) {
    return cudaSuccess;
  }
  decode_cf8_f64_kernel<<<blocks_for(nsamp), kThreads, 0, stream>>>(
      static_cast<const uint32_t*>(d_in), static_cast<double2*>(d_out),
      nsamp, num_samples, d_amp);
  return cudaGetLastError();
}

}  // namespace holoscan::radar
