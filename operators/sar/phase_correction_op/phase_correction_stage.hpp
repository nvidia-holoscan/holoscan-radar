// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Broadcast per-pulse phase onto range profiles: out[b,p,r] =
// in[b,p,r] * exp(sign * i * φ[b,p]), sign = −1 for subtract and +1
// for add. CUDA-TU only.

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/cuda_check.hpp"

namespace holoscan::radar::detail {

template <typename T>
__device__ void sincos_t(T x, T* s, T* c) {
  if constexpr (std::is_same_v<T, float>) {
    sincosf(x, s, c);
  } else {
    sincos(x, s, c);
  }
}

template <typename T>
__global__ void phase_correct_kernel(
    cuda::std::complex<T>* out, const cuda::std::complex<T>* in,
    const T* phase, int64_t n, int64_t pulses, int64_t bins,
    T sign) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
       i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t tmp = i / bins;
    const int64_t pulse = tmp % pulses;
    const int64_t batch = tmp / pulses;
    T s = 0;
    T c = 0;
    sincos_t(sign * phase[batch * pulses + pulse], &s, &c);
    const auto z = in[i];
    out[i] = cuda::std::complex<T>(z.real() * c - z.imag() * s,
                                   z.real() * s + z.imag() * c);
  }
}

// in  — (batch, pulses, bins) complex<T>, device.
// phase — (batch, pulses) T radians, device.
// out — same layout as in, preallocated.
// sense kSubtract → exp(−iφ); kAdd → exp(+iφ).
template <typename T>
void run_phase_correction(PhaseCorrectionSense sense, const void* in,
                          const void* phase, int64_t batch, int64_t pulses,
                          int64_t bins, void* out, cudaStream_t stream,
                          const std::string& who) {
  const int64_t n = batch * pulses * bins;
  if (n <= 0 || pulses <= 0 || bins <= 0) {
    return;
  }
  const T sign = (sense == PhaseCorrectionSense::kSubtract)
                     ? static_cast<T>(-1)
                     : static_cast<T>(1);
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      (n + kThreads - 1) / kThreads);
  phase_correct_kernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<cuda::std::complex<T>*>(out),
      static_cast<const cuda::std::complex<T>*>(in),
      static_cast<const T*>(phase), n, pulses, bins, sign);
  check_cuda("phase_correct kernel", who);
}

}  // namespace holoscan::radar::detail
