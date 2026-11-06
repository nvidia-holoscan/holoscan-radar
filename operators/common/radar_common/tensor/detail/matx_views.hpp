// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Private: not installed; include only from CUDA translation units (.cu) and
// *_stage.hpp headers. MatX-side views and helpers shared by the operator
// kernels; this header includes <matx.h>, which installed headers must not.

#pragma once

#include <matx.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Map the locked fft_norm parameter values onto MatX's normalization modes.
inline matx::FFTNorm to_matx_norm(FftNorm norm) {
  switch (norm) {
    case FftNorm::kBackward:
      return matx::FFTNorm::BACKWARD;
    case FftNorm::kOrtho:
      return matx::FFTNorm::ORTHO;
    case FftNorm::kForward:
      return matx::FFTNorm::FORWARD;
  }
  return matx::FFTNorm::BACKWARD;
}

// Zero-copy 3-D complex view over a collapsed (batch, rows, cols) buffer.
template <typename T>
auto make_complex_view3(void* data, int64_t batch, int64_t rows, int64_t cols) {
  return matx::make_tensor<cuda::std::complex<T>, 3>(static_cast<cuda::std::complex<T>*>(data),
                                                     {batch, rows, cols});
}

template <typename T>
auto make_const_complex_view3(const void* data, int64_t batch, int64_t rows, int64_t cols) {
  // MatX views are not const-qualified; inputs are treated as read-only by
  // convention (kernels never write through input views).
  return make_complex_view3<T>(const_cast<void*>(data), batch, rows, cols);
}

// Zero-copy 3-D real view (real-IF front end path).
template <typename T>
auto make_real_view3(const void* data, int64_t batch, int64_t rows, int64_t cols) {
  return matx::make_tensor<T, 3>(static_cast<T*>(const_cast<void*>(data)), {batch, rows, cols});
}

// Upload host-generated double-precision coefficients (e.g. window taps)
// into a device MatX tensor of the instantiation precision. Synchronous on
// `stream` completion is NOT required by callers: the copy is stream-ordered
// and the returned tensor participates in stream-ordered use.
template <typename T>
matx::tensor_t<T, 1> upload_coefficients(const std::vector<double>& host, cudaStream_t stream) {
  const int64_t n = static_cast<int64_t>(host.size());
  std::vector<T> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = static_cast<T>(host[i]);
  }
  auto device = matx::make_tensor<T>({n}, matx::MATX_DEVICE_MEMORY);
  if (cudaMemcpyAsync(device.Data(), cast.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("upload_coefficients: H2D copy failed");
  }
  // The host staging vector is freed on return: block until the copy lands.
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("upload_coefficients: stream sync failed");
  }
  return device;
}

// Upload a host complex<double> vector as complex<T> (dechirp references).
template <typename T>
matx::tensor_t<cuda::std::complex<T>, 1> upload_complex_coefficients(
    const std::vector<std::complex<double>>& host, cudaStream_t stream) {
  const int64_t n = static_cast<int64_t>(host.size());
  std::vector<cuda::std::complex<T>> cast(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    cast[i] = cuda::std::complex<T>(static_cast<T>(host[i].real()), static_cast<T>(host[i].imag()));
  }
  auto device = matx::make_tensor<cuda::std::complex<T>>({n}, matx::MATX_DEVICE_MEMORY);
  if (cudaMemcpyAsync(device.Data(), cast.data(), sizeof(cuda::std::complex<T>) * host.size(),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("upload_complex_coefficients: H2D copy failed");
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("upload_complex_coefficients: stream sync failed");
  }
  return device;
}

}  // namespace holoscan::radar
