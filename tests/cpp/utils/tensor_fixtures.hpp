// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host <-> device holoscan::Tensor helpers for the operator tests.

#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::testing {

// Upload host samples as a device holoscan::Tensor with the given
// full-rank shape and the DLPack dtype matching ElementT.
template <typename ElementT>
std::shared_ptr<holoscan::Tensor> to_device_tensor(
    const std::vector<ElementT>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  if (static_cast<size_t>(count) != host.size()) {
    throw std::invalid_argument("to_device_tensor: shape/size mismatch");
  }
  auto tensor = holoscan::radar::make_device_tensor(shape, dtype);
  if (cudaMemcpy(tensor->data(), host.data(), sizeof(ElementT) * host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("to_device_tensor: cudaMemcpy H2D failed");
  }
  return tensor;
}

// Upload a bool mask as kDLBool on device (uint8_t host values 0/1).
inline std::shared_ptr<holoscan::Tensor> to_device_bool_tensor(
    const std::vector<uint8_t>& host, const std::vector<int64_t>& shape) {
  return to_device_tensor(host, shape, DLDataType{kDLBool, 8, 1});
}

// Upload a uint8 mask on device (graph-transport workaround for kDLBool on
// HSDK 4.3 when wiring is_valid into downstream operators).
inline std::shared_ptr<holoscan::Tensor> to_device_uint8_tensor(
    const std::vector<uint8_t>& host, const std::vector<int64_t>& shape) {
  return to_device_tensor(host, shape, DLDataType{kDLUInt, 8, 1});
}

// Convert the simulator's complex<double> cube to the instantiation
// precision and upload it as a complex device tensor.
template <typename T>
std::shared_ptr<holoscan::Tensor> complex_cube_to_device(
    const std::vector<std::complex<double>>& cube,
    const std::vector<int64_t>& shape) {
  std::vector<std::complex<T>> cast(cube.size());
  for (size_t i = 0; i < cube.size(); ++i) {
    cast[i] = std::complex<T>(static_cast<T>(cube[i].real()),
                              static_cast<T>(cube[i].imag()));
  }
  return to_device_tensor(cast, shape,
                          holoscan::radar::complex_dtype<T>());
}

// Real-IF variant.
template <typename T>
std::shared_ptr<holoscan::Tensor> real_cube_to_device(
    const std::vector<double>& cube, const std::vector<int64_t>& shape) {
  std::vector<T> cast(cube.size());
  for (size_t i = 0; i < cube.size(); ++i) {
    cast[i] = static_cast<T>(cube[i]);
  }
  return to_device_tensor(cast, shape, holoscan::radar::real_dtype<T>());
}

// A host-side copy of a (device or host) tensor, for assertions.
struct HostTensor {
  std::vector<int64_t> shape;
  DLDataType dtype{};
  std::vector<std::byte> data;

  int64_t size() const {
    int64_t count = 1;
    for (int64_t d : shape) {
      count *= d;
    }
    return count;
  }

  template <typename ElementT>
  const ElementT* as() const {
    return reinterpret_cast<const ElementT*>(data.data());
  }
};

inline HostTensor to_host(const holoscan::Tensor& tensor) {
  HostTensor host;
  host.shape = tensor.shape();
  host.dtype = tensor.dtype();
  host.data.resize(static_cast<size_t>(tensor.nbytes()));
  // Surface an earlier async kernel failure here, with its own error,
  // rather than as a confusing failure of the copy below.
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    throw std::runtime_error(
        std::string("to_host: cudaDeviceSynchronize failed: ") +
        cudaGetErrorString(sync));
  }
  const cudaMemcpyKind kind =
      holoscan::radar::is_cuda_accessible(tensor) ? cudaMemcpyDeviceToHost
                                                  : cudaMemcpyHostToHost;
  if (cudaMemcpy(host.data.data(), tensor.data(),
                 static_cast<size_t>(tensor.nbytes()), kind) != cudaSuccess) {
    throw std::runtime_error("to_host: cudaMemcpy failed");
  }
  return host;
}

namespace detail {

struct PageableHostBufferHolder {
  std::vector<int64_t> shape;
  std::vector<std::byte> data;
  DLManagedTensor dl_managed{};

  PageableHostBufferHolder(std::vector<int64_t> shape_in,
                           std::vector<std::byte> bytes, DLDataType dtype) {
    shape = std::move(shape_in);
    data = std::move(bytes);
    dl_managed.dl_tensor.data = data.data();
    dl_managed.dl_tensor.device = DLDevice{kDLCPU, 0};
    dl_managed.dl_tensor.ndim = static_cast<int32_t>(shape.size());
    dl_managed.dl_tensor.dtype = dtype;
    dl_managed.dl_tensor.shape = shape.data();
    dl_managed.dl_tensor.strides = nullptr;
    dl_managed.dl_tensor.byte_offset = 0;
    dl_managed.manager_ctx = this;
    dl_managed.deleter = [](DLManagedTensor* self) {
      delete static_cast<PageableHostBufferHolder*>(self->manager_ctx);
    };
  }
};

}  // namespace detail

// Wrap pageable (non-pinned) host memory as a holoscan::Tensor. Operators
// that call require_cuda_accessible must reject this input.
inline std::shared_ptr<holoscan::Tensor> to_pageable_host_tensor(
    const std::vector<int64_t>& shape, DLDataType dtype,
    std::vector<std::byte> bytes) {
  auto holder = std::make_unique<detail::PageableHostBufferHolder>(
      shape, std::move(bytes), dtype);
  auto tensor =
      std::make_shared<holoscan::Tensor>(&holder->dl_managed);
  holder.release();
  return tensor;
}

template <typename ElementT>
std::shared_ptr<holoscan::Tensor> to_pageable_host_tensor(
    const std::vector<ElementT>& host, const std::vector<int64_t>& shape,
    DLDataType dtype) {
  std::vector<std::byte> bytes(host.size() * sizeof(ElementT));
  if (!host.empty()) {
    std::memcpy(bytes.data(), host.data(), bytes.size());
  }
  return to_pageable_host_tensor(shape, dtype, std::move(bytes));
}

}  // namespace holoscan::radar::testing
