// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>

namespace holoscan::radar {

namespace detail {

// Holoscan-owned device allocation kept alive via TensorMemoryReference.
// Prefer DLManagedTensorContext (local-writable) over importing a raw
// DLManagedTensor* — HSDK 5 treats DLPack imports as permanently read-only.
struct CudaDeviceMemory final : holoscan::TensorMemoryReference {
  void* data{nullptr};

  CudaDeviceMemory() = default;
  CudaDeviceMemory(const CudaDeviceMemory&) = delete;
  CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;
  ~CudaDeviceMemory() override {
    if (data != nullptr) {
      cudaFree(data);
      data = nullptr;
    }
  }
};

}  // namespace detail

// Allocate an uninitialized device buffer of the given full-rank shape and
// wrap it as a writable holoscan::Tensor. The Tensor owns the allocation
// (cudaFree on release, which orders after outstanding work on the device).
//
// Compute kernels view the same memory through rank-collapsed MatX views
// (radar_common/tensor/matx_utils.hpp) while the emitted Tensor keeps the
// caller's full-rank shape — this is how rank-agnostic batching round-trips
// leading axes through the operators.
inline std::shared_ptr<holoscan::Tensor> make_device_tensor(const std::vector<int64_t>& shape,
                                                            DLDataType dtype) {
  // Reject negative dims and int64 overflow before the size_t cast: a wrapped
  // count would allocate an undersized buffer that downstream full-extent
  // views write past. Zero-sized dims are allowed — an empty tensor is a valid
  // pipeline state (e.g. zero detections → empty detection_idx / points) and
  // the operators allocate-then-check for it. The overflow division is skipped
  // when dim == 0 (count stays 0, which cannot overflow and would divide by
  // zero here).
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0 || (dim != 0 && count > std::numeric_limits<int64_t>::max() / dim)) {
      throw std::invalid_argument("make_device_tensor: invalid or overflowing shape");
    }
    count *= dim;
  }
  // Guard the byte computation against size_t overflow: even a count that
  // fits in int64 can overflow when scaled by the per-element byte size, and
  // a wrapped result would under-allocate the device buffer.
  const size_t bytes_per_element = ((static_cast<size_t>(dtype.bits) + 7) / 8) * dtype.lanes;
  if (bytes_per_element == 0 ||
      static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / bytes_per_element) {
    throw std::invalid_argument("make_device_tensor: invalid or overflowing shape");
  }
  const size_t bytes = static_cast<size_t>(count) * bytes_per_element;

  auto memory = std::make_shared<detail::CudaDeviceMemory>();
  if (bytes > 0 && cudaMalloc(&memory->data, bytes) != cudaSuccess) {
    throw std::runtime_error("make_device_tensor: cudaMalloc failed");
  }

  int current_device_id = -1;
  if (cudaGetDevice(&current_device_id) != cudaSuccess) {
    throw std::runtime_error("make_device_tensor: cudaGetDevice failed");
  }

  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->memory_reference = memory;
  context->dl_shape = shape;
  context->tensor.dl_tensor.data = memory->data;
  context->tensor.dl_tensor.device = DLDevice{kDLCUDA, current_device_id};
  context->tensor.dl_tensor.ndim = static_cast<int32_t>(context->dl_shape.size());
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;  // compact row-major
  context->tensor.dl_tensor.byte_offset = 0;

  return std::make_shared<holoscan::Tensor>(std::move(context));
}

}  // namespace holoscan::radar
