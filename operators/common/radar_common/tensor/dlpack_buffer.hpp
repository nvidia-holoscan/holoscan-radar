// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace detail {

// Ownership block for a device allocation exported through DLPack. The
// deleter runs when the last holoscan::Tensor reference is released.
struct DeviceBufferHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  DLManagedTensor dl_managed{};
};

}  // namespace detail

// Allocate an uninitialized device buffer of the given full-rank shape and
// wrap it as a holoscan::Tensor via DLPack. The Tensor owns the allocation
// (cudaFree on release, which orders after outstanding work on the device).
//
// Compute kernels view the same memory through rank-collapsed MatX views
// (radar_common/tensor/matx_utils.hpp) while the emitted Tensor keeps the
// caller's full-rank shape — this is how rank-agnostic batching round-trips
// leading axes through the operators.
inline std::shared_ptr<holoscan::Tensor> make_device_tensor(
    const std::vector<int64_t>& shape, DLDataType dtype) {
  // Reject non-positive dims and int64 overflow before the size_t cast: a
  // wrapped count would allocate an undersized buffer that downstream
  // full-extent views write past.
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim) {
      throw std::invalid_argument(
          "make_device_tensor: invalid or overflowing shape");
    }
    count *= dim;
  }
  const size_t bytes = static_cast<size_t>(count) *
                       ((static_cast<size_t>(dtype.bits) + 7) / 8) *
                       dtype.lanes;

  auto holder = std::make_unique<detail::DeviceBufferHolder>();
  holder->shape = shape;
  if (cudaMalloc(&holder->data, bytes) != cudaSuccess) {
    throw std::runtime_error("make_device_tensor: cudaMalloc failed");
  }

  int device_id = 0;
  cudaGetDevice(&device_id);

  DLManagedTensor& dl = holder->dl_managed;
  dl.dl_tensor.data = holder->data;
  dl.dl_tensor.device = DLDevice{kDLCUDA, device_id};
  dl.dl_tensor.ndim = static_cast<int32_t>(holder->shape.size());
  dl.dl_tensor.dtype = dtype;
  dl.dl_tensor.shape = holder->shape.data();
  dl.dl_tensor.strides = nullptr;  // compact row-major
  dl.dl_tensor.byte_offset = 0;
  dl.manager_ctx = holder.get();
  dl.deleter = [](DLManagedTensor* self) {
    auto* h = static_cast<detail::DeviceBufferHolder*>(self->manager_ctx);
    cudaFree(h->data);
    delete h;
  };

  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl_managed);
  holder.release();  // ownership transferred to the DLPack deleter
  return tensor;
}

}  // namespace holoscan::radar
