// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device tensor allocation (radar_common/tensor): make_device_tensor shape
// validation -- empty dims are legal, negative and overflowing products throw
// before any allocation.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

TEST(MakeDeviceTensor, RejectsInvalidAndOverflowingShapes) {
  const auto dtype = holoscan::radar::complex_dtype<float>();
  // Zero-sized dims are valid: an empty tensor is a legitimate pipeline state
  // (e.g. zero detections → empty detection_idx / points) that operators
  // allocate-then-check. Only negative dims and int64-overflowing products
  // must throw before any allocation (a wrapped count would produce an
  // undersized buffer that full-extent views write past).
  EXPECT_NO_THROW(holoscan::radar::make_device_tensor({0, 4}, dtype));
  EXPECT_THROW(holoscan::radar::make_device_tensor({-1, 4}, dtype), std::invalid_argument);
  EXPECT_THROW(holoscan::radar::make_device_tensor({int64_t{1} << 32, int64_t{1} << 32}, dtype),
               std::invalid_argument);
}

TEST(CudaAccessibility, RejectsPageableHostTensorWithContext) {
  const auto tensor = holoscan::radar::testing::to_pageable_host_tensor<float>(
      {1.0F}, {1}, holoscan::radar::real_dtype<float>());

  EXPECT_FALSE(holoscan::radar::is_cuda_accessible(*tensor));
  try {
    holoscan::radar::require_cuda_accessible(*tensor, "input", "TestOp");
    FAIL() << "expected pageable-host Tensor rejection";
  } catch (const std::invalid_argument& error) {
    EXPECT_EQ(std::string(error.what()),
              "TestOp: tensor received on input port 'input' is not "
              "CUDA-accessible (device, managed, or pinned-host memory "
              "required)");
  }
}

TEST(Contiguity, RejectsNonContiguousTensorWithContext) {
  struct Holder {
    void* data{nullptr};
    std::vector<int64_t> shape{1, 2};
    std::vector<int64_t> strides{2, 2};
    DLManagedTensor dl{};
  };
  auto holder = std::make_unique<Holder>();
  ASSERT_EQ(cudaMalloc(&holder->data, 4 * sizeof(int64_t)), cudaSuccess);
  const std::vector<int64_t> host = {0, 0, 1, 0};
  ASSERT_EQ(cudaMemcpy(holder->data, host.data(), 4 * sizeof(int64_t), cudaMemcpyHostToDevice),
            cudaSuccess);
  int current_device_id = -1;
  ASSERT_EQ(cudaGetDevice(&current_device_id), cudaSuccess);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, current_device_id};
  holder->dl.dl_tensor.ndim = 2;
  holder->dl.dl_tensor.dtype = DLDataType{kDLInt, 64, 1};
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = holder->strides.data();
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    auto* context = static_cast<Holder*>(self->manager_ctx);
    cudaFree(context->data);
    delete context;
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();

  ASSERT_FALSE(tensor->is_contiguous());
  try {
    holoscan::radar::require_contiguous(*tensor, "input", "TestOp");
    FAIL() << "expected non-contiguous Tensor rejection";
  } catch (const std::invalid_argument& error) {
    EXPECT_EQ(std::string(error.what()),
              "TestOp: tensor received on input port 'input' must be "
              "contiguous (compact row-major)");
  }
}

}  // namespace
