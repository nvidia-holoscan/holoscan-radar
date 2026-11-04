// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "radar_app/app/gpu_placement.hpp"

namespace {

namespace app = holoscan::radar::app;

class ApplicationGpuPlacement : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess); }

  void TearDown() override { EXPECT_EQ(cudaSetDevice(original_device_), cudaSuccess); }

  int original_device_ = -1;
};

TEST_F(ApplicationGpuPlacement, NonvisualGraphRetainsCallersDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_GT(count, 0);
  const int selected = count - 1;
  ASSERT_EQ(cudaSetDevice(selected), cudaSuccess);
  EXPECT_EQ(app::select_application_device(false, false), selected);
  EXPECT_EQ(app::application_device().value, selected);
}

}  // namespace
