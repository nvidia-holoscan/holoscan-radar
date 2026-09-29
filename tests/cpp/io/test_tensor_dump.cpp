// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

TEST(TensorDumpOp, WritesNpyHeaderAndInputBytes) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("sar_tensor_dump_" + std::to_string(::getpid()) + ".npy");
  std::filesystem::remove(path);
  auto fixture = holoscan::testing::create_operator_test<holoscan::radar::TensorDumpOp>();
  holoscan::radar::TensorDumpParams params;
  params.path = path.string();
  params.max_files = 1;
  fixture.set_params(params);
  const std::vector<float> values{1.25F, -2.5F, 3.0F};
  auto input =
      holoscan::radar::testing::to_device_tensor(values, {3}, holoscan::radar::real_dtype<float>());
  fixture.push(&holoscan::radar::TensorDumpOp::tensor, *input);
  fixture.input_source_device("tensor", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  for (int attempt = 0; attempt < 500 && !std::filesystem::exists(path); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(std::filesystem::exists(path));
  std::ifstream stream(path, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  EXPECT_GE(bytes.size(), input->nbytes() + 10U);
  EXPECT_EQ(bytes.substr(0, 6), "\x93NUMPY");
  std::filesystem::remove(path);
}

}  // namespace
