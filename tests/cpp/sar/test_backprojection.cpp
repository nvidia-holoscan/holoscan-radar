// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::BackprojectionOp;
using holoscan::radar::BackprojectionParams;
using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameSpec;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

TEST(BackprojectionOp, PointTargetFormsAtSceneCentreAndResets) {
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<float>>(extents);
  BackprojectionParams params;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -2.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  fixture.set_params(params);
  std::vector<std::complex<float>> impulse(64U, {0.F, 0.F});
  impulse[32] = {1.F, 0.F};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<float>());
  auto positions =
      to_device_tensor(std::vector<float>{0.F, 0.F, 100.F}, {1, 3}, real_dtype<float>());
  auto reference = to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
  auto control = to_device_tensor(std::vector<uint8_t>{1U}, {1}, DLDataType{kDLUInt, 8U, 1U});
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{1}, holoscan::testing::input("control", *control),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{2}, holoscan::testing::input("control", *control),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  for (const char* port : {"range_profiles", "positions", "reference_position", "control"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.capture_snapshot<holoscan::Tensor>("image", 2U, 5U * 5U * sizeof(std::complex<float>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 2U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.snapshots<holoscan::Tensor>("image");
  ASSERT_EQ(captures.size(), 2U);
  const auto image = to_host(captures[0]);
  const auto reset_image = to_host(captures[1]);
  ASSERT_EQ(image.shape, (std::vector<int64_t>{5, 5}));
  const auto* data = image.as<std::complex<float>>();
  EXPECT_NEAR(std::abs(data[12]), 1.F, 1e-3F);
  for (int64_t i = 0; i < 25; ++i) {
    EXPECT_LE(std::abs(data[i]), std::abs(data[12]) + 1e-3F);
    EXPECT_NEAR(std::abs(reset_image.as<std::complex<float>>()[i] - data[i]), 0.F, 1e-3F)
        << "reset retained accumulation at pixel " << i;
  }
}

TEST(BackprojectionOp, ScrollRetainsRowsAndRepeatedGeometryIsNoOp) {
  using holoscan::radar::GeometryChange;
  using holoscan::radar::to_wire;
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<float>>(extents);
  BackprojectionParams params;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -2.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  params.geometry_change = GeometryChange::kScroll;
  fixture.set_params(params);
  const auto moved = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -1.0, 0.0},
                                                       .axis_u = {1.0, 0.0, 0.0},
                                                       .axis_v = {0.0, 1.0, 0.0},
                                                       .spacing = {1.0, 1.0},
                                                       .counts = {5, 5}});
  std::vector<std::complex<float>> impulse(64U, {0.F, 0.F});
  impulse[32] = {1.F, 0.F};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<float>());
  auto positions =
      to_device_tensor(std::vector<float>{0.F, 0.F, 100.F}, {1, 3}, real_dtype<float>());
  auto reference = to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
  auto control = to_device_tensor(std::vector<uint8_t>{0U}, {1}, DLDataType{kDLUInt, 8U, 1U});
  for (int tick = 0; tick < 3; ++tick) {
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{tick + 1},
        holoscan::testing::input("sample_geometry",
                                 to_wire(tick == 2 ? moved : params.sample_geometry)),
        holoscan::testing::input("control", *control),
        holoscan::testing::input("range_profiles", *profiles),
        holoscan::testing::input("positions", *positions),
        holoscan::testing::input("reference_position", *reference)));
  }
  for (const char* port : {"range_profiles", "positions", "reference_position", "control"})
    fixture.input_source_device(port, holoscan::DeviceId{0});
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.capture_snapshot<holoscan::Tensor>("image", 3U, 5U * 5U * sizeof(std::complex<float>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.snapshots<holoscan::Tensor>("image");
  ASSERT_EQ(captures.size(), 3U);
  const auto first = to_host(captures[0]);
  const auto second = to_host(captures[1]);
  const auto third = to_host(captures[2]);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  const auto* c = third.as<std::complex<float>>();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 5; ++col) {
      const int old_index = (row + 1) * 5 + col;
      EXPECT_NEAR(std::abs(b[old_index] - 2.F * a[old_index]), 0.F, 2e-3F);
      EXPECT_NEAR(std::abs(c[row * 5 + col] - 3.F * a[old_index]), 0.F, 3e-3F);
    }
  }
}
}  // namespace
