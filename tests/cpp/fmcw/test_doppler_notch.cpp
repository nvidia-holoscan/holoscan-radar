// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace radar_testing = holoscan::radar::testing;

using holoscan::radar::DopplerNotchOp;
using holoscan::radar::DopplerNotchParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 5;
constexpr int64_t kRange = 2;

MaxTensorExtents extents_for_rank(std::size_t rank) {
  MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;
  if (rank >= 3U)
    extents.max_rx = 2;
  if (rank == 4U)
    extents.max_tx = 2;
  return extents;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_axis(const std::vector<T>& values = std::vector<T>{
                                                T{-2}, T{-1}, T{0}, T{1}, T{2}}) {
  return to_device_tensor<T>(values, {kDoppler}, real_dtype<T>());
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_maps(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (const int64_t extent : shape)
    count *= extent;
  std::vector<std::complex<T>> values(static_cast<std::size_t>(count),
                                      std::complex<T>{T{2}, T{-1}});
  return to_device_tensor<std::complex<T>>(values, shape, complex_dtype<T>());
}

template <typename T>
holoscan::testing::TestResult run_notch(const std::vector<int64_t>& shape,
                                        DopplerNotchParams params,
                                        std::shared_ptr<holoscan::Tensor> axis = make_axis<T>(),
                                        std::optional<double> center = std::nullopt) {
  using Op = DopplerNotchOp<T>;
  auto maps = make_maps<T>(shape);
  auto fixture = holoscan::testing::create_operator_test<Op>(extents_for_rank(shape.size()));
  fixture.set_params(std::move(params));
  auto frame =
      center.has_value()
          ? holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                        holoscan::testing::input("velocity_bins", *axis),
                                        holoscan::testing::input("rd_maps", *maps),
                                        holoscan::testing::input("center_velocity", *center))
          : holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                        holoscan::testing::input("velocity_bins", *axis),
                                        holoscan::testing::input("rd_maps", *maps));
  fixture.push_frame(frame);
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 1U, maps->nbytes());
  fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("rd_maps_out", 1U, std::chrono::seconds{5});
  (void)observed;
  running.request_stop();
  return running.finish();
}

std::vector<HostTensor> host_frames(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>("rd_maps_out")) {
    frames.push_back(radar_testing::to_host(snapshot));
  }
  return frames;
}

template <typename T>
void expect_gain(const HostTensor& output, int64_t doppler, T expected) {
  const auto* values = output.as<std::complex<T>>();
  for (int64_t index = doppler * kRange; index < (doppler + 1) * kRange; ++index) {
    EXPECT_NEAR(values[index].real(), T{2} * expected, T{1e-5});
    EXPECT_NEAR(values[index].imag(), T{-1} * expected, T{1e-5});
  }
}

TEST(DopplerNotchOp, HardNullsNearestCenterBin) {
  const auto result = run_notch<float>({kDoppler, kRange}, DopplerNotchParams{});
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 1.0F);
  expect_gain(frames[0], 1, 1.0F);
  expect_gain(frames[0], 2, 0.0F);
  expect_gain(frames[0], 3, 1.0F);
  expect_gain(frames[0], 4, 1.0F);
}

TEST(DopplerNotchOp, RaisedCosineMatchesPhysicalAxis) {
  DopplerNotchParams params;
  params.transition_width = 2.0;
  const auto result = run_notch<double>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 1.0);
  expect_gain(frames[0], 1, 0.5);
  expect_gain(frames[0], 2, 0.0);
  expect_gain(frames[0], 3, 0.5);
  expect_gain(frames[0], 4, 1.0);
}

TEST(DopplerNotchOp, CyclicDistanceWrapsAtAxisEdge) {
  DopplerNotchParams params;
  params.center_velocity = 2.0;
  params.transition_width = 2.0;
  const auto result = run_notch<float>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 0.5F);
  expect_gain(frames[0], 4, 0.0F);
}

TEST(DopplerNotchOp, NamedRanksPreserveShapeAndChannelPhase) {
  for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{
           {kDoppler, kRange}, {2, kDoppler, kRange}, {2, 2, kDoppler, kRange}}) {
    DopplerNotchParams params;
    params.transition_width = 2.0;
    const auto result = run_notch<double>(shape, params);
    ASSERT_TRUE(result.ok()) << result.describe_failure();

    const auto frames = host_frames(result);
    ASSERT_EQ(frames.size(), 1U);
    EXPECT_EQ(frames[0].shape, shape);
    const auto* values = frames[0].as<std::complex<double>>();
    int64_t count = 1;
    for (const int64_t extent : shape)
      count *= extent;
    for (int64_t index = 0; index < count; ++index) {
      if (values[index].real() != 0.0) {
        EXPECT_NEAR(values[index].imag() / values[index].real(), -0.5, 1e-12);
      }
    }
  }
}

TEST(DopplerNotchOp, LiveCenterOverrideAppliesForOneTick) {
  using Op = DopplerNotchOp<float>;
  auto axis0 = make_axis<float>();
  auto axis1 = make_axis<float>();
  auto maps0 = make_maps<float>({kDoppler, kRange});
  auto maps1 = make_maps<float>({kDoppler, kRange});
  auto fixture = holoscan::testing::create_operator_test<Op>(extents_for_rank(2U));
  fixture.set_params(DopplerNotchParams{});
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{100},
                                                 holoscan::testing::input("center_velocity", 1.0),
                                                 holoscan::testing::input("velocity_bins", *axis0),
                                                 holoscan::testing::input("rd_maps", *maps0)));
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{200},
                                                 holoscan::testing::input("velocity_bins", *axis1),
                                                 holoscan::testing::input("rd_maps", *maps1)));
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 2U, maps0->nbytes());
  fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  auto running =
      fixture.run_async({.expect_stop = true, .frames = holoscan::testing::FrameRelease::kManual});
  const auto released0 = running.release_frame(std::chrono::seconds{5});
  const bool observed0 = running.wait_for_output("rd_maps_out", 1U, std::chrono::seconds{5});
  const auto released1 = running.release_frame(std::chrono::seconds{5});
  const bool observed1 = running.wait_for_output("rd_maps_out", 2U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(released0) << released0.error().message;
  ASSERT_TRUE(released1) << released1.error().message;
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(observed0);
  ASSERT_TRUE(observed1);
  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 2U);
  expect_gain(frames[0], 3, 0.0F);
  expect_gain(frames[1], 2, 0.0F);
}

TEST(DopplerNotchOp, RejectsVelocityLengthMismatch) {
  auto short_axis = to_device_tensor<float>({-1.0F, 0.0F, 1.0F}, {3}, real_dtype<float>());
  const auto result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, std::move(short_axis));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DopplerNotchOp, RejectsPrecisionMismatch) {
  const auto result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, make_axis<double>());
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DopplerNotchOp, RejectsNegativeWidths) {
  DopplerNotchParams params;
  params.stop_half_width = -1.0;
  const auto result = run_notch<float>({kDoppler, kRange}, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

}  // namespace
