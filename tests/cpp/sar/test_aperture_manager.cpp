// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::ApertureManagerOp;
using holoscan::radar::ApertureManagerParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

TEST(ApertureManagerOp, RetainsEvictsAndFlushesOnTypedBoundary) {
  MaxTensorExtents extents;
  extents.max_pulses = 3;
  extents.max_range_bins = 2;
  extents.max_image_v = 1;
  extents.max_image_u = 1;
  auto fixture = holoscan::testing::create_operator_test<ApertureManagerOp<float>>(extents);
  fixture.set_params(ApertureManagerParams{.max_aperture_pulses = 2, .emit_interval_pulses = 0});
  std::vector<std::shared_ptr<holoscan::Tensor>> retained;
  for (int tick = 0; tick < 3; ++tick) {
    auto profile =
        to_device_tensor(std::vector<std::complex<float>>{{static_cast<float>(tick + 1), 0.F},
                                                          {static_cast<float>(tick + 1), 0.F}},
                         {1, 2}, complex_dtype<float>());
    auto position = to_device_tensor(std::vector<float>{static_cast<float>(tick), 0.F, 0.F}, {1, 3},
                                     real_dtype<float>());
    auto reference =
        to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{tick + 1}, holoscan::testing::input("range_profiles", *profile),
        holoscan::testing::input("positions", *position),
        holoscan::testing::input("reference_position", *reference),
        holoscan::testing::input("stream_boundary", static_cast<uint8_t>(tick == 2 ? 2 : 0))));
    retained.insert(retained.end(), {profile, position, reference});
  }
  for (const char* port : {"range_profiles", "positions", "reference_position"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  for (const char* port : {"range_profiles_out", "positions_out", "reference_position_out",
                           "weight", "sample_points_out", "control_out"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles_out", 3U, 3U * 2U * 8U);
  fixture.capture_snapshot<holoscan::Tensor>("positions_out", 3U, 3U * 3U * 4U);
  fixture.capture_snapshot<holoscan::Tensor>("reference_position_out", 3U, 3U * 3U * 4U);
  fixture.capture_snapshot<holoscan::Tensor>("weight", 3U, 3U * 4U);
  fixture.capture_snapshot<holoscan::Tensor>("control_out", 1U, 1U);
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "control_out", 1U) &&
                        completion.wait_for_output(running, "weight", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto weights = result.snapshots<holoscan::Tensor>("weight");
  ASSERT_EQ(weights.size(), 3U);
  auto last_weight = to_host(weights[2]);
  ASSERT_EQ(last_weight.shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(last_weight.as<float>()[0], 1.F);
  EXPECT_FLOAT_EQ(last_weight.as<float>()[1], -1.F);
  auto controls = result.snapshots<holoscan::Tensor>("control_out");
  ASSERT_EQ(controls.size(), 1U);
  EXPECT_EQ(to_host(controls[0]).as<uint8_t>()[0], 1U);
}

// A reform resets and replays the whole retained aperture, so every pulse is
// an admission and no pulse is a departure. With no `aperture_window` -- the
// WindowSpec default, and so the default reform configuration -- this used to
// fall through to the additions/departures layout and mark everything past the
// current tick's arrival count as weight -1, telling the backprojector to
// subtract most of the replay it had just reset for. Latent only because no
// app wires `weight` while reforming.
TEST(ApertureManagerOp, ReformWeightsTheWholeReplayPositively) {
  constexpr int64_t kTicks = 3;
  MaxTensorExtents extents;
  extents.max_pulses = kTicks;
  extents.max_range_bins = 2;
  extents.max_image_v = 1;
  extents.max_image_u = 1;
  auto fixture = holoscan::testing::create_operator_test<ApertureManagerOp<float>>(extents);
  fixture.set_params(ApertureManagerParams{
      .max_aperture_pulses = kTicks,
      .emit_interval_pulses = 0,
      .emit_mode = holoscan::radar::ApertureEmitMode::kReform,
  });
  std::vector<std::shared_ptr<holoscan::Tensor>> retained;
  for (int64_t tick = 0; tick < kTicks; ++tick) {
    auto profile =
        to_device_tensor(std::vector<std::complex<float>>{{static_cast<float>(tick + 1), 0.F},
                                                          {static_cast<float>(tick + 1), 0.F}},
                         {1, 2}, complex_dtype<float>());
    auto position = to_device_tensor(std::vector<float>{static_cast<float>(tick), 0.F, 0.F}, {1, 3},
                                     real_dtype<float>());
    auto reference =
        to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{tick + 1}, holoscan::testing::input("range_profiles", *profile),
        holoscan::testing::input("positions", *position),
        holoscan::testing::input("reference_position", *reference),
        holoscan::testing::input("stream_boundary",
                                 static_cast<uint8_t>(tick == kTicks - 1 ? 2 : 0))));
    retained.insert(retained.end(), {profile, position, reference});
  }
  for (const char* port : {"range_profiles", "positions", "reference_position"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  for (const char* port : {"range_profiles_out", "positions_out", "reference_position_out",
                           "weight", "sample_points_out", "control_out"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  // The three required outputs need a consumer or the emit fails the tick.
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, kTicks * 2U * 8U);
  fixture.capture_snapshot<holoscan::Tensor>("positions_out", 1U, kTicks * 3U * 4U);
  fixture.capture_snapshot<holoscan::Tensor>("reference_position_out", 1U, kTicks * 3U * 4U);
  fixture.capture_snapshot<holoscan::Tensor>("weight", 1U,
                                             static_cast<std::size_t>(kTicks) * sizeof(float));
  fixture.capture_snapshot<holoscan::Tensor>("control_out", 1U, 1U);
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "weight", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  // A reform emits once, on the tick that closes the aperture, and the replay
  // is the whole retention rather than this tick's arrivals.
  auto weights = result.snapshots<holoscan::Tensor>("weight");
  ASSERT_EQ(weights.size(), 1U);
  const auto replay = to_host(weights[0]);
  ASSERT_EQ(replay.shape, (std::vector<int64_t>{kTicks}));
  for (int64_t i = 0; i < kTicks; ++i) {
    EXPECT_FLOAT_EQ(replay.as<float>()[i], 1.F) << "replayed pulse " << i << " is not an admission";
  }
}
}  // namespace
