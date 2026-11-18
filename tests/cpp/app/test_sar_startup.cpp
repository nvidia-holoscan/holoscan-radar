// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/sar/hud_format.hpp"
#include "radar_app/sar/sar_image_viz.hpp"
#include "radar_app/sar/sar_overlay.hpp"
#include "radar_app/sar/timeline_raster.hpp"
#include "radar_app/viz/frame_composer_op.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
namespace sar = holoscan::radar::app::sar;
namespace viz = holoscan::radar::app::viz;
namespace testing = holoscan::radar::testing;
using holoscan::testing::frame_at;
using holoscan::testing::input;

TEST(SarStartup, EmitsOnePlaceholderThenPreservesTheRealImageStamp) {
  const holoscan::DeviceId device{holoscan::radar::app::application_device()};
  std::vector<std::complex<float>> data(16);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = {static_cast<float>(i + 1), 0};
  const auto image = testing::to_device_tensor(data, {4, 4}, DLDataType{kDLComplex, 64U, 1U});
  auto harness = holoscan::testing::create_operator_test<sar::SarImageVizOp>(
      4, 4, 4, false, sar::OverlayConfig{}, std::array<int64_t, 2>{0, 0}, 60.0,
      std::function<bool()>{}, true);
  harness.push_frame(frame_at(std::chrono::milliseconds{123}, input("image", *image)));
  harness.input_source_device("image", device);
  harness.output_device("frame", device);
  harness.capture_snapshot<holoscan::schema::ImageT>("frame", 2U, 4 * 4 * 4);
  harness.capture_snapshot<holoscan::Tensor>("labels", 2U, sar::kHudRuns * sar::kHudRowBytes);
  harness.capture_snapshot<holoscan::Tensor>(
      "overlay", 2U, sar::kHudMaxLines * sar::kHudLineFloats * sizeof(float));
  testing::FixtureCompletion completion;
  auto running =
      harness.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
  const bool startup = completion.wait_for_output(running, "frame", 1U);
  // Clock ticks while waiting must not regenerate or transport the main canvas.
  const bool repeated = running.wait_for_output("frame", 2U, std::chrono::milliseconds{150});
  const auto released = running.release_frame(completion.remaining());
  const bool formed = completion.wait_for_output(running, "frame", 2U);
  const bool labels = completion.wait_for_output(running, "labels", 2U);
  const bool replayed = running.wait_for_output("frame", 3U, std::chrono::milliseconds{150});
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(startup);
  EXPECT_FALSE(repeated);
  ASSERT_TRUE(released);
  ASSERT_TRUE(formed);
  ASSERT_TRUE(labels);
  EXPECT_FALSE(replayed);
  const auto frames = result.snapshots<holoscan::schema::ImageT>("frame");
  const auto text = result.snapshots<holoscan::Tensor>("labels");
  ASSERT_EQ(frames.size(), 2U);
  ASSERT_EQ(text.size(), 2U);
  EXPECT_EQ(frames[0].sample_metadata().capture_time().timestamp_ns, -1);
  EXPECT_EQ(text[0].sample_metadata().capture_time().timestamp_ns, -1);
  EXPECT_EQ(frames[1].sample_metadata().capture_time().timestamp_ns, 123000000);
  EXPECT_EQ(text[1].sample_metadata().capture_time().timestamp_ns, 123000000);
  const auto pixel = testing::to_host(frames[0].tensor(holoscan::schema::ImageTensorFields::data));
  ASSERT_EQ(pixel.shape, (std::vector<int64_t>{1, 1, 4}));
  EXPECT_EQ(pixel.as<std::uint8_t>()[0], 19);
  EXPECT_EQ(pixel.as<std::uint8_t>()[3], 255);
  EXPECT_EQ(testing::to_host(frames[1].tensor(holoscan::schema::ImageTensorFields::data)).shape,
            (std::vector<int64_t>{4, 4, 4}));
}

TEST(SarStartup, ComposesDistinctNativeTextWithoutAdvancingAcquisitionTime) {
  constexpr int width = 640, height = 256, footer = 112;
  constexpr DLDataType rgba{kDLUInt, 8U, 1U};
  sar::OverlayConfig layout;
  layout.canvas_width = width;
  layout.image_height = height;
  layout.footer_height = footer;
  layout.status_width = 320;
  layout.scene_names = {"Panama Canal"};
  std::vector<std::uint8_t> records(sar::kHudRuns * sar::kHudRowBytes);
  sar::write_startup_hud_rows(records.data(), layout);
  const auto text =
      testing::to_pageable_host_tensor(records, {sar::kHudRuns, sar::kHudRowBytes}, rgba);
  holoscan::schema::ImageT image;
  image.width = 1;
  image.height = 1;
  image.encoding = holoscan::schema::ImageEncoding_RGBA8;
  image.data =
      testing::to_device_tensor(std::vector<std::uint8_t>{19, 24, 31, 255}, {1, 1, 4}, rgba);
  auto harness = holoscan::testing::create_operator_test<viz::FrameComposerOp>(
      width, height + footer, sar::kHudMaxGlyphs, true, 0U, 0, height);
  harness.push_frame(
      frame_at(std::chrono::nanoseconds{-1}, input("image", image), input("text", *text)));
  const holoscan::DeviceId device{holoscan::radar::app::application_device()};
  harness.input_source_device("image", device);
  harness.output_device("output", device);
  harness.capture_snapshot<holoscan::schema::ImageT>("output", 1U, width * (height + footer) * 4);
  testing::FixtureCompletion completion;
  auto running = harness.run_async(completion.run_options());
  const bool ready = completion.wait_for_output(running, "output", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(ready) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frame = result.snapshot<holoscan::schema::ImageT>("output");
  EXPECT_EQ(frame.sample_metadata().capture_time().timestamp_ns, -1);
  const auto pixels = testing::to_host(frame.tensor(holoscan::schema::ImageTensorFields::data));
  const auto* data = pixels.as<std::uint8_t>();
  EXPECT_EQ(data[0], 19);
  EXPECT_EQ(data[1], 24);
  EXPECT_EQ(data[2], 31);
  int accent = 0, title = 0;
  for (int y = 50; y < 190; ++y) {
    for (int x = 100; x < 540; ++x) {
      const auto i = 4 * (y * width + x);
      accent += data[i + 1] > 150 && data[i] < 100;
      title += data[i] > 150 && data[i + 1] > 150;
    }
  }
  EXPECT_GT(accent, 20);
  EXPECT_GT(title, 100);
}

TEST(SarTimeline, PlacesCollectionTitleAboveItsColoredBar) {
  constexpr int width = 640, height = 112;
  auto hud = std::make_shared<holoscan::radar::FormationHud>();
  hud->publish_reader(
      {holoscan::radar::CollectionSpan{
          .file_index = 0,
          .dwell_start = std::chrono::steady_clock::now() - std::chrono::seconds{10}}},
      {}, {}, {});
  auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
      hud, std::vector<std::string>{"Panama Canal"}, false, 10.0, width, height);
  harness.output_device("frame", holoscan::DeviceId{holoscan::radar::app::application_device()});
  harness.capture_snapshot<holoscan::schema::ImageT>("frame", 1U, width * height * 4);
  harness.capture_snapshot<holoscan::Tensor>("labels", 1U, 65536U);
  testing::FixtureCompletion completion;
  auto running = harness.run_async(completion.run_options());
  const bool frame_ready = completion.wait_for_output(running, "frame", 1U);
  const bool text_ready = completion.wait_for_output(running, "labels", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(frame_ready);
  ASSERT_TRUE(text_ready);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frame = result.snapshot<holoscan::schema::ImageT>("frame");
  const auto pixels = testing::to_host(frame.tensor(holoscan::schema::ImageTensorFields::data));
  const auto records = testing::to_host(result.snapshot<holoscan::Tensor>("labels"));
  const auto* rgba = pixels.as<std::uint8_t>();
  int bar_top = height;
  for (int y = 0; y < height; ++y) {
    const int pixel = 4 * (y * width + width * 3 / 4);
    if (rgba[pixel] != 16 || rgba[pixel + 1] != 16 || rgba[pixel + 2] != 16) {
      bar_top = y;
      break;
    }
  }
  ASSERT_LT(bar_top, height);
  bool found = false;
  for (int64_t row = 0; row < records.shape[0]; ++row) {
    const auto* record = records.as<std::uint8_t>() + row * records.shape[1];
    if (std::memcmp(record + sar::kTextRunHeaderBytes, "Panama Canal", 12) != 0)
      continue;
    holoscan::visualization::TextRunHeader header;
    std::memcpy(&header, record, sizeof(header));
    EXPECT_LT(header.y + header.size_px, bar_top);
    found = true;
  }
  EXPECT_TRUE(found) << "The collection title must remain attached to its visible span";
}

TEST(SarDisplayRate, UsesReadablePrefixesWithoutChangingConsoleMeasurements) {
  const auto report = sar::format_bp_rate_line(0.02, 1000, 248000, 0, 0, 0, 0);
  EXPECT_EQ(report, "20ms  1.24e10 px*pls/s");
  EXPECT_EQ(sar::format_display_rate_line(report), "20ms  12.4 Gpx*pulse/s");
  EXPECT_EQ(sar::format_display_rate_line("64x64  1ms  1.75e6 px*pls/s"),
            "64x64  1ms  1.8 Mpx*pulse/s");
  EXPECT_EQ(sar::format_display_rate_line("1ms  9.9999e8 px*pls/s"), "1ms  1.0 Gpx*pulse/s");
  EXPECT_EQ(sar::format_display_rate_line("1ms  7.43e4 px*pls/s"), "1ms  74.3 kpx*pulse/s");
  EXPECT_EQ(sar::format_display_rate_line("Unknown"), "Unknown");
  EXPECT_EQ(sar::format_display_rate_line("1ms  nan px*pls/s"), "1ms  nan px*pls/s");
}
}  // namespace
