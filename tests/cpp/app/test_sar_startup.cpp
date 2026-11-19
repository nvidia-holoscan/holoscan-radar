// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/sar/hud_format.hpp"
#include "radar_app/sar/autofocus_stage_op.hpp"
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

TEST(SarTimeline, ShowsSeparateSourceBatchReceiptAndCompletedProcessingLanes) {
  constexpr int width = 704, height = 112;
  auto hud = std::make_shared<holoscan::radar::FormationHud>();
  hud->enable_progress(0, 20);
  auto monitor = hud->progress();
  monitor->begin_packet(1, 0, 0, 20, 100, 10, 1, true);
  const auto receipt = std::chrono::steady_clock::now() - std::chrono::seconds{35};
  monitor->received(1, receipt);
  monitor->admit(1, 20, false, true);
  const auto work = monitor->take_work(1);
  ASSERT_TRUE(work);
  monitor->processed(*work, receipt + std::chrono::seconds{5});
  auto viewport =
      std::make_shared<holoscan::radar::app::viz::FrameViewport>(width, height, width, height);
  auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
      hud, std::vector<std::string>{"Panama Canal"}, false, 10.0, width, height,
      std::shared_ptr<holoscan::radar::app::FrameProgress>{}, viewport);
  harness.output_device("frame", holoscan::DeviceId{holoscan::radar::app::application_device()});
  harness.capture_snapshot<holoscan::schema::ImageT>("frame", 4U, width * height * 4);
  harness.capture_snapshot<holoscan::Tensor>("labels", 4U, 65536U);
  testing::FixtureCompletion completion;
  auto running = harness.run_async(completion.run_options());
  const bool frame_ready = completion.wait_for_output(running, "frame", 1U);
  const bool text_ready = completion.wait_for_output(running, "labels", 1U);
  // History older than 20 seconds of wall time stays visible, and idle clock
  // ticks do not allocate/emit identical GPU frames. Both event kinds advance it.
  const bool idle_frame = running.wait_for_output("frame", 2U, std::chrono::milliseconds{250});
  monitor->begin_packet(2, 0, 20, 20, 100, 10, 1, false);
  monitor->received(2, receipt + std::chrono::seconds{10});
  const bool receipt_frame = completion.wait_for_output(running, "frame", 2U) &&
                             completion.wait_for_output(running, "labels", 2U);
  monitor->admit(2, 20, false, true);
  const auto second_work = monitor->take_work(2);
  if (second_work)
    monitor->processed(*second_work, receipt + std::chrono::seconds{15});
  const bool completed_frame = completion.wait_for_output(running, "frame", 3U) &&
                               completion.wait_for_output(running, "labels", 3U);
  viewport->set(width - 32, height);
  const bool resized_frame = completion.wait_for_output(running, "frame", 4U) &&
                             completion.wait_for_output(running, "labels", 4U);
  const bool after_resize = running.wait_for_output("frame", 5U, std::chrono::milliseconds{1200});
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  EXPECT_FALSE(idle_frame);
  EXPECT_FALSE(after_resize);
  ASSERT_TRUE(receipt_frame);
  ASSERT_TRUE(second_work);
  ASSERT_TRUE(completed_frame);
  ASSERT_TRUE(resized_frame);
  ASSERT_TRUE(frame_ready);
  ASSERT_TRUE(text_ready);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frame = result.snapshots<holoscan::schema::ImageT>("frame").front();
  const auto pixels = testing::to_host(frame.tensor(holoscan::schema::ImageTensorFields::data));
  const auto records = testing::to_host(result.snapshots<holoscan::Tensor>("labels").front());
  bool received = false, processed = false;
  for (int64_t row = 0; row < records.shape[0]; ++row) {
    const auto* record = records.as<std::uint8_t>() + row * records.shape[1];
    const std::string text{reinterpret_cast<const char*>(record + sar::kTextRunHeaderBytes)};
    received |= text == "Pulse batch received";
    processed |= text == "Pulse batch processed";
  }
  EXPECT_TRUE(received);
  EXPECT_TRUE(processed);
  int receive_ticks = 0, process_ticks = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 264; x < width - 2; ++x) {
      const auto* pixel = pixels.as<std::uint8_t>() + 4 * (y * width + x);
      if (pixel[0] == 75 && pixel[1] == 205 && pixel[2] == 220) {
        ++receive_ticks;
        EXPECT_LT(y, height / 2);
      }
      if (pixel[0] == 115 && pixel[1] == 200 && pixel[2] == 150) {
        ++process_ticks;
        EXPECT_GT(y, height / 2);
      }
    }
  }
  EXPECT_GT(receive_ticks, 0);
  EXPECT_GT(process_ticks, 0);
}

TEST(SarTimeline, DrawsMatchedDiagonalsWithoutFillAndPlacesLargerLabelsOutsideTracks) {
  constexpr int height = 112;
  for (const int width : {320, 704}) {
    for (const int scenario : {0, 1, 2}) {
      const bool completed = scenario != 0;
      SCOPED_TRACE(width);
      SCOPED_TRACE(scenario);
      auto hud = std::make_shared<holoscan::radar::FormationHud>();
      hud->enable_progress(0, 20);
      const auto monitor = hud->progress();
      const auto receipt =
          std::chrono::steady_clock::now() - std::chrono::seconds(scenario == 2 ? 24 : 8);
      monitor->begin_packet(1, 0, 0, 20, 20, 10, 1, true);
      monitor->received(1, receipt);
      if (completed) {
        monitor->admit(1, 20, true, true);
        const auto work = monitor->take_work(1);
        ASSERT_TRUE(work);
        monitor->processed(*work, receipt + std::chrono::seconds(scenario == 2 ? 20 : 4));
      }
      auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
          hud, std::vector<std::string>{"Panama Canal"}, false, 10.0, width, height);
      harness.output_device("frame",
                            holoscan::DeviceId{holoscan::radar::app::application_device()});
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
      const bool compact = width < 620;
      const int history_x = compact ? 0 : 264, history_y = compact ? 56 : 0;
      const int received_y = history_y + (compact ? 18 : 36);
      const int processed_y = history_y + (compact ? 34 : 70);
      const auto connector_columns = [&](int y) {
        std::vector<int> columns;
        for (int x = history_x; x < width; ++x) {
          const auto* pixel = pixels.as<std::uint8_t>() + 4 * (y * width + x);
          if (pixel[0] == 75 && pixel[1] == 94 && pixel[2] == 113)
            columns.push_back(x);
        }
        return columns;
      };
      const auto upper = connector_columns(received_y + (processed_y - received_y) / 4);
      const auto lower = connector_columns(received_y + 3 * (processed_y - received_y) / 4);
      if (completed) {
        ASSERT_FALSE(upper.empty());
        ASSERT_FALSE(lower.empty());
        EXPECT_GT(lower.front(), upper.back()) << "A matched delay slopes toward newer time";
        EXPECT_LT(lower.size(), static_cast<std::size_t>((width - history_x) / 4))
            << "A connection must be a thin line, never an interval fill";
      } else {
        EXPECT_TRUE(upper.empty());
        EXPECT_TRUE(lower.empty()) << "Pending work has no completion endpoint";
      }
      int green_on_track = 0;
      for (int x = history_x; x < width; ++x) {
        const auto* pixel = pixels.as<std::uint8_t>() + 4 * (processed_y * width + x);
        green_on_track += pixel[0] == 115 && pixel[1] == 200 && pixel[2] == 150;
      }
      EXPECT_EQ(green_on_track, completed ? 2 : 0);
      const auto* at_now =
          pixels.as<std::uint8_t>() + 4 * (((received_y + processed_y) / 2) * width + width - 9);
      EXPECT_EQ(at_now[0], 0);
      EXPECT_EQ(at_now[1], 0);
      EXPECT_EQ(at_now[2], 0);
      std::optional<holoscan::visualization::TextRunHeader> received_label, processed_label;
      std::optional<holoscan::visualization::TextRunHeader> past_label, latest_label;
      bool relative_caption = false;
      for (int64_t row = 0; row < records.shape[0]; ++row) {
        const auto* record = records.as<std::uint8_t>() + row * records.shape[1];
        const std::string text{reinterpret_cast<const char*>(record + sar::kTextRunHeaderBytes)};
        holoscan::visualization::TextRunHeader header;
        std::memcpy(&header, record, sizeof(header));
        if (text == "Pulse batch received")
          received_label = header;
        else if (text == "Pulse batch processed")
          processed_label = header;
        else if (text == "-20s")
          past_label = header;
        else if (text == "Latest batch")
          latest_label = header;
        else if (text == "Time relative to last batch update")
          relative_caption = true;
        EXPECT_NE(text, "Now");
      }
      ASSERT_TRUE(received_label);
      ASSERT_TRUE(processed_label);
      ASSERT_TRUE(past_label);
      ASSERT_TRUE(latest_label);
      EXPECT_EQ(relative_caption, !compact);
      const int font = compact ? 14 : 16;
      EXPECT_EQ(received_label->size_px, font);
      EXPECT_EQ(processed_label->size_px, font);
      EXPECT_EQ(past_label->size_px, font);
      EXPECT_EQ(latest_label->size_px, font);
      EXPECT_LT(received_label->y + font, received_y);
      EXPECT_GT(processed_label->y, processed_y + 2);
      EXPECT_EQ(past_label->y, latest_label->y);
      EXPECT_GT(past_label->y, received_y);
      EXPECT_LT(past_label->y + font, processed_y);
      const int advance = font * 7 / 13;
      EXPECT_LT(past_label->x + 4 * advance, history_x + (compact ? 40 : 64));
      EXPECT_GT(latest_label->x, width - 16 - 12 * font * 7.0 / 13);
      EXPECT_LE(latest_label->x + 12 * advance, width - 8);
      if (scenario == 1) {
        // Compare actual native-rendered pixels for the common "Pulse batch "
        // prefix, independently of the distinct words' ascenders/descenders.
        constexpr DLDataType rgba{kDLUInt, 8U, 1U};
        holoscan::schema::ImageT canvas;
        canvas.width = width;
        canvas.height = height;
        canvas.encoding = holoscan::schema::ImageEncoding_RGBA8;
        canvas.data = testing::to_device_tensor(
            std::vector<std::uint8_t>(pixels.as<std::uint8_t>(),
                                      pixels.as<std::uint8_t>() + width * height * 4),
            {height, width, 4}, rgba);
        const auto text = testing::to_pageable_host_tensor(
            std::vector<std::uint8_t>(
                records.as<std::uint8_t>(),
                records.as<std::uint8_t>() + records.shape[0] * records.shape[1]),
            records.shape, rgba);
        auto composer =
            holoscan::testing::create_operator_test<viz::FrameComposerOp>(width, height, 16384U);
        composer.push_frame(
            frame_at(std::chrono::nanoseconds{123}, input("image", canvas), input("text", *text)));
        const holoscan::DeviceId device{holoscan::radar::app::application_device()};
        composer.input_source_device("image", device);
        composer.output_device("output", device);
        composer.capture_snapshot<holoscan::schema::ImageT>("output", 1U, width * height * 4);
        auto composing = composer.run_async(completion.run_options());
        const bool ready = completion.wait_for_output(composing, "output", 1U);
        composing.request_stop();
        const auto composed = composing.finish();
        completion.check(composed);
        ASSERT_TRUE(ready);
        ASSERT_TRUE(composed.ok()) << composed.describe_failure();
        const auto rendered =
            testing::to_host(composed.snapshot<holoscan::schema::ImageT>("output").tensor(
                holoscan::schema::ImageTensorFields::data));
        int max_difference = 0, bright = 0;
        for (int y = 0; y < font; ++y) {
          for (int x = 0; x < 12 * advance; ++x) {
            const auto* first = rendered.as<std::uint8_t>() +
                                4 * ((static_cast<int>(received_label->y) + y) * width +
                                     static_cast<int>(received_label->x) + x);
            const auto* second = rendered.as<std::uint8_t>() +
                                 4 * ((static_cast<int>(processed_label->y) + y) * width +
                                      static_cast<int>(processed_label->x) + x);
            bright += first[0] > 150;
            for (int c = 0; c < 3; ++c)
              max_difference =
                  std::max(max_difference, std::abs(static_cast<int>(first[c]) - second[c]));
          }
        }
        EXPECT_GT(bright, 20);
        EXPECT_LE(max_difference, 2) << "Matching label prefixes must render at the same size";
      }
    }
  }
}

TEST(SarFooterTiming, MatchesProcessingCostToImageCaptureInEventAndClockedPaths) {
  const holoscan::DeviceId device{holoscan::radar::app::application_device()};
  const auto image = testing::to_device_tensor(std::vector<std::complex<float>>(16, {1, 0}), {4, 4},
                                               DLDataType{kDLComplex, 64U, 1U});
  for (const bool clocked : {false, true}) {
    for (const int mode : {0, 1, 2}) {
      const bool matching = mode != 1;
      SCOPED_TRACE(mode);
      SCOPED_TRACE(clocked);
      SCOPED_TRACE(matching);
      sar::OverlayConfig layout;
      layout.canvas_width = 640;
      layout.image_height = 320;
      layout.footer_height = 112;
      layout.status_width = 320;
      layout.scene_names = {"Panama Canal"};
      layout.diag = [matching] {
        sar::FormationDiag diag;
        diag.capture_time_ns = matching ? 123000000 : 124000000;
        diag.collection_elapsed_s = 94.7;
        diag.bp_compute_s = 8.2;
        diag.budget_s = 2.4;
        diag.budget_playback_rate = 0.2;
        diag.pulses = 1000;
        diag.pixels = 315700000;
        return diag;
      };
      if (mode == 2) {
        auto hud = std::make_shared<holoscan::radar::FormationHud>();
        const auto expected = layout.diag();
        hud->publish_bp({.capture_time_ns = 123000000,
                         .compute_s = expected.bp_compute_s,
                         .budget_s = expected.budget_s,
                         .budget_playback_rate = expected.budget_playback_rate,
                         .pulses = expected.pulses,
                         .pixels = expected.pixels});
        // A later formation completes before the older mosaic reaches display.
        hud->publish_bp({.capture_time_ns = 124000000, .compute_s = 99, .pulses = 42});
        layout.diag_at_capture = [hud](int64_t stamp) {
          const auto bp = hud->snapshot(stamp).bp;
          sar::FormationDiag diag;
          diag.capture_time_ns = bp.capture_time_ns;
          diag.bp_compute_s = bp.compute_s;
          diag.budget_s = bp.budget_s;
          diag.budget_playback_rate = bp.budget_playback_rate;
          diag.pulses = bp.pulses;
          diag.pixels = bp.pixels;
          return diag;
        };
      }
      auto harness = holoscan::testing::create_operator_test<sar::SarImageVizOp>(
          4, 4, 320, false, layout, std::array<int64_t, 2>{0, 0}, 60.0, std::function<bool()>{},
          clocked);
      harness.push_frame(frame_at(std::chrono::milliseconds{123}, input("image", *image)));
      harness.input_source_device("image", device);
      harness.output_device("frame", device);
      const unsigned outputs = clocked ? 2U : 1U;
      harness.capture_snapshot<holoscan::schema::ImageT>("frame", outputs, 320 * 320 * 4);
      harness.capture_snapshot<holoscan::Tensor>(
          "overlay", outputs, sar::kHudMaxLines * sar::kHudLineFloats * sizeof(float));
      harness.capture_snapshot<holoscan::Tensor>("labels", outputs,
                                                 sar::kHudRuns * sar::kHudRowBytes);
      testing::FixtureCompletion completion;
      auto running =
          harness.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
      const bool placeholder = !clocked || completion.wait_for_output(running, "labels", 1U);
      const auto released = running.release_frame(completion.remaining());
      const bool ready = completion.wait_for_output(running, "labels", outputs) &&
                         completion.wait_for_output(running, "frame", outputs);
      running.request_stop();
      const auto result = running.finish();
      completion.check(result);
      ASSERT_TRUE(placeholder);
      ASSERT_TRUE(released);
      ASSERT_TRUE(ready);
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto records = testing::to_host(result.snapshots<holoscan::Tensor>("labels").back());
      holoscan::visualization::TextRunHeader title_header;
      std::memcpy(&title_header, records.as<std::uint8_t>(), sizeof(title_header));
      EXPECT_EQ(title_header.size_px, 18);
      EXPECT_EQ(title_header.color, sar::hud_rgba(75, 205, 220));
      EXPECT_FLOAT_EQ(title_header.x, 8.0F / layout.canvas_width);
      EXPECT_FLOAT_EQ(title_header.y,
                      (layout.image_height + 2.0F) / (layout.image_height + layout.footer_height));
      const auto* work = reinterpret_cast<const char*>(
          records.as<std::uint8_t>() + 2 * sar::kHudRowBytes + sar::kTextRunHeaderBytes);
      const auto* dimensions = reinterpret_cast<const char*>(
          records.as<std::uint8_t>() + sar::kHudRowBytes + sar::kTextRunHeaderBytes);
      EXPECT_STREQ(dimensions, matching ? "4x4 pixels  1,000 pulses" : "4x4 pixels  -- pulses");
      const auto* latency = reinterpret_cast<const char*>(
          records.as<std::uint8_t>() + 3 * sar::kHudRowBytes + sar::kTextRunHeaderBytes);
      const auto* throughput = reinterpret_cast<const char*>(
          records.as<std::uint8_t>() + 4 * sar::kHudRowBytes + sar::kTextRunHeaderBytes);
      EXPECT_STREQ(work, matching ? "BP work: 315.7 Gpx*pulses" : "BP work: -- Gpx*pulses");
      EXPECT_STREQ(latency, matching ? "Latency: 8.2s / 12.0s @0.2x" : "Latency: -- / -- @--");
      EXPECT_STREQ(throughput,
                   matching ? "Throughput: 38.5 Gpx*pulses/sec" : "Throughput: -- Gpx*pulses/sec");
      EXPECT_EQ(result.snapshots<holoscan::schema::ImageT>("frame")
                    .back()
                    .sample_metadata()
                    .capture_time()
                    .timestamp_ns,
                123000000);
    }
  }
}

TEST(SarFooterTiming, BoundsCompletedMeasurementHistoryAndNeverSubstitutesNewerWork) {
  holoscan::radar::FormationHud hud;
  for (int64_t stamp = 1; stamp <= 65; ++stamp)
    hud.publish_bp({.capture_time_ns = stamp, .pulses = stamp * 10});
  EXPECT_EQ(hud.snapshot().bp.capture_time_ns, 65);
  EXPECT_EQ(hud.snapshot(2).bp.pulses, 20);
  EXPECT_EQ(hud.snapshot(65).bp.pulses, 650);
  EXPECT_EQ(hud.snapshot(1).bp.pulses, 0);  // Evicted, never replaced with stamp 65.
  EXPECT_EQ(hud.snapshot(66).bp.pulses, 0);
}

TEST(SarFooterTiming, SeparatesMosaicExtentFromLastStripUpdateWorkload) {
  sar::OverlayConfig layout;
  layout.canvas_width = 1400;
  layout.image_height = 200;
  layout.footer_height = 112;
  layout.status_width = 400;
  layout.strip_update_metrics = true;
  sar::FormationDiag diag;
  diag.pulses = 256;
  diag.pixels = 1000000;
  diag.bp_compute_s = 0.5;
  std::vector<std::uint8_t> rows(sar::kHudRuns * sar::kHudRowBytes);
  sar::write_footer_hud_rows(rows.data(), {"Imperial Valley  40247x6029"}, diag, false, layout);
  const auto text = [&](int index) {
    return reinterpret_cast<const char*>(rows.data() + index * sar::kHudRowBytes +
                                         sar::kTextRunHeaderBytes);
  };
  EXPECT_STREQ(text(1), "40247x6029 pixels (mosaic)");
  EXPECT_STREQ(text(2), "BP work: 0.3 Gpx*pulses");
  EXPECT_STREQ(text(5), "Last strip update: 256 pulses");
  for (std::size_t i = 0; i < sar::kHudRuns; ++i) {
    holoscan::visualization::TextRunHeader header;
    std::memcpy(&header, rows.data() + i * sar::kHudRowBytes, sizeof(header));
    EXPECT_LE(header.y * (layout.image_height + layout.footer_height) + header.size_px,
              layout.image_height + layout.footer_height);
  }
}

TEST(SarTimeline, ShowsCheckpointProgressAtWideAndCompactSizes) {
  const std::array<std::array<int, 2>, 5> frontiers{
      {{0, 0}, {20, 40}, {40, 40}, {0, 100}, {100, 100}}};
  for (const auto& counts : frontiers) {
    const auto formed_count = counts[0], received_count = counts[1];
    SCOPED_TRACE(formed_count);
    SCOPED_TRACE(received_count);
    for (const int width : {320, 704}) {
      SCOPED_TRACE(width);
      constexpr int height = 112;
      auto hud = std::make_shared<holoscan::radar::FormationHud>();
      // Full receipt does not prove a processing boundary was reached. Until
      // a real stage notification arrives, autofocus remains at receiving.
      if (formed_count == 0 && received_count == 100)
        hud->enable_autofocus();
      hud->enable_progress(0, 40);
      auto monitor = hud->progress();
      monitor->begin_packet(1, 1, 0, std::max(1, received_count), 100, 99, 0.1, true);
      if (received_count > 0)
        monitor->received(1);
      if (formed_count > 0) {
        monitor->admit(1, formed_count, formed_count == 100, true);
        auto work = monitor->take_work(1);
        ASSERT_TRUE(work);
        monitor->submitted();
        work->pixels = 10;
        monitor->set_probe([work] {
          return holoscan::radar::sar_progress::GpuState{
              .completed = work,
              .pixel_pulses_per_s = 100,
              .measured_pixels = 10,
              .completed_at = std::chrono::steady_clock::now()};
        });
      }
      auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
          hud, std::vector<std::string>{"Displayed image", "Bingham Copper Mine"}, false, 10.0,
          width, height);
      harness.output_device("frame",
                            holoscan::DeviceId{holoscan::radar::app::application_device()});
      harness.capture_snapshot<holoscan::schema::ImageT>("frame", 1U, width * height * 4);
      harness.capture_snapshot<holoscan::Tensor>("labels", 1U, 65536U);
      testing::FixtureCompletion completion;
      auto running = harness.run_async(completion.run_options());
      const bool ready = completion.wait_for_output(running, "frame", 1U) &&
                         completion.wait_for_output(running, "labels", 1U);
      running.request_stop();
      const auto result = running.finish();
      completion.check(result);
      ASSERT_TRUE(ready);
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto frame = result.snapshot<holoscan::schema::ImageT>("frame");
      const auto pixels = testing::to_host(frame.tensor(holoscan::schema::ImageTensorFields::data));
      EXPECT_EQ(pixels.shape, (std::vector<int64_t>{height, width, 4}));
      const auto records = testing::to_host(result.snapshot<holoscan::Tensor>("labels"));
      bool received = false, formed = false, stage = false, update_hint = false, next = false;
      bool workload = false;
      float stage_size = 0, count_size = 0, stage_y = 0, title_y = 0, title_size = 0;
      float completed_right = 0, receipt_x = 0;
      for (int64_t row = 0; row < records.shape[0]; ++row) {
        const auto* record = records.as<std::uint8_t>() + row * records.shape[1];
        const auto* text = reinterpret_cast<const char*>(record + sar::kTextRunHeaderBytes);
        received = received || text == "Received " + std::to_string(received_count) + "%";
        formed = formed || text == "Last completed " + std::to_string(formed_count) + "%";
        const bool image_title = std::strncmp(text, "Bingham", 7) == 0;
        next = next || image_title;
        holoscan::visualization::TextRunHeader header;
        std::memcpy(&header, record, sizeof(header));
        if (std::strncmp(text, "Last completed ", 15) == 0)
          completed_right = header.x + std::strlen(text) * header.size_px * 0.6F;
        if (std::strncmp(text, "Received ", 9) == 0)
          receipt_x = header.x;
        EXPECT_NE(std::strncmp(text, "Next image", 10), 0);
        EXPECT_NE(std::strncmp(text, "ETA ", 4), 0);
        EXPECT_NE(std::strncmp(text, "Total time ", 11), 0);
        if (std::strcmp(text, "Forming image") == 0 || std::strcmp(text, "Receiving pulses") == 0 ||
            std::strcmp(text, "Image ready") == 0) {
          stage = true;
          stage_size = header.size_px;
          stage_y = header.y;
          EXPECT_STREQ(text, formed_count == 100                          ? "Image ready"
                             : formed_count == 0 && received_count == 100 ? "Receiving pulses"
                             : received_count > 0                         ? "Forming image"
                                                                          : "Receiving pulses");
        }
        if (image_title) {
          title_y = header.y;
          title_size = header.size_px;
        }
        EXPECT_NE(std::strncmp(text, "Elapsed ", 8), 0);
        if (std::strcmp(text, "Updates on completion") == 0 ||
            std::strcmp(text, "Last reported stage") == 0)
          update_hint = true;
        if (std::strncmp(text, "BP work: ", 9) == 0) {
          workload = true;
          EXPECT_EQ(header.size_px, 14);
          EXPECT_STREQ(text,
                       formed_count > 0 ? "BP work: 0.0 Gpx*pulses" : "BP work: -- Gpx*pulses");
        }
        if (std::strncmp(text, "Received ", 9) == 0)
          count_size = header.size_px;
      }
      EXPECT_TRUE(received);
      EXPECT_TRUE(formed);
      EXPECT_TRUE(stage);
      if (!(width < 620 && formed_count == 0 && received_count == 100))
        EXPECT_TRUE(update_hint);
      EXPECT_GE(receipt_x, completed_right + 6);
      EXPECT_TRUE(workload);
      EXPECT_TRUE(next);
      EXPECT_GE(title_y, stage_y + stage_size);
      EXPECT_EQ(title_size, 14);
      if (width >= 620)
        EXPECT_GT(stage_size, count_size);
      const int bar_width = width < 620 ? width - 16 : 248;
      const bool compact_autofocus = width < 620 && formed_count == 0 && received_count == 100;
      const int bar_y = width < 620 ? (compact_autofocus ? 53 : 46) : 82;
      const int bar_height = width < 620 ? (compact_autofocus ? 3 : 4) : 6;
      const int filled = bar_width * formed_count / 100;
      const int marker = std::min(bar_width * received_count / 100, bar_width - 2);
      int cyan = 0, green = 0;
      for (int64_t pixel = 0; pixel < width * height; ++pixel) {
        if ((width < 620 && pixel / width >= 56) || (width >= 620 && pixel % width >= 264))
          continue;
        const auto* rgba = pixels.as<std::uint8_t>() + 4 * pixel;
        if (rgba[0] == 75 && rgba[1] == 205 && rgba[2] == 220) {
          ++cyan;
          EXPECT_GE(pixel % width, 8 + marker);
          EXPECT_LT(pixel % width, 10 + marker);
          EXPECT_GE(pixel / width, bar_y - 2);
          EXPECT_LT(pixel / width, bar_y + bar_height + 2);
        }
        green += rgba[0] == 115 && rgba[1] == 200 && rgba[2] == 150;
      }
      EXPECT_EQ(cyan, 2 * (bar_height + (compact_autofocus ? 2 : 4)))
          << "Received is a visible marker on the same track";
      const int overlap = std::max(0, std::min(filled, marker + 2) - marker);
      EXPECT_EQ(green, (filled - overlap) * bar_height)
          << "Fill tracks GPU completion; marker remains visible when both frontiers meet";
    }
  }
}

TEST(SarFooterTiming, BudgetReflectsPlaybackRateAndUnpacedHasNoFixedDeadline) {
  EXPECT_EQ(sar::format_playback_budget(2.4, 0.2), "12.0s @0.2x");
  EXPECT_EQ(sar::format_playback_budget(2.4, 1.), "2.4s @1x");
  EXPECT_EQ(sar::format_playback_budget(2.4, 2.), "1.2s @2x");
  EXPECT_EQ(sar::format_playback_budget(0.01, 2.), "5ms @2x");
  EXPECT_EQ(sar::format_playback_budget(2.4, 0.), "-- unpaced");
  EXPECT_EQ(sar::format_playback_budget(std::nullopt, 0.2), "-- @0.2x");
  EXPECT_EQ(sar::format_playback_budget(2.4, std::nullopt), "-- @--");
  EXPECT_EQ(sar::format_playback_budget(2.4, -1.), "-- @--");
  EXPECT_EQ(sar::format_playback_budget(2.4, std::numeric_limits<double>::quiet_NaN()), "-- @--");
  EXPECT_EQ(sar::format_playback_budget(2.4, std::numeric_limits<double>::infinity()), "-- @--");
  EXPECT_EQ(sar::format_playback_budget(-2.4, 0.2), "-- @0.2x");
  EXPECT_EQ(sar::format_playback_budget(std::numeric_limits<double>::infinity(), 0.2), "-- @0.2x");
  EXPECT_EQ(sar::format_playback_budget(1., 1e-10), "1.00e10s @1e-10x");
  EXPECT_EQ(sar::format_playback_budget(std::numeric_limits<double>::max(), 0.2), "-- @0.2x");
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
  EXPECT_EQ(sar::format_bp_giga_throughput(2, 1000, 1000000), "0.5");
  EXPECT_EQ(sar::format_bp_giga_throughput(0, 1000, 1000000), "--");
}
}  // namespace

TEST(SarAutofocusStatus, UsesTerminationAndAcceptedIterationsRatherThanSharpnessGain) {
  holoscan::radar::FormationHud hud;
  using Reason = holoscan::radar::AutofocusTermination;
  hud.enable_autofocus();
  auto text = [&](bool ready, bool received = true) {
    return sar::format_autofocus_status(hud.snapshot().autofocus, ready, received);
  };
  EXPECT_EQ(text(false, false), "Preparing autofocus");
  EXPECT_EQ(text(false), "Autofocus running");
  for (const auto reason :
       {Reason::kRefusedSharpness, Reason::kRefusedScatterers, Reason::kRefusedMaxCorrection}) {
    hud.publish_autofocus(static_cast<int32_t>(reason), 0);
    EXPECT_EQ(text(true), "Correction declined");
    EXPECT_EQ(text(false), "AF declined; forming image");
  }
  for (const auto reason :
       {Reason::kConverged, Reason::kBudgetExhausted, Reason::kStoppedScatterers}) {
    hud.publish_autofocus(static_cast<int32_t>(reason), 2);
    EXPECT_EQ(text(true), "Correction accepted");
    EXPECT_EQ(text(false), "AF accepted; forming image");
  }
  hud.publish_autofocus(static_cast<int32_t>(Reason::kBudgetExhausted), 0);
  EXPECT_EQ(text(true), "No correction applied");
  hud.publish_autofocus({}, 0);
  EXPECT_EQ(text(true), "AF status unavailable");
  hud.publish_autofocus(999, 2);
  EXPECT_EQ(text(true), "AF status unavailable");
  hud.enable_autofocus(true);
  EXPECT_EQ(text(true), "Oracle correction");
}

TEST(SarAutofocusStatus, NewCollectionClearsOldOutcomeAndRejectsLateReports) {
  holoscan::radar::FormationHud hud;
  using Outcome = holoscan::radar::AutofocusHudOutcome;
  using Reason = holoscan::radar::AutofocusTermination;
  hud.enable_autofocus();
  const auto now = std::chrono::steady_clock::now();
  hud.publish_reader({{.dwell_start = now}}, {}, {}, {}, false, 100);
  hud.publish_autofocus(static_cast<int32_t>(Reason::kConverged), 2, 110);
  EXPECT_EQ(hud.snapshot().autofocus.outcome, Outcome::kAccepted);
  hud.publish_reader({{.dwell_start = now + std::chrono::seconds(1)}}, {}, {}, {}, false, 200);
  EXPECT_EQ(hud.snapshot().autofocus.outcome, Outcome::kPending);
  hud.publish_autofocus(static_cast<int32_t>(Reason::kConverged), 2, 110);
  EXPECT_EQ(hud.snapshot().autofocus.outcome, Outcome::kPending);
  hud.publish_autofocus(static_cast<int32_t>(Reason::kRefusedSharpness), 0, 210);
  EXPECT_EQ(hud.snapshot().autofocus.outcome, Outcome::kDeclined);
}

TEST(SarAutofocusStatus, TimelineShowsOutcomeAndDeclineWarningInBothLayouts) {
  using Reason = holoscan::radar::AutofocusTermination;
  for (const int width : {320, 704}) {
    for (const bool declined : {false, true}) {
      auto hud = std::make_shared<holoscan::radar::FormationHud>();
      hud->enable_progress(0, 20);
      hud->enable_autofocus();
      hud->publish_autofocus(
          static_cast<int32_t>(declined ? Reason::kRefusedSharpness : Reason::kConverged),
          declined ? 0 : 2);
      const auto monitor = hud->progress();
      monitor->begin_packet(100, 0, 0, 20, 20, 1, 1, true);
      monitor->admit(100, 20, true, true);
      const auto work = monitor->take_work(100);
      monitor->submitted();
      monitor->set_probe([work] {
        return holoscan::radar::sar_progress::GpuState{.completed = work};
      });
      auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
          hud, std::vector<std::string>{"Kourou"}, false, 10.0, width, 112);
      harness.output_device("frame",
                            holoscan::DeviceId{holoscan::radar::app::application_device()});
      harness.capture_snapshot<holoscan::schema::ImageT>("frame", 1U, width * 112 * 4);
      harness.capture_snapshot<holoscan::Tensor>("labels", 1U, 65536U);
      holoscan::radar::testing::FixtureCompletion completion;
      auto running = harness.run_async(completion.run_options());
      const bool ready = completion.wait_for_output(running, "labels", 1U);
      running.request_stop();
      const auto result = running.finish();
      completion.check(result);
      ASSERT_TRUE(ready);
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto records =
          holoscan::radar::testing::to_host(result.snapshot<holoscan::Tensor>("labels"));
      bool found = false;
      for (int64_t row = 0; row < records.shape[0]; ++row) {
        const auto* record = records.as<std::uint8_t>() + row * records.shape[1];
        const auto* text = reinterpret_cast<const char*>(record + sar::kTextRunHeaderBytes);
        if (std::strcmp(text, declined ? "Correction declined" : "Correction accepted") != 0)
          continue;
        found = true;
        holoscan::visualization::TextRunHeader header;
        std::memcpy(&header, record, sizeof(header));
        if (declined)
          EXPECT_EQ(header.color, sar::hud_rgba(255, 180, 60));
      }
      EXPECT_TRUE(found);
    }
  }
}

namespace {
TEST(SarAutofocusStages, RetainsOrderingOutcomeAndCollectionCoherence) {
  using Stage = holoscan::radar::AutofocusHudStage;
  using Reason = holoscan::radar::AutofocusTermination;
  holoscan::radar::FormationHud hud;
  hud.publish_autofocus_stage(Stage::kEstimating, 110);
  EXPECT_TRUE(sar::format_autofocus_stage(hud.snapshot().autofocus).empty());
  hud.enable_autofocus();
  const auto now = std::chrono::steady_clock::now();
  hud.publish_reader({{.dwell_start = now}}, {}, {}, {}, false, 100);
  EXPECT_EQ(sar::format_autofocus_stage(hud.snapshot().autofocus), "Receiving pulses");
  hud.publish_autofocus_stage(Stage::kFormingAuxiliary, 110);
  EXPECT_EQ(sar::format_autofocus_stage(hud.snapshot().autofocus), "Forming autofocus image");
  hud.publish_autofocus_stage(Stage::kEstimating, 110);
  hud.publish_autofocus_stage(Stage::kFormingAuxiliary, 110);
  EXPECT_EQ(sar::format_autofocus_stage(hud.snapshot().autofocus), "Estimating correction");
  hud.publish_autofocus(static_cast<int32_t>(Reason::kRefusedSharpness), 0, 110);
  EXPECT_EQ(hud.snapshot().autofocus.stage, Stage::kEstimating);
  hud.publish_autofocus_stage(Stage::kFormingFinal, 110);
  hud.publish_bp({.capture_time_ns = 110, .emit_at = now});
  EXPECT_EQ(sar::format_autofocus_stage(hud.snapshot().autofocus), "Image ready");
  hud.publish_autofocus(static_cast<int32_t>(Reason::kRefusedSharpness), 0, 110);
  EXPECT_EQ(hud.snapshot().autofocus.stage, Stage::kReady);
  EXPECT_EQ(sar::format_autofocus_status(hud.snapshot().autofocus, true, true),
            "Correction declined");
  hud.publish_reader({{.dwell_start = now + std::chrono::seconds(1)}}, {}, {}, {}, false, 200);
  hud.publish_autofocus_stage(Stage::kReady, 110);
  EXPECT_EQ(hud.snapshot().autofocus.stage, Stage::kReceiving);
  EXPECT_EQ(hud.snapshot().autofocus.stage_capture_ns, 200);
}

TEST(SarAutofocusStages, ForwardsTensorContentsAndCaptureStamp) {
  auto hud = std::make_shared<holoscan::radar::FormationHud>();
  hud->enable_autofocus();
  const auto tensor = testing::to_device_tensor(std::vector<float>{1, 2, 3, 4}, {2, 2},
                                                DLDataType{kDLFloat, 32U, 1U});
  auto harness = holoscan::testing::create_operator_test<sar::AutofocusStageOp>(
      hud, holoscan::radar::AutofocusHudStage::kEstimating);
  harness.push_frame(frame_at(std::chrono::milliseconds{123}, input("input", *tensor)));
  const holoscan::DeviceId device{holoscan::radar::app::application_device()};
  harness.input_source_device("input", device);
  harness.output_device("output", device);
  harness.capture_snapshot<holoscan::Tensor>("output", 1U, 4U * sizeof(float));
  testing::FixtureCompletion completion;
  auto running = harness.run_async(completion.run_options());
  const bool ready = completion.wait_for_output(running, "output", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(ready);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto samples = result.snapshots<holoscan::Tensor>("output");
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_EQ(samples[0].sample_metadata().capture_time().timestamp_ns, 123000000);
  EXPECT_EQ(hud->snapshot().autofocus.stage_capture_ns, 123000000);
  const auto values = testing::to_host(samples[0]);
  EXPECT_FLOAT_EQ(values.as<float>()[3], 4);
}

TEST(SarAutofocusStages, FullStageNamesFitAndOutcomeIsASeparateRow) {
  using Stage = holoscan::radar::AutofocusHudStage;
  for (const auto stage : {Stage::kReceiving, Stage::kFormingAuxiliary, Stage::kEstimating,
                           Stage::kFormingFinal, Stage::kReady}) {
    for (const int width : {320, 704}) {
      auto hud = std::make_shared<holoscan::radar::FormationHud>();
      hud->enable_autofocus();
      hud->enable_progress(0, 10);
      hud->publish_autofocus_stage(stage, 123);
      hud->publish_autofocus(
          static_cast<int32_t>(holoscan::radar::AutofocusTermination::kConverged), 1, 123);
      const auto expected = sar::format_autofocus_stage(hud->snapshot().autofocus);
      auto harness = holoscan::testing::create_operator_test<sar::TimelineRasterOp>(
          hud, std::vector<std::string>{"Kourou"}, false, 10.0, width, 112);
      harness.output_device("frame",
                            holoscan::DeviceId{holoscan::radar::app::application_device()});
      harness.capture_snapshot<holoscan::schema::ImageT>("frame", 1U, width * 112 * 4);
      harness.capture_snapshot<holoscan::Tensor>("labels", 1U, 65536U);
      testing::FixtureCompletion completion;
      auto running = harness.run_async(completion.run_options());
      const bool ready = completion.wait_for_output(running, "labels", 1U);
      running.request_stop();
      const auto result = running.finish();
      completion.check(result);
      ASSERT_TRUE(ready);
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto records = testing::to_host(result.snapshot<holoscan::Tensor>("labels"));
      bool found_stage = false, found_outcome = false;
      float stage_y = 0, outcome_y = 0;
      for (int64_t i = 0; i < records.shape[0]; ++i) {
        const auto* record = records.as<std::uint8_t>() + i * records.shape[1];
        const auto* text = reinterpret_cast<const char*>(record + sar::kTextRunHeaderBytes);
        holoscan::visualization::TextRunHeader header;
        std::memcpy(&header, record, sizeof(header));
        if (text == expected) {
          found_stage = true;
          stage_y = header.y;
        }
        if (std::strcmp(text, "Correction accepted") == 0) {
          found_outcome = true;
          outcome_y = header.y;
        }
      }
      EXPECT_TRUE(found_stage) << expected << " at width " << width;
      EXPECT_TRUE(found_outcome);
      EXPECT_GT(outcome_y, stage_y);
    }
  }
}

}  // namespace

namespace {
TEST(SarSensorLook, FitsEdgesAndPointsIntoSceneWithPixelSizedGeometry) {
  for (const auto size : {std::array{640, 256}, std::array{256, 640}, std::array{1024, 1024}}) {
    for (const auto direction : {std::array{1., 0.}, std::array{-1., 0.}, std::array{0., 1.},
                                 std::array{0., -1.}, std::array{1., 1.}, std::array{-1., -1.}}) {
      for (const bool transpose : {false, true}) {
        sar::FormationDiag diag;
        diag.apc_valid = true;
        diag.raster_u = direction[0];
        diag.raster_v = direction[1];
        const auto look = sar::sensor_look_annotation(diag, transpose, size[0], size[1]);
        ASSERT_TRUE(look);
        EXPECT_GE(look->label_x, 0);
        EXPECT_GE(look->label_y, 0);
        EXPECT_LE(look->label_x + look->label_width, size[0]);
        EXPECT_LE(look->label_y + look->label_height, size[1]);
        const auto& shaft = look->segments[2];
        const float dx = shaft[2] - shaft[0], dy = shaft[3] - shaft[1];
        EXPECT_LE(std::hypot(dx, dy), 54.001);
        EXPECT_GT(dx * (size[0] / 2 - shaft[0]) + dy * (size[1] / 2 - shaft[1]), 0);
        if (std::abs(dx) < 0.001F) {
          if (dy > 0)
            EXPECT_GT(look->label_y, shaft[3] + 8);
          else
            EXPECT_LT(look->label_y + look->label_height, shaft[3] - 8);
        }
        std::vector<std::uint8_t> rows(sar::kHudRuns * sar::kHudRowBytes);
        sar::write_sensor_look_row(rows.data(), *look, size[0], size[1] + 112);
        holoscan::visualization::TextRunHeader header;
        std::memcpy(&header, rows.data() + 6U * sar::kHudRowBytes, sizeof(header));
        EXPECT_EQ(header.size_px, 14);
        EXPECT_LT(header.y + 14.0F / (size[1] + 112),
                  static_cast<float>(size[1]) / (size[1] + 112));
        EXPECT_STREQ(reinterpret_cast<const char*>(rows.data() + 6U * sar::kHudRowBytes +
                                                   sar::kTextRunHeaderBytes),
                     "Sensor look angle");
      }
    }
  }
  sar::FormationDiag diag;
  EXPECT_FALSE(sar::sensor_look_annotation(diag, false, 640, 256));
  diag.apc_valid = true;
  EXPECT_FALSE(sar::sensor_look_annotation(diag, false, 640, 256));
  diag.raster_u = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(sar::sensor_look_annotation(diag, false, 640, 256));
  diag.raster_u = 1;
  EXPECT_FALSE(sar::sensor_look_annotation(diag, false, 160, 80));
}

TEST(SarSensorLook, PaintsOnlySmallDisplayPatchAndPreservesCaptureStamp) {
  constexpr int side = 256;
  const auto image = testing::to_device_tensor(std::vector<std::complex<float>>(16, {1, 0}), {4, 4},
                                               DLDataType{kDLComplex, 64U, 1U});
  sar::FormationDiag diag;
  diag.apc_valid = true;
  diag.raster_u = 1;
  sar::OverlayConfig layout;
  layout.canvas_width = side;
  layout.image_height = side;
  layout.diag = [diag] {
    return diag;
  };
  const auto look = sar::sensor_look_annotation(diag, false, side, side);
  ASSERT_TRUE(look);
  auto harness =
      holoscan::testing::create_operator_test<sar::SarImageVizOp>(4, 4, side, false, layout);
  harness.push_frame(frame_at(std::chrono::milliseconds{123}, input("image", *image)));
  const holoscan::DeviceId device{holoscan::radar::app::application_device()};
  harness.input_source_device("image", device);
  harness.output_device("frame", device);
  harness.capture_snapshot<holoscan::schema::ImageT>("frame", 1U, side * side * 4U);
  harness.capture_snapshot<holoscan::Tensor>("labels", 1U, sar::kHudRuns * sar::kHudRowBytes);
  harness.capture_snapshot<holoscan::Tensor>(
      "overlay", 1U, sar::kHudMaxLines * sar::kHudLineFloats * sizeof(float));
  testing::FixtureCompletion completion;
  auto running = harness.run_async(completion.run_options());
  const bool ready = completion.wait_for_output(running, "frame", 1U) &&
                     completion.wait_for_output(running, "labels", 1U) &&
                     completion.wait_for_output(running, "overlay", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(ready) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frame = result.snapshot<holoscan::schema::ImageT>("frame");
  EXPECT_EQ(frame.sample_metadata().capture_time().timestamp_ns, 123000000);
  const auto pixels = testing::to_host(frame.tensor(holoscan::schema::ImageTensorFields::data));
  int cyan = 0, dark = 0;
  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      const auto* pixel = pixels.as<std::uint8_t>() + (y * side + x) * 4;
      if (x < std::floor(look->left) || x >= std::ceil(look->right) || y < std::floor(look->top) ||
          y >= std::ceil(look->bottom)) {
        EXPECT_EQ(pixel[0], 255);
        EXPECT_EQ(pixel[1], 255);
        EXPECT_EQ(pixel[2], 255);
      }
      cyan += pixel[2] > pixel[0] && pixel[1] > pixel[0];
      dark += pixel[0] < 50;
    }
  }
  EXPECT_GT(cyan, 20);
  EXPECT_GT(dark, 100);
  const auto rows = testing::to_host(result.snapshot<holoscan::Tensor>("labels"));
  EXPECT_STREQ(reinterpret_cast<const char*>(rows.as<std::uint8_t>() + 6U * sar::kHudRowBytes +
                                             sar::kTextRunHeaderBytes),
               "Sensor look angle");
}
}  // namespace

namespace {
TEST(SarStartup, FlushesNewestRateDeferredImageWithoutAnotherInputOrReaderEof) {
  for (const int mode : {0, 1, 2}) {
    const bool clocked = mode != 2;
    SCOPED_TRACE(mode);
    const auto eof = std::make_shared<std::atomic<bool>>(false);
    const auto image = testing::to_device_tensor(std::vector<std::complex<float>>(16, {1, 0}),
                                                 {4, 4}, DLDataType{kDLComplex, 64U, 1U});
    sar::OverlayConfig layout;
    layout.scene_names = {"First", "Second", "Third"};
    auto harness = holoscan::testing::create_operator_test<sar::SarImageVizOp>(
        4, 4, 4, false, layout, std::array<int64_t, 2>{0, 0}, 1.0,
        [eof] {
          return eof->load();
        },
        clocked);
    const auto stamps = mode == 1 ? std::vector{123, 125} : std::vector{123, 124, 125};
    for (const int stamp : stamps)
      harness.push_frame(frame_at(std::chrono::milliseconds{stamp}, input("image", *image)));
    // Keep the finite fixture source alive without delivering any more data.
    // Releasing its final staged frame would stop the whole fixture graph,
    // unlike the application driver's explicit EOF display-drain phase.
    harness.push_frame(frame_at(std::chrono::seconds{99}, input("image", *image)));
    const holoscan::DeviceId device{holoscan::radar::app::application_device()};
    harness.input_source_device("image", device);
    harness.output_device("frame", device);
    harness.capture_snapshot<holoscan::schema::ImageT>("frame", 4U, 64U);
    harness.capture_snapshot<holoscan::Tensor>("labels", 4U, sar::kHudRuns * sar::kHudRowBytes);
    harness.capture_snapshot<holoscan::Tensor>("overlay", 4U, 4U * sizeof(float));
    testing::FixtureCompletion completion{std::chrono::seconds{5}};
    auto running =
        harness.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
    if (clocked)
      ASSERT_TRUE(completion.wait_for_output(running, "frame", 1U));
    ASSERT_TRUE(running.release_frame(completion.remaining()));
    const unsigned first = clocked ? 2U : 1U;
    ASSERT_TRUE(completion.wait_for_output(running, "frame", first));
    ASSERT_TRUE(completion.wait_for_output(running, "labels", first));
    ASSERT_TRUE(running.release_frame(completion.remaining()));
    if (clocked)
      EXPECT_FALSE(running.wait_for_output("frame", first + 1U, std::chrono::milliseconds{150}));
    if (mode != 1)
      ASSERT_TRUE(running.release_frame(completion.remaining()));
    if (mode == 1)
      eof->store(true);  // EOF may follow the rate-deferred input on a later tick.
    const unsigned expected = 3U;
    const bool ready = completion.wait_for_output(running, "frame", expected) &&
                       completion.wait_for_output(running, "labels", expected) &&
                       completion.wait_for_output(running, "overlay", expected);
    const bool repeated =
        running.wait_for_output("frame", expected + 1U, std::chrono::milliseconds{150});
    running.request_stop();
    const auto result = running.finish();
    ASSERT_TRUE(ready) << "mode=" << mode << " " << result.describe_failure();
    completion.check(result);
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_FALSE(repeated);
    const auto frames = result.snapshots<holoscan::schema::ImageT>("frame");
    const auto labels = result.snapshots<holoscan::Tensor>("labels");
    ASSERT_EQ(frames.size(), expected);
    ASSERT_EQ(labels.size(), expected);
    EXPECT_EQ(frames[first - 1U].sample_metadata().capture_time().timestamp_ns, 123000000);
    EXPECT_EQ(frames.back().sample_metadata().capture_time().timestamp_ns, 125000000);
    EXPECT_EQ(labels.back().sample_metadata().capture_time().timestamp_ns, 125000000);
    const auto text = testing::to_host(labels.back());
    EXPECT_STREQ(reinterpret_cast<const char*>(text.as<std::uint8_t>() + sar::kTextRunHeaderBytes),
                 mode == 1 ? "Second  4x4" : "Third  4x4");
    if (!clocked)
      EXPECT_EQ(frames[1].sample_metadata().capture_time().timestamp_ns, 124000000);
    if (mode == 0)
      EXPECT_FALSE(eof->load());
  }
}
}  // namespace
