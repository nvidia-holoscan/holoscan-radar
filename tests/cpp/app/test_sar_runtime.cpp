// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>

#include <holoscan/core/graph.hpp>

#include "radar_app/sar/graph_driver.hpp"
#include "applications/sar/spotlight/cpp/config.hpp"

namespace {

TEST(SarRuntime, PreparedPlacementSurvivesInterleavedDeviceSelection) {
  const std::string path = HOLOSCAN_RADAR_SPOTLIGHT_FIXTURE;
  auto cfg = load_spotlight_config(path, {});
  cfg.visualize = false;
  auto prepared = prepare_sar_runtime(cfg, path);
  const auto compute = prepared.devices.compute;
  // Distinct logical renderer IDs exercise placement on a single-GPU test
  // host. This test authors the graph; it never executes these fake devices.
  prepared.devices.visualize = true;
  prepared.devices.display = holoscan::DeviceId{41};
  prepared.devices.composer = holoscan::DeviceId{42};
  (void)prepare_sar_runtime(cfg, path);
  for (const std::string placement : {"bottom", "window"}) {
    for (bool headless : {true, false}) {
      cfg.headless = headless;
      cfg.visualization.headless = headless;
      cfg.visualization.x11_presentation = false;
      cfg.visualization.timeline.enabled = true;
      cfg.visualization.timeline.placement = placement;
      SarGraphRuntime<float, double> runtime("placement", CliPaths{path, {}}, cfg, nullptr,
                                             &prepared);
      EXPECT_EQ(runtime.devices.display.value, 41);
      EXPECT_EQ(runtime.devices.composer.value, 42);
      runtime.finish(&runtime.bp->image);
      bool reader_bound = false, composer_bound = false, timeline_bound = false;
      bool display_bound = false, footer_display_bound = false, replay_bound = false;
      for (const auto& binding : runtime.options.deployment.tensor_output_devices()) {
        if (binding.operator_path == "cphd_reader") {
          EXPECT_EQ(binding.device.value, compute.value);
          reader_bound = true;
        }
        if (binding.operator_path == "sar_composer_upload") {
          EXPECT_EQ(binding.device.value, 42);
          composer_bound = true;
        }
        if (binding.operator_path == "timeline_composer_upload") {
          EXPECT_EQ(binding.device.value, 42);
          timeline_bound = true;
        }
        if (binding.operator_path == "holoviz" || binding.operator_path == "timeline_frame")
          EXPECT_EQ(binding.device.value, 42);
        if (binding.operator_path == "sar_dock" || binding.operator_path == "sar_replay") {
          EXPECT_EQ(binding.device.value, 42);
          replay_bound = true;
        }
        if (binding.operator_path == "sar_display_upload") {
          EXPECT_EQ(binding.device.value, 41);
          display_bound = true;
        }
        if (binding.operator_path == "timeline_display_upload") {
          EXPECT_EQ(binding.device.value, 41);
          footer_display_bound = true;
        }
      }
      EXPECT_TRUE(reader_bound);
      EXPECT_TRUE(composer_bound);
      EXPECT_TRUE(timeline_bound);
      EXPECT_EQ(replay_bound, !headless);
      EXPECT_EQ(display_bound, !headless);
      EXPECT_EQ(footer_display_bound, !headless && placement == "window");
    }
  }
}

}  // namespace
