// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoscan::radar::app {

// App-owned run cadence and visualization settings shared by FMCW demos.
// Kept YAML-free so HSDK 5 config::Document applications do not acquire a
// yaml-cpp dependency through their runtime configuration type.
struct CommonRunConfig {
  int64_t num_frames{1};
  double frame_period{0.1};
  bool meter_cadence{true};
  int64_t meter_log_every{0};
  bool visualize{false};
  bool skip_display_check{false};
  bool headless{false};

  struct VisualizationConfig {
    float marker_radius{0.004F};
    float marker_outline{0.002F};
    int64_t marker_segments{20};
    bool vsync{true};
    int64_t window_width{1280};
    int64_t window_height{720};
  } viz;

  bool peak_only{false};

  struct PlotsConfig {
    bool enable{true};
    int64_t window{120};
  } plots;
};

}  // namespace holoscan::radar::app
