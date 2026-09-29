// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

#include "utils/perf_yaml_pilot.hpp"

namespace holoscan::radar::perf {

inline std::array<int64_t, 2> sparse_window(const YAML::Node& node) {
  const auto window = recipe_pair<int64_t>(node, "snapshot_window");
  for (const auto extent : window)
    if (extent < 0 || extent > 2)
      throw std::invalid_argument("sparse MIMO snapshot window must be between 0 and 2");
  return window;
}

inline int64_t sparse_window_area(const std::array<int64_t, 2>& window) {
  return (2 * window[0] + 1) * (2 * window[1] + 1);
}

inline MaxTensorExtents sparse_mimo_extents(const RecipeScenario& scenario) {
  MaxTensorExtents extents;
  extents.max_tx = scenario.axis("tx");
  extents.max_rx = scenario.axis("rx");
  extents.max_virtual_channels = scenario.axis("tx") * scenario.axis("rx");
  extents.max_doppler_bins = scenario.axis("doppler");
  extents.max_range_bins = scenario.axis("range");
  extents.max_detections = scenario.axis("detections");
  const auto baseline = sparse_window(scenario.params["snapshot_window"]);
  const auto changed = scenario.live_port == "snapshot_window" ? sparse_window(scenario.live_value)
                                                               : std::array<int64_t, 2>{0, 0};
  extents.max_snapshots = std::max(sparse_window_area(baseline), sparse_window_area(changed));
  return extents;
}

}  // namespace holoscan::radar::perf
