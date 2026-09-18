// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <tuple>

#include <holoscan/params/config.hpp>

#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar::app {

struct VisualizationConfig {
  bool enabled{true};
  int64_t window_width{1280};
  int64_t window_height{720};
  std::string window_title{"FMCW Radar"};
  double display_rate_hz{60.0};
  double dynamic_range_db{60.0};
  double detection_radius{0.010};
  int64_t history_frames{120};
  int64_t detection_trail_frames{12};
  std::string profile;
};

}  // namespace holoscan::radar::app

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::VisualizationConfig>
    : radar::config_detail::PortableParamStorageTraits<
          radar::app::VisualizationConfig> {};  // NOLINT

}  // namespace holoscan

namespace holoscan::config {

template <>
struct ConfigCodec<radar::app::VisualizationConfig>
    : ObjectCodec<radar::app::VisualizationConfig> {  // NOLINT
  using Config = radar::app::VisualizationConfig;
  static constexpr auto fields = std::make_tuple(
      defaulted_object_field("enabled", &Config::enabled),
      defaulted_object_field("window_width", &Config::window_width),
      defaulted_object_field("window_height", &Config::window_height),
      defaulted_object_field("window_title", &Config::window_title),
      defaulted_object_field("display_rate_hz", &Config::display_rate_hz),
      defaulted_object_field("dynamic_range_db", &Config::dynamic_range_db),
      defaulted_object_field("detection_radius", &Config::detection_radius),
      defaulted_object_field("history_frames", &Config::history_frames),
      defaulted_object_field("detection_trail_frames",
                             &Config::detection_trail_frames),
      defaulted_object_field("profile", &Config::profile));
};

}  // namespace holoscan::config
