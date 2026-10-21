// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
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
  bool noise_scaling{true};
  int64_t noise_sample_limit{2048};
  int64_t noise_range_half_width{3};
  int64_t noise_doppler_half_width{3};
  double noise_sigma{2.0};
  double noise_headroom_db{0.0};
  double range_doppler_width_percent{45.0};
  double detection_radius{0.010};
  int64_t history_frames{120};
  int64_t detection_trail_frames{12};
  std::string profile;

  void validate_layout() const {
    if (noise_sample_limit < 64 || noise_sample_limit > 65536 || noise_range_half_width < 0 ||
        noise_range_half_width > 64 || noise_doppler_half_width < 0 ||
        noise_doppler_half_width > 64 || !std::isfinite(noise_sigma) || noise_sigma < 0.0 ||
        noise_sigma > 10.0 || !std::isfinite(static_cast<float>(noise_headroom_db)) ||
        noise_headroom_db < 0.0) {
      throw std::runtime_error(
          "app.visualization: invalid noise scaling settings "
          "(samples 64..65536, half-widths 0..64, sigma 0..10, headroom finite and nonnegative)");
    }
    if (!std::isfinite(range_doppler_width_percent) || range_doppler_width_percent < 25.0 ||
        range_doppler_width_percent > 70.0) {
      throw std::runtime_error(
          "app.visualization.range_doppler_width_percent must be finite and in [25, 70]");
    }
  }
};

}  // namespace holoscan::radar::app

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::VisualizationConfig>
: radar::config_detail::PortableParamStorageTraits<radar::app::VisualizationConfig> {};  // NOLINT

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
      defaulted_object_field("noise_scaling", &Config::noise_scaling),
      defaulted_object_field("noise_sample_limit", &Config::noise_sample_limit),
      defaulted_object_field("noise_range_half_width", &Config::noise_range_half_width),
      defaulted_object_field("noise_doppler_half_width", &Config::noise_doppler_half_width),
      defaulted_object_field("noise_sigma", &Config::noise_sigma),
      defaulted_object_field("noise_headroom_db", &Config::noise_headroom_db),
      defaulted_object_field("range_doppler_width_percent", &Config::range_doppler_width_percent),
      defaulted_object_field("detection_radius", &Config::detection_radius),
      defaulted_object_field("history_frames", &Config::history_frames),
      defaulted_object_field("detection_trail_frames", &Config::detection_trail_frames),
      defaulted_object_field("profile", &Config::profile));
};

}  // namespace holoscan::config
