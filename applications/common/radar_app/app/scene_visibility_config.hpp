// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <holoscan/params/config.hpp>

#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::app {

struct SceneVisibilityConfig {
  std::string model{"front_half_space"};
  std::array<double, 3> boresight_direction{1.0, 0.0, 0.0};
  std::array<double, 3> up_direction{0.0, 0.0, 1.0};
  std::optional<double> horizontal_fov{};
  std::optional<double> vertical_fov{};
};

using SceneVisibilityStorage = radar::config_detail::PortableParamStorageTraits<
    SceneVisibilityConfig>;  // NOLINT

inline sim::SceneVisibility configure_scene_visibility(
    const SceneVisibilityConfig& authored) {
  sim::SceneVisibility result;
  if (authored.model == "front_half_space") {
    result.model = sim::SceneVisibilityModel::kFrontHalfSpace;
  } else if (authored.model == "frustum") {
    result.model = sim::SceneVisibilityModel::kFrustum;
  } else if (authored.model == "all") {
    result.model = sim::SceneVisibilityModel::kAll;
  } else {
    throw std::runtime_error(
        "app.scene_visibility.model must be front_half_space, frustum, or "
        "all");
  }
  result.boresight_direction = authored.boresight_direction;
  result.up_direction = authored.up_direction;
  result.horizontal_fov = authored.horizontal_fov;
  result.vertical_fov = authored.vertical_fov;
  try {
    sim::validate_scene_visibility(result);
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error(std::string("app.scene_visibility: ") +
                             error.what());
  }
  return result;
}

}  // namespace holoscan::radar::app

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::SceneVisibilityConfig>
    : radar::app::SceneVisibilityStorage {};  // NOLINT

}  // namespace holoscan

namespace holoscan::config {

template <>
struct ConfigCodec<radar::app::SceneVisibilityConfig>
    : ObjectCodec<radar::app::SceneVisibilityConfig> {  // NOLINT
  using Config = radar::app::SceneVisibilityConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("model", &Config::model),
      defaulted_object_field("boresight_direction",
                             &Config::boresight_direction),
      defaulted_object_field("up_direction", &Config::up_direction),
      defaulted_object_field("horizontal_fov", &Config::horizontal_fov),
      defaulted_object_field("vertical_fov", &Config::vertical_fov));
};

}  // namespace holoscan::config
