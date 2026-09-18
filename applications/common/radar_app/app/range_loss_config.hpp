// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>

#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::app {

struct RangeLossConfig {
  std::string model{"none"};
  std::optional<double> reference_range{};
};

using RangeLossStorage = radar::config_detail::PortableParamStorageTraits<
    RangeLossConfig>;  // NOLINT

inline void configure_range_loss(
    const std::optional<RangeLossConfig>& authored, sim::SimScene* scene) {
  scene->range_loss = {};
  if (!authored.has_value()) {
    HOLOSCAN_LOG_WARN(
        "app.range_loss is omitted; target amplitudes remain independent of "
        "physical range (model: none); configure "
        "monostatic_free_space with a positive reference_range to enable "
        "R^-4 spreading");
    return;
  }
  if (authored->model == "none") {
    if (authored->reference_range.has_value() &&
        (!std::isfinite(*authored->reference_range) ||
         *authored->reference_range <= 0.0)) {
      throw std::runtime_error(
          "app.range_loss.reference_range must be finite and positive");
    }
    HOLOSCAN_LOG_WARN(
        "app.range_loss.model is none; target amplitudes remain independent "
        "of physical range; configure monostatic_free_space with a positive "
        "reference_range to enable R^-4 spreading");
    return;
  }
  if (authored->model != "monostatic_free_space") {
    throw std::runtime_error(
        "app.range_loss.model must be none or monostatic_free_space");
  }
  if (!authored->reference_range.has_value() ||
      !std::isfinite(*authored->reference_range) ||
      *authored->reference_range <= 0.0) {
    throw std::runtime_error(
        "app.range_loss.reference_range must be finite and positive for "
        "monostatic_free_space");
  }
  scene->range_loss.model = sim::RangeLossModel::kMonostaticFreeSpace;
  scene->range_loss.reference_range = *authored->reference_range;
}

}  // namespace holoscan::radar::app

namespace holoscan {

template <>
struct ParamStorageTraits<radar::app::RangeLossConfig>
    : radar::app::RangeLossStorage {};  // NOLINT

}  // namespace holoscan

namespace holoscan::config {

template <>
struct ConfigCodec<radar::app::RangeLossConfig>
    : ObjectCodec<radar::app::RangeLossConfig> {  // NOLINT
  using Config = radar::app::RangeLossConfig;
  static constexpr auto fields = std::make_tuple(
      object_field("model", &Config::model),
      defaulted_object_field("reference_range", &Config::reference_range));
};

}  // namespace holoscan::config
