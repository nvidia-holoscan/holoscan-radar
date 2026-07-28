// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for fmcw_siso: the AppConfig type and its YAML loader.
// Colocated with the app; the graph wiring lives in fmcw_siso.cpp. Operator
// blocks are NOT parsed here — they flow through Holoscan's from_config().

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>

#include "radar_app/app/config.hpp"
#include "radar_common/sim/scene.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

// SISO config: the shared run/viz/plots fields plus the app's own SimScene.
struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;  // waveform, cube dims, targets, noise
};

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);  // waveform + run + cube
  parse_targets_range1d(root, &cfg.scene);
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);
  // READ-ONLY MIRROR of the cfar_detector operator block: peak_only stays a
  // native CFAR param (applied via from_config); the app reads it here only to
  // decide the graph shape (drop clustering when set).
  if (const YAML::Node cd = root["cfar_detector"];
      cd && cd["peak_only"] && !cd["peak_only"].IsNull()) {
    cfg.peak_only = cd["peak_only"].as<bool>();
  }
  return cfg;
}
