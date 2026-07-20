// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_siso: the AppConfig type and its YAML loader.
// Colocated with the app; the graph wiring lives in fmcw_siso.cpp. Operator
// blocks are NOT parsed here — they flow through Holoscan's from_config().

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>

#include "radar_app/app/config.hpp"
#include "radar_common/sim/point_target_sim.hpp"

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
  return cfg;
}
