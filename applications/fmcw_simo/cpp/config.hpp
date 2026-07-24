// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_simo: the AppConfig type and its YAML loader
// (URA geometry + 3-D targets + the boresight-axis mirror of the beam grid).
// Colocated with the app; the graph wiring lives in fmcw_simo.cpp.

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_app/app/config.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

// SIMO config: shared run/viz/plots fields plus the scene, the URA geometry,
// and the boresight-view angular extents.
struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  // Beamscan angular grid extents (radians), for the boresight view axes.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
};

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);  // waveform + run + cube

  // rx_array → URA geometry + simulator RX positions. Both derive from the same
  // (az, el, spacing) at the carrier wavelength, so the ArrayGeometry the
  // beamscan estimator uses and the scene's rx_positions stay row-aligned. The
  // element pitch `spacing` is a fraction of the carrier wavelength (0.5 =
  // half-wavelength).
  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required for SIMO");
  }
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  const double spacing = rx["spacing"].as<double>();
  const double wavelength = cfg.scene.waveform.wavelength();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2 to "
        "observe both azimuth and elevation");
  }
  cfg.array_geometry =
      radar::sim::make_ura_geometry(az_elements, el_elements, spacing,
                                    wavelength);
  cfg.scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, spacing, wavelength);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_simo");
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

  // Boresight view axes mirror the beamscan estimator's angle grid so the axes
  // match what the estimator can resolve. This is a READ-ONLY MIRROR of an
  // operator block — the estimator still owns az_angles/el_angles via
  // from_config(); we only read them to size the plot. Pad a degenerate axis.
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(extract_grid(be ? be["az_angles"] : YAML::Node()), -0.35, 0.35,
              &cfg.az_lo, &cfg.az_hi);
  grid_extent(extract_grid(be ? be["el_angles"] : YAML::Node()), -0.1, 0.1,
              &cfg.el_lo, &cfg.el_hi);
  return cfg;
}
