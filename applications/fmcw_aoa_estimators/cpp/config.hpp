// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_aoa_estimators: the AppConfig type and its
// YAML loader (a filled y-z URA shared by all five estimators, 3-D targets, the
// boresight-view extents, and the estimator-router control policy). Colocated
// with the app; the graph wiring lives in fmcw_aoa_estimators.cpp.

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

// Router control policy parsed from the app-owned `estimator_router` block.
// mode: "rotate" (fixed dwell), "hold" (pin one), "file" (poll control_path,
// latch a live override, fall back to rotation).
struct RouterConfig {
  std::string mode{"rotate"};
  int64_t rotate_period{40};
  std::string hold{"beamscan"};
  std::string control_path{};
  int64_t log_every{0};
};

// Config: shared run/viz/plots fields plus the scene, the URA geometry, the
// boresight-view angular extents, and the router policy.
struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  RouterConfig router;
  // Boresight view extents (radians), mirrored from the grid-scan estimator.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.35}, el_hi{0.35};
};

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);  // waveform + run + cube

  // rx_array -> filled y-z URA geometry + row-aligned simulator RX positions,
  // in the (iz outer, iy inner) channel order all five estimators share. The
  // two orthogonal apertures resolve azimuth (y) AND elevation (z): the
  // gridless Root-MUSIC / ESPRIT estimators take their native 2-D paths, and
  // the grid-scan estimators scan the full (az, el) grid. `spacing` is a
  // fraction of the carrier wavelength (0.5 = half-wavelength).
  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required");
  }
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  const double spacing = rx["spacing"].as<double>();
  const double wavelength = cfg.scene.waveform.wavelength();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "rx_array.az_elements and rx_array.el_elements must both be >= 2 for a "
        "filled URA the gridless estimators can resolve in 2-D");
  }
  cfg.array_geometry = radar::sim::make_ura_geometry(az_elements, el_elements,
                                                     spacing, wavelength);
  cfg.scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, spacing, wavelength);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_aoa_estimators");
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

  if (const YAML::Node er = root["estimator_router"]; er && !er.IsNull()) {
    if (er["mode"]) {
      cfg.router.mode = er["mode"].as<std::string>();
    }
    if (er["rotate_period"]) {
      cfg.router.rotate_period = er["rotate_period"].as<int64_t>();
    }
    if (er["hold"]) {
      cfg.router.hold = er["hold"].as<std::string>();
    }
    if (er["control_path"]) {
      cfg.router.control_path = er["control_path"].as<std::string>();
    }
    if (er["log_every"]) {
      cfg.router.log_every = er["log_every"].as<int64_t>();
    }
  }
  if (cfg.router.rotate_period <= 0) {
    throw std::runtime_error("estimator_router.rotate_period must be > 0");
  }
  if (cfg.router.log_every < 0) {
    throw std::runtime_error("estimator_router.log_every must be >= 0");
  }

  // Boresight view axes mirror the beamscan estimator's angle grid so the axes
  // match what the estimators can resolve. READ-ONLY MIRROR of an operator
  // block — the estimators still own az_angles/el_angles via from_config(); we
  // only read them to size the plot. Pad a degenerate axis.
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(extract_grid(be ? be["az_angles"] : YAML::Node()), -0.35, 0.35,
              &cfg.az_lo, &cfg.az_hi);
  grid_extent(extract_grid(be ? be["el_angles"] : YAML::Node()), -0.35, 0.35,
              &cfg.el_lo, &cfg.el_hi);
  return cfg;
}
