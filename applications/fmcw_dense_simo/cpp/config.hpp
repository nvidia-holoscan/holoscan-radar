// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_dense_simo: the AppConfig type and its YAML
// loader (URA geometry + beamformer angle-grid mirror + 3-D targets). Colocated
// with the app; the graph wiring lives in fmcw_dense_simo.cpp.

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_app/app/config.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

// Dense config: shared run/viz/plots fields plus the scene, URA geometry, the
// beamformer angle grids (mirrored for the az/el lookup + report + boresight),
// and the dense-only velocity-plot Y clamp.
struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  std::vector<double> az_angles;
  std::vector<double> el_angles;
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
};

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);  // waveform + run + cube

  // rx_array → URA geometry + simulator RX positions (matches fmcw_simo so the
  // two demos share a scene).
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
        "SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2");
  }
  cfg.array_geometry = radar::sim::make_ura_geometry(az_elements, el_elements,
                                                     spacing, wavelength);
  cfg.scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, spacing, wavelength);

  // The dense report and the az/el lookup bridge both need the beam grid. This
  // is a READ-ONLY MIRROR of the phase_shift_beamformer block — the beamformer
  // still owns az_angles/el_angles via from_config().
  const YAML::Node beam = root["phase_shift_beamformer"];
  if (!beam || !beam["az_angles"] || !beam["el_angles"]) {
    throw std::runtime_error(
        "phase_shift_beamformer.az_angles and el_angles are required");
  }
  cfg.az_angles = extract_grid(beam["az_angles"]);
  cfg.el_angles = extract_grid(beam["el_angles"]);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_dense_simo");
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);
  parse_peak_only(root, &cfg);

  // Dense-only plot knob: optional fixed velocity-plot Y extent.
  if (const YAML::Node plots = root["plots"]; plots && !plots.IsNull()) {
    const YAML::Node vy = plots["velocity_ylim"];
    if (vy && vy.IsSequence() && vy.size() == 2) {
      cfg.velocity_ylim_lo = vy[0].as<float>();
      cfg.velocity_ylim_hi = vy[1].as<float>();
    }
  }
  return cfg;
}
