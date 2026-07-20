// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_mimo: AppConfig plus the scene-geometry and
// YAML loaders (TDM/DDM branch, virtual array geometry, targets, the
// boresight-axis mirror of the beamscan grid). Colocated with the app. The
// scene is the shared radar::sim::SimScene (multi-TX); the MIMO simulators live
// in radar_common/sim/mimo_sim.hpp.

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/mimo_sim.hpp"
#include "radar_app/app/config.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

struct AppConfig : CommonRunConfig {
  radar::MultiplexingScheme scheme{radar::MultiplexingScheme::kTdm};
  radar::sim::SimScene scene;
  radar::TXSchedule tx_schedule;
  radar::DDMCodeDescriptor ddm_code;
  std::vector<double> ddm_doppler_offsets;
  radar::ArrayGeometry array_geometry;
  std::vector<double> tx_positions_flat;
  std::vector<double> rx_positions_flat;
  double wavelength{0.0};
  // Optional fixed velocity-plot Y extent (m/s); NaN keeps the full-axis span.
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
  radar::FMCWWaveformDescriptor report_waveform;
  int64_t report_num_doppler{0};
};

// Build the TX row + RX URA + the virtual (TX⊕RX) geometry into the scene.
inline void load_scene_geometry(AppConfig* cfg, const YAML::Node& root,
                                double wavelength, int64_t num_tx) {
  const YAML::Node tx = root["tx_array"];
  const double tx_spacing = tx["spacing"].as<double>();
  cfg->scene.tx_positions =
      radar::sim::make_linear_positions(num_tx, tx_spacing, wavelength);

  const YAML::Node rx = root["rx_array"];
  if (!rx) {
    throw std::runtime_error("rx_array section is required for MIMO");
  }
  const int64_t az_elements = rx["az_elements"].as<int64_t>();
  const int64_t el_elements = rx["el_elements"].as<int64_t>();
  const double rx_spacing = rx["spacing"].as<double>();
  if (az_elements < 2 || el_elements < 2) {
    throw std::runtime_error(
        "MIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2");
  }
  cfg->scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, rx_spacing, wavelength);

  cfg->array_geometry = radar::sim::make_virtual_geometry(
      cfg->scene.tx_positions, cfg->scene.rx_positions, wavelength);
}

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;

  const auto waveform = root["waveform"].as<radar::FMCWWaveformDescriptor>();
  cfg.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  cfg.scene.waveform.bandwidth = waveform.bandwidth();
  cfg.scene.waveform.chirp_duration = waveform.chirp_duration();
  cfg.scene.waveform.sample_rate = waveform.sample_rate();
  cfg.report_waveform = waveform;

  const YAML::Node mcc = root["mimo_coherent_compensation"];
  if (!mcc || !mcc["multiplexing_scheme"]) {
    throw std::runtime_error(
        "fmcw_mimo: mimo_coherent_compensation.multiplexing_scheme is "
        "required");
  }
  cfg.scheme = mcc["multiplexing_scheme"].as<radar::MultiplexingScheme>();
  if (cfg.scheme != radar::MultiplexingScheme::kTdm &&
      cfg.scheme != radar::MultiplexingScheme::kDdm) {
    throw std::runtime_error(
        "fmcw_mimo: multiplexing_scheme must be tdm or ddm");
  }

  parse_run(root, &cfg);
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

  const YAML::Node tx = root["tx_array"];
  if (!tx) {
    throw std::runtime_error("tx_array section is required for MIMO");
  }
  const int64_t num_tx = tx["num_tx"].as<int64_t>();
  if (num_tx < 1) {
    throw std::runtime_error("tx_array.num_tx must be >= 1");
  }

  const double wavelength = cfg.scene.waveform.wavelength();
  cfg.wavelength = wavelength;
  load_scene_geometry(&cfg, root, wavelength, num_tx);

  if (cfg.scheme == radar::MultiplexingScheme::kDdm) {
    cfg.scene.waveform.chirp_repetition_interval =
        waveform.chirp_repetition_interval();
    if (root["cube"]["num_chirps"]) {
      cfg.scene.num_chirps = root["cube"]["num_chirps"].as<int64_t>();
    } else {
      cfg.scene.num_chirps = root["cube"]["chirps_per_tx"].as<int64_t>();
    }
    if (!mcc["ddm_code"]) {
      throw std::runtime_error(
          "fmcw_mimo: mimo_coherent_compensation.ddm_code is required for DDM");
    }
    cfg.ddm_code = mcc["ddm_code"].as<radar::DDMCodeDescriptor>();
    if (cfg.ddm_code.num_tx != num_tx) {
      throw std::runtime_error(
          "fmcw_mimo: ddm_code.num_tx must match tx_array.num_tx");
    }
    cfg.ddm_doppler_offsets = radar::sim::resolve_ddm_offsets(cfg.ddm_code);
    if (cfg.scene.num_chirps % num_tx != 0) {
      throw std::runtime_error(
          "fmcw_mimo: cube.num_chirps must be divisible by tx_array.num_tx "
          "for DDM");
    }
    const int64_t band = cfg.scene.num_chirps / num_tx;
    cfg.report_num_doppler = band - cfg.ddm_code.guard_bands;
    if (cfg.report_num_doppler <= 0) {
      throw std::runtime_error(
          "fmcw_mimo: DDM guard_bands leaves no Doppler bins per TX");
    }
  } else {
    const YAML::Node sim = root["sim"];
    if (!sim || !sim["physical_chirp_repetition_interval"]) {
      throw std::runtime_error(
          "fmcw_mimo: sim.physical_chirp_repetition_interval is required "
          "for TDM");
    }
    cfg.scene.waveform.chirp_repetition_interval =
        sim["physical_chirp_repetition_interval"].as<double>();
    cfg.scene.num_chirps = root["cube"]["chirps_per_tx"].as<int64_t>();
    if (!root["mimo_deinterleave"]) {
      throw std::runtime_error(
          "fmcw_mimo: mimo_deinterleave section is required for TDM");
    }
    cfg.tx_schedule =
        root["mimo_deinterleave"]["tx_schedule"].as<radar::TXSchedule>();
    if (cfg.tx_schedule.num_tx != num_tx) {
      throw std::runtime_error(
          "mimo_deinterleave.tx_schedule.num_tx must match tx_array.num_tx");
    }
    cfg.report_num_doppler = cfg.scene.num_chirps;
  }

  cfg.tx_positions_flat = radar::sim::flatten_positions(cfg.scene.tx_positions);
  cfg.rx_positions_flat = radar::sim::flatten_positions(cfg.scene.rx_positions);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_mimo");
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);

  // Optional fixed velocity-plot Y extent (dense/MIMO plot knob).
  if (const YAML::Node plots = root["plots"]; plots && !plots.IsNull()) {
    if (const YAML::Node ylim = plots["velocity_ylim"]) {
      if (!ylim.IsSequence() || ylim.size() != 2) {
        throw std::runtime_error("plots.velocity_ylim must be a [lo, hi] pair");
      }
      cfg.velocity_ylim_lo = ylim[0].as<float>();
      cfg.velocity_ylim_hi = ylim[1].as<float>();
    }
  }

  // Boresight view axes mirror the beamscan estimator's angle grid (READ-ONLY
  // MIRROR of an operator block; the estimator owns az_angles/el_angles via
  // from_config()).
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(extract_grid(be ? be["az_angles"] : YAML::Node()), -0.35, 0.35,
              &cfg.az_lo, &cfg.az_hi);
  grid_extent(extract_grid(be ? be["el_angles"] : YAML::Node()), -0.1, 0.1,
              &cfg.el_lo, &cfg.el_hi);
  return cfg;
}
