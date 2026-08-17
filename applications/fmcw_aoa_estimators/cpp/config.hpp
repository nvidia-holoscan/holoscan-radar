// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for fmcw_aoa_estimators: AppConfig + YAML loader.
// [EA HSDK5 ONLY] Operator blocks are parsed into Params structs (Graph has
// no from_config()).

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_app/app/config.hpp"
#include "radar_app/app/graph_params.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/scene.hpp"
#include "custom_ops.hpp"  // NOLINT(build/include_subdir): colocated app header

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  EstimatorRouterConfig router;
  radar::FMCWRangeProcessorParams range_params;
  radar::ClutterFilterParams clutter_params;
  radar::FMCWDopplerProcessorParams doppler_params;
  radar::CFARParams cfar_params;
  radar::SnapshotGatherParams gather_params;
  radar::BeamscanParams beamscan_params;
  radar::MVDRParams mvdr_params;
  radar::MUSICParams music_params;
  radar::RootMUSICParams root_music_params;
  radar::ESPRITParams esprit_params;
  radar::ClusterDBSCANParams dbscan_params;
  radar::DetectionReductionParams reduction_params;
  radar::MaxTensorExtents extents;
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.35}, el_hi{0.35};
};

inline int64_t snapshot_count(const std::vector<int64_t>& window) {
  int64_t doppler_half = 0;
  int64_t range_half = 0;
  if (window.size() >= 1) {
    doppler_half = window[0];
  }
  if (window.size() >= 2) {
    range_half = window[1];
  } else if (window.size() == 1) {
    range_half = window[0];
  }
  return (2 * doppler_half + 1) * (2 * range_half + 1);
}

inline radar::MaxTensorExtents make_aoa_extents(const AppConfig& cfg) {
  radar::MaxTensorExtents e;
  const int64_t n_rx = cfg.array_geometry.num_elements();
  const int64_t chirps = cfg.scene.num_chirps;
  const int64_t samples = cfg.scene.samples_per_chirp;
  const int64_t range_bins =
      cfg.range_params.nfft > 0 ? cfg.range_params.nfft : samples;
  const int64_t doppler_bins =
      cfg.doppler_params.nfft > 0 ? cfg.doppler_params.nfft : chirps;
  const int64_t map_cells = doppler_bins * range_bins;
  const int64_t num_az =
      static_cast<int64_t>(cfg.beamscan_params.az_angles.size());
  const int64_t num_el =
      static_cast<int64_t>(cfg.beamscan_params.el_angles.size());
  const int64_t k_snaps = snapshot_count(cfg.gather_params.snapshot_window);

  e.max_rx = n_rx;
  e.max_chirps = chirps;
  e.max_samples_per_chirp = samples;
  e.max_range_bins = range_bins;
  e.max_doppler_bins = doppler_bins;
  e.max_channels = n_rx;
  e.max_detections = map_cells;
  e.max_items = map_cells;
  e.max_points = map_cells;
  e.max_snapshots = k_snaps > 0 ? k_snaps : 1;
  e.max_az = num_az > 0 ? num_az : 1;
  e.max_el = num_el > 0 ? num_el : 1;
  e.max_clusters = map_cells;
  e.max_sources = std::max(
      {cfg.mvdr_params.max_sources, cfg.music_params.max_sources,
       cfg.root_music_params.max_sources, cfg.esprit_params.max_sources,
       int64_t{1}});
  return e;
}

inline RouterMode parse_router_mode(const std::string& text) {
  if (text == "rotate") {
    return RouterMode::kRotate;
  }
  if (text == "hold") {
    return RouterMode::kHold;
  }
  if (text == "file") {
    return RouterMode::kFile;
  }
  throw std::runtime_error(
      "estimator_router.mode: unknown value '" + text +
      "' (accepted: rotate, hold, file)");
}

inline int64_t resolve_hold_index(const std::string& hold,
                                  const std::vector<std::string>& names) {
  std::string norm;
  for (char c : hold) {
    if (c == '-' || c == '_') {
      continue;
    }
    norm.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }
  for (size_t i = 0; i < names.size(); ++i) {
    std::string cand;
    for (char c : names[i]) {
      if (c == '-' || c == '_') {
        continue;
      }
      cand.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
    }
    if (norm == cand) {
      return static_cast<int64_t>(i);
    }
  }
  throw std::runtime_error("estimator_router.hold: unknown estimator '" +
                           hold + "'");
}

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);

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
        "rx_array.az_elements and rx_array.el_elements must both be >= 2");
  }
  cfg.array_geometry = radar::sim::make_ura_geometry(az_elements, el_elements,
                                                     spacing, wavelength);
  cfg.scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, spacing, wavelength);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_aoa_estimators");
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);

  load_range_params(root["range_processor"], &cfg.range_params);
  load_clutter_params(root["clutter_filter"], &cfg.clutter_params);
  load_doppler_params(root["doppler_processor"], &cfg.doppler_params);
  load_cfar_params(root["cfar_detector"], &cfg.cfar_params);
  cfg.cfar_params.nci_fold = cfg.array_geometry.num_elements();
  load_snapshot_gather_params(root["snapshot_gather"], &cfg.gather_params);
  load_beamscan_params(root["beamscan_estimator"], &cfg.beamscan_params);
  load_mvdr_params(root["mvdr_estimator"], &cfg.mvdr_params);
  load_music_params(root["music_estimator"], &cfg.music_params);
  load_root_music_params(root["root_music_estimator"], &cfg.root_music_params);
  load_esprit_params(root["esprit_estimator"], &cfg.esprit_params);
  load_dbscan_params(root["cluster_dbscan"], &cfg.dbscan_params);
  load_reduction_params(root["reduction"], &cfg.reduction_params);

  cfg.beamscan_params.array_geometry = cfg.array_geometry;
  cfg.mvdr_params.array_geometry = cfg.array_geometry;
  cfg.music_params.array_geometry = cfg.array_geometry;
  cfg.root_music_params.array_geometry = cfg.array_geometry;
  cfg.esprit_params.array_geometry = cfg.array_geometry;

  cfg.peak_only = cfg.cfar_params.peak_only;

  if (const YAML::Node er = root["estimator_router"]; er && !er.IsNull()) {
    if (er["mode"]) {
      cfg.router.mode = parse_router_mode(er["mode"].as<std::string>());
    }
    if (er["rotate_period"]) {
      cfg.router.rotate_period = er["rotate_period"].as<int64_t>();
    }
    std::string hold = cfg.router.display_names[0];
    if (er["hold"]) {
      hold = er["hold"].as<std::string>();
    }
    cfg.router.hold_index = resolve_hold_index(hold, cfg.router.display_names);
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

  grid_extent(cfg.beamscan_params.az_angles, -0.35, 0.35, &cfg.az_lo,
              &cfg.az_hi);
  grid_extent(cfg.beamscan_params.el_angles, -0.35, 0.35, &cfg.el_lo,
              &cfg.el_hi);
  cfg.extents = make_aoa_extents(cfg);
  return cfg;
}
