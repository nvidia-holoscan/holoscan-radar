// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for fmcw_siso: AppConfig + YAML loader.
// [EA HSDK5 ONLY] Operator blocks are parsed here into Params structs
// (Graph has no from_config()); restore native from_config() when Application
// demos return.

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>

#include "radar_app/app/config.hpp"
#include "radar_app/app/graph_params.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sim/scene.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::FMCWRangeProcessorParams range_params;
  radar::ClutterFilterParams clutter_params;
  radar::FMCWDopplerProcessorParams doppler_params;
  radar::CFARParams cfar_params;
  radar::ClusterDBSCANParams dbscan_params;
  radar::DetectionReductionParams reduction_params;
  radar::MaxTensorExtents extents;
};

inline radar::MaxTensorExtents make_siso_extents(const AppConfig& cfg) {
  radar::MaxTensorExtents e;
  const int64_t chirps = cfg.scene.num_chirps;
  const int64_t samples = cfg.scene.samples_per_chirp;
  const int64_t range_bins =
      cfg.range_params.nfft > 0 ? cfg.range_params.nfft : samples;
  const int64_t doppler_bins =
      cfg.doppler_params.nfft > 0 ? cfg.doppler_params.nfft : chirps;
  e.max_chirps = chirps;
  e.max_samples_per_chirp = samples;
  e.max_range_bins = range_bins;
  e.max_doppler_bins = doppler_bins;
  const int64_t map_cells = doppler_bins * range_bins;
  e.max_detections = map_cells;
  e.max_points = map_cells;
  e.max_features = 2;
  const int64_t cluster_cap = cfg.dbscan_params.max_clusters > 0
                                  ? cfg.dbscan_params.max_clusters
                                  : map_cells;
  e.max_clusters = cluster_cap;
  return e;
}

inline AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;
  parse_run_and_cube(root, &cfg, &cfg.scene);
  parse_targets_range1d(root, &cfg.scene);
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);

  load_range_params(root["range_processor"], &cfg.range_params);
  load_clutter_params(root["clutter_filter"], &cfg.clutter_params);
  load_doppler_params(root["doppler_processor"], &cfg.doppler_params);
  load_cfar_params(root["cfar_detector"], &cfg.cfar_params);
  load_dbscan_params(root["cluster_dbscan"], &cfg.dbscan_params);
  load_reduction_params(root["reduction"], &cfg.reduction_params);

  cfg.use_cfar_peaks =
      cfg.cfar_params.peak_mode != holoscan::radar::CfarPeakMode::kNone;
  cfg.extents = make_siso_extents(cfg);
  return cfg;
}
