// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for fmcw_dense_simo: AppConfig + YAML loader.
// [EA HSDK5 ONLY] Operator blocks are parsed into Params structs (Graph has
// no from_config()).

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_app/app/angle_presentation.hpp"
#include "radar_app/app/config.hpp"
#include "radar_app/app/graph_params.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/scene.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

struct AppConfig : CommonRunConfig {
  radar::sim::SimScene scene;
  radar::ArrayGeometry array_geometry;
  radar::FMCWRangeProcessorParams range_params;
  radar::ClutterFilterParams clutter_params;
  radar::FMCWDopplerProcessorParams doppler_params;
  radar::PhaseShiftBeamformerParams beamformer_params;
  radar::CFARParams cfar_params;
  radar::ClusterDBSCANParams dbscan_params;
  radar::DetectionReductionParams reduction_params;
  radar::MaxTensorExtents extents;
  // Dense CFAR on (az, el, D, R) needs leading-rank-2 pool sizing via
  // max_tx/max_rx standing in for max_az/max_el (named_tx_rx_leading_rank).
  radar::MaxTensorExtents cfar_extents;
  // Boresight panel axes, from the beamformer scan grid.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
};

inline radar::MaxTensorExtents make_dense_simo_extents(const AppConfig& cfg) {
  radar::MaxTensorExtents e;
  const int64_t n_rx = cfg.array_geometry.num_elements();
  const int64_t chirps = cfg.scene.num_chirps;
  const int64_t samples = cfg.scene.samples_per_chirp;
  const int64_t range_bins =
      cfg.range_params.nfft > 0 ? cfg.range_params.nfft : samples;
  const int64_t doppler_bins =
      cfg.doppler_params.nfft > 0 ? cfg.doppler_params.nfft : chirps;
  const int64_t num_az =
      static_cast<int64_t>(cfg.beamformer_params.az_coordinates.size());
  const int64_t num_el =
      static_cast<int64_t>(cfg.beamformer_params.el_coordinates.size());
  const int64_t map_cells = doppler_bins * range_bins;
  const int64_t beam_cells = (num_az > 0 ? num_az : 1) *
                             (num_el > 0 ? num_el : 1) * map_cells;

  e.max_rx = n_rx;
  e.max_chirps = chirps;
  e.max_samples_per_chirp = samples;
  e.max_range_bins = range_bins;
  e.max_doppler_bins = doppler_bins;
  e.max_channels = num_az * num_el;
  e.max_az = num_az > 0 ? num_az : 1;
  e.max_el = num_el > 0 ? num_el : 1;
  e.max_detections = beam_cells;
  e.max_items = beam_cells;
  e.max_points = beam_cells;
  e.max_features = 4;
  const int64_t cluster_cap = cfg.dbscan_params.max_clusters > 0
                                  ? cfg.dbscan_params.max_clusters
                                  : beam_cells;
  e.max_clusters = cluster_cap;
  return e;
}

// CFAR on the RDA cube: (az, el) leading rank must be 2. Reuse max_tx/max_rx
// as az/el pool dims so named_tx_rx_leading_rank reports 2 and idx_cols=4.
inline radar::MaxTensorExtents make_dense_cfar_extents(
    const radar::MaxTensorExtents& base) {
  radar::MaxTensorExtents e = base;
  e.max_tx = base.max_az;
  e.max_rx = base.max_el;
  e.max_channels = std::nullopt;
  return e;
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
  cfg.array_geometry = radar::sim::make_ura_geometry(
      az_elements, el_elements, spacing, wavelength);
  cfg.scene.rx_positions = radar::sim::make_ura_rx_positions(
      az_elements, el_elements, spacing, wavelength);

  parse_targets_range_az_el(root, &cfg.scene, "fmcw_dense_simo");
  parse_noise(root, &cfg.scene);
  parse_viz_toggles(root, &cfg);
  parse_plots(root, &cfg);

  load_range_params(root["range_processor"], &cfg.range_params);
  load_clutter_params(root["clutter_filter"], &cfg.clutter_params);
  load_doppler_params(root["doppler_processor"], &cfg.doppler_params);
  load_phase_shift_beamformer_params(root["phase_shift_beamformer"],
                                     &cfg.beamformer_params);
  cfg.beamformer_params.array_geometry = cfg.array_geometry;
  if (cfg.beamformer_params.az_coordinates.empty() ||
      cfg.beamformer_params.el_coordinates.empty()) {
    throw std::runtime_error(
        "phase_shift_beamformer.az_coordinates and el_coordinates are required");
  }
  load_cfar_params(root["cfar_detector"], &cfg.cfar_params);
  // Dense path: nci_axes empty so az/el ride as batch columns on
  // detection_idx.
  if (!cfg.cfar_params.nci_axes.empty()) {
    throw std::runtime_error(
        "fmcw_dense_simo: cfar_detector.nci_axes must be empty (dense RDA "
        "path)");
  }
  load_dbscan_params(root["cluster_dbscan"], &cfg.dbscan_params);
  load_reduction_params(root["reduction"], &cfg.reduction_params);

  cfg.use_cfar_peaks =
      cfg.cfar_params.peak_mode != holoscan::radar::CfarPeakMode::kNone;
  const PhysicalAngleBounds physical_bounds = physical_scan_grid_bounds(
      cfg.beamformer_params.angle_convention, cfg.beamformer_params.az_coordinates,
      cfg.beamformer_params.el_coordinates);
  cfg.az_lo = physical_bounds.az[0];
  cfg.az_hi = physical_bounds.az[1];
  cfg.el_lo = physical_bounds.el[0];
  cfg.el_hi = physical_bounds.el[1];
  cfg.extents = make_dense_simo_extents(cfg);
  cfg.cfar_extents = make_dense_cfar_extents(cfg.extents);
  return cfg;
}
