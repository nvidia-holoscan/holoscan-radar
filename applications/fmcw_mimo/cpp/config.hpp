// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned config for fmcw_mimo: AppConfig + YAML loader (TDM/DDM branch,
// virtual array geometry, targets). [EA HSDK5 ONLY] Operator blocks are
// parsed into Params structs (Graph has no from_config()).

#pragma once

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_app/app/angle_presentation.hpp"
#include "radar_app/app/config.hpp"
#include "radar_app/app/graph_params.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/scene.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

struct AppConfig : CommonRunConfig {
  radar::MultiplexingScheme scheme{radar::MultiplexingScheme::kTdm};
  radar::sim::SimScene scene;
  radar::TXSchedule tx_schedule;
  radar::DDMCodeDescriptor ddm_code;
  std::vector<double> ddm_doppler_offsets;
  // True when ddm_code is non-linear → correlate demux + phase-coded sim.
  bool ddm_correlate{false};
  radar::ArrayGeometry array_geometry;
  std::vector<double> tx_positions_flat;
  std::vector<double> rx_positions_flat;
  double wavelength{0.0};
  // Optional fixed velocity-plot Y extent (m/s); NaN keeps the full-axis span.
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
  // Boresight panel axes, from the beamscan scan grid.
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
  radar::FMCWWaveformDescriptor report_waveform;
  int64_t report_num_doppler{0};

  radar::FMCWRangeProcessorParams range_params;
  radar::ClutterFilterParams clutter_params;
  radar::FMCWDopplerProcessorParams doppler_params;
  radar::MIMODeinterleaveParams deinterleave_params;
  radar::TDMDenseCompensationParams tdm_compensation_params;
  radar::DDMDenseDemultiplexParams ddm_demultiplex_params;
  radar::VirtualArrayAssemblyParams vaa_params;
  radar::CFARParams cfar_params;
  radar::SnapshotGatherParams gather_params;
  radar::BeamscanParams beamscan_params;
  radar::ClusterDBSCANParams dbscan_params;
  radar::DetectionReductionParams reduction_params;
  radar::MaxTensorExtents extents;
};

// Demo helpers for generate_phase_codes: expand unit-magnitude codes without
// dumping N*M complex values into YAML. Codes must be *non-linear* so the
// operator takes the correlate path (linear-phase ramps reduce to gather).

inline void fill_sin_phase_codes(radar::DDMCodeDescriptor* code,
                                 int64_t num_chirps) {
  if (code == nullptr || code->num_tx < 1 || num_chirps < 1) {
    throw std::invalid_argument("fill_sin_phase_codes: invalid args");
  }
  code->doppler_offsets.clear();
  code->guard_bands = 0;
  code->phase_codes.assign(
      static_cast<size_t>(code->num_tx * num_chirps), {0.0, 0.0});
  for (int64_t m = 0; m < code->num_tx; ++m) {
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double phase =
          std::sin(0.7 * static_cast<double>(k) + static_cast<double>(m)) *
          2.0;
      code->phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
}

inline void fill_zadoff_chu_phase_codes(radar::DDMCodeDescriptor* code,
                                        int64_t num_chirps) {
  if (code == nullptr || code->num_tx < 1 || num_chirps < 1) {
    throw std::invalid_argument("fill_zadoff_chu_phase_codes: invalid args");
  }
  std::vector<int64_t> roots;
  roots.reserve(static_cast<size_t>(code->num_tx));
  for (int64_t u = 1; u < num_chirps &&
                      static_cast<int64_t>(roots.size()) < code->num_tx;
       ++u) {
    if (std::gcd(u, num_chirps) == 1) {
      roots.push_back(u);
    }
  }
  if (static_cast<int64_t>(roots.size()) < code->num_tx) {
    throw std::invalid_argument(
        "fill_zadoff_chu_phase_codes: need num_tx distinct roots coprime to "
        "num_chirps (try a longer cube.num_chirps)");
  }

  code->doppler_offsets.clear();
  code->guard_bands = 0;
  code->phase_codes.assign(
      static_cast<size_t>(code->num_tx * num_chirps), {0.0, 0.0});
  const bool odd_n = (num_chirps % 2) != 0;
  for (int64_t m = 0; m < code->num_tx; ++m) {
    const double u = static_cast<double>(roots[static_cast<size_t>(m)]);
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double kd = static_cast<double>(k);
      const double arg =
          odd_n ? (-radar::kPi * u * kd * (kd + 1.0) /
                   static_cast<double>(num_chirps))
                : (-radar::kPi * u * kd * kd / static_cast<double>(num_chirps));
      code->phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(arg), std::sin(arg));
    }
  }
}

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

inline radar::MaxTensorExtents make_mimo_extents(const AppConfig& cfg) {
  radar::MaxTensorExtents e;
  const int64_t n_tx =
      static_cast<int64_t>(cfg.scene.tx_positions.size());
  const int64_t n_rx =
      static_cast<int64_t>(cfg.scene.rx_positions.size());
  const int64_t chirps = cfg.scene.num_chirps;
  const int64_t samples = cfg.scene.samples_per_chirp;
  const int64_t range_bins =
      cfg.range_params.nfft > 0 ? cfg.range_params.nfft : samples;
  // Pre-demux / pre-FFT Doppler length; demuxed DDM maps are smaller.
  const int64_t doppler_bins =
      cfg.doppler_params.nfft > 0 ? cfg.doppler_params.nfft : chirps;
  const int64_t report_doppler =
      cfg.report_num_doppler > 0 ? cfg.report_num_doppler : doppler_bins;
  const int64_t map_doppler =
      report_doppler > doppler_bins ? report_doppler : doppler_bins;
  const int64_t map_cells = map_doppler * range_bins;
  const int64_t n_virtual = cfg.array_geometry.num_elements();
  const int64_t num_az =
      static_cast<int64_t>(cfg.beamscan_params.az_coordinates.size());
  const int64_t num_el =
      static_cast<int64_t>(cfg.beamscan_params.el_coordinates.size());

  e.max_tx = n_tx;
  e.max_rx = n_rx;
  e.max_chirps = chirps;
  e.max_interleaved_chirps = n_tx * chirps;
  e.max_samples_per_chirp = samples;
  e.max_range_bins = range_bins;
  e.max_doppler_bins = map_doppler;
  e.max_virtual_channels = n_virtual;
  e.max_channels = n_virtual;
  e.max_detections = map_cells;
  e.max_items = map_cells;
  e.max_points = map_cells;
  e.max_features = 4;
  e.max_snapshots = snapshot_count(cfg.gather_params.snapshot_window);
  e.max_az = num_az > 0 ? num_az : 1;
  e.max_el = num_el > 0 ? num_el : 1;
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

  const auto waveform = root["waveform"].as<radar::FMCWWaveformDescriptor>();
  cfg.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  cfg.scene.waveform.bandwidth = waveform.bandwidth();
  cfg.scene.waveform.chirp_duration = waveform.chirp_duration();
  cfg.scene.waveform.sample_rate = waveform.sample_rate();
  cfg.report_waveform = waveform;

  if (!root["multiplexing_scheme"]) {
    throw std::runtime_error("fmcw_mimo: multiplexing_scheme is required");
  }
  cfg.scheme = root["multiplexing_scheme"].as<radar::MultiplexingScheme>();
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
    const YAML::Node ddm = root["ddm_dense_demultiplex"];
    if (!ddm || !ddm["ddm_code"]) {
      throw std::runtime_error(
          "fmcw_mimo: ddm_dense_demultiplex.ddm_code is required for DDM");
    }
    cfg.ddm_code = ddm["ddm_code"].as<radar::DDMCodeDescriptor>();
    if (cfg.ddm_code.num_tx != num_tx) {
      throw std::runtime_error(
          "fmcw_mimo: ddm_code.num_tx must match tx_array.num_tx");
    }
    // Demo-only: expand generate_phase_codes into explicit phase_codes so YAML
    // need not list num_tx * num_chirps complex values.
    if (ddm["ddm_code"]["generate_phase_codes"]) {
      const std::string kind =
          ddm["ddm_code"]["generate_phase_codes"].as<std::string>();
      if (kind == "zadoff_chu" || kind == "zc") {
        fill_zadoff_chu_phase_codes(&cfg.ddm_code, cfg.scene.num_chirps);
      } else if (kind == "sin") {
        fill_sin_phase_codes(&cfg.ddm_code, cfg.scene.num_chirps);
      } else {
        throw std::runtime_error(
            "fmcw_mimo: unsupported generate_phase_codes '" + kind +
            "' (supported: zadoff_chu, zc, sin)");
      }
    }
    cfg.ddm_code.validate();
    cfg.ddm_correlate = radar::detail::ddm_classify_codes(cfg.ddm_code) ==
                        radar::detail::DdmCodeMode::kNonLinear;
    if (cfg.ddm_correlate) {
      if (cfg.ddm_code.guard_bands != 0) {
        throw std::runtime_error(
            "fmcw_mimo: non-linear DDM phase_codes require guard_bands == 0");
      }
      const int64_t num_chirps_codes =
          static_cast<int64_t>(cfg.ddm_code.phase_codes.size()) /
          cfg.ddm_code.num_tx;
      if (num_chirps_codes != cfg.scene.num_chirps) {
        throw std::runtime_error(
            "fmcw_mimo: non-linear phase_codes length/num_tx must equal "
            "cube.num_chirps");
      }
      cfg.report_num_doppler = cfg.scene.num_chirps;
    } else {
      cfg.ddm_doppler_offsets =
          radar::detail::ddm_resolve_offsets(cfg.ddm_code);
      if (cfg.scene.num_chirps % num_tx != 0) {
        throw std::runtime_error(
            "fmcw_mimo: cube.num_chirps must be divisible by tx_array.num_tx "
            "for linear DDM");
      }
      const int64_t band = cfg.scene.num_chirps / num_tx;
      cfg.report_num_doppler = band - cfg.ddm_code.guard_bands;
      if (cfg.report_num_doppler <= 0) {
        throw std::runtime_error(
            "fmcw_mimo: DDM guard_bands leaves no Doppler bins per TX");
      }
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

  load_range_params(root["range_processor"], &cfg.range_params);
  load_clutter_params(root["clutter_filter"], &cfg.clutter_params);
  load_doppler_params(root["doppler_processor"], &cfg.doppler_params);
  load_deinterleave_params(root["mimo_deinterleave"],
                           &cfg.deinterleave_params);
  cfg.deinterleave_params.tx_schedule = cfg.tx_schedule;
  load_tdm_dense_compensation_params(root["tdm_dense_compensation"],
                                     &cfg.tdm_compensation_params);
  cfg.tdm_compensation_params.tx_schedule = cfg.tx_schedule;
  if (!cfg.tdm_compensation_params.waveform.is_valid()) {
    cfg.tdm_compensation_params.waveform = waveform;
  }
  load_ddm_dense_demultiplex_params(root["ddm_dense_demultiplex"],
                                    &cfg.ddm_demultiplex_params);
  cfg.ddm_demultiplex_params.ddm_code = cfg.ddm_code;
  load_vaa_params(root["virtual_array_assembly"], &cfg.vaa_params);
  cfg.vaa_params.tx_positions = cfg.tx_positions_flat;
  cfg.vaa_params.rx_positions = cfg.rx_positions_flat;
  cfg.vaa_params.wavelength = cfg.wavelength;

  load_cfar_params(root["cfar_detector"], &cfg.cfar_params);
  load_snapshot_gather_params(root["snapshot_gather"], &cfg.gather_params);
  load_beamscan_params(root["beamscan_estimator"], &cfg.beamscan_params);
  // Params-only geometry on EA (§9): seed from the virtual array built here.
  // VAA still emits a packed Tensor (drained to NullSink); live wire deferred.
  cfg.beamscan_params.array_geometry = cfg.array_geometry;
  if (!cfg.cfar_params.nci_axes.empty()) {
    cfg.cfar_params.nci_fold = cfg.array_geometry.num_elements();
  }
  load_dbscan_params(root["cluster_dbscan"], &cfg.dbscan_params);
  load_reduction_params(root["reduction"], &cfg.reduction_params);

  cfg.use_cfar_peaks =
      cfg.cfar_params.peak_mode != holoscan::radar::CfarPeakMode::kNone;
  const PhysicalAngleBounds physical_bounds = physical_scan_grid_bounds(
      cfg.beamscan_params.angle_convention, cfg.beamscan_params.az_coordinates,
      cfg.beamscan_params.el_coordinates);
  cfg.az_lo = physical_bounds.az[0];
  cfg.az_hi = physical_bounds.az[1];
  cfg.el_lo = physical_bounds.el[0];
  cfg.el_hi = physical_bounds.el[1];
  cfg.extents = make_mimo_extents(cfg);
  return cfg;
}
