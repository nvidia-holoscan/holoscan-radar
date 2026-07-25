// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-owned configuration for fmcw_mimo: AppConfig plus the scene-geometry and
// YAML loaders (TDM/DDM branch, virtual array geometry, targets, the
// boresight-axis mirror of the beamscan grid). Colocated with the app. The
// scene is the shared radar::sim::SimScene (multi-TX); the source operators
// synthesize the multiplexed cube on the device (sim_gpu.hpp::simulate_mimo_gpu).

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

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_app/app/config.hpp"

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
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
  radar::FMCWWaveformDescriptor report_waveform;
  int64_t report_num_doppler{0};
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
    // Demo-only: expand generate_phase_codes into explicit phase_codes so YAML
    // need not list num_tx * num_chirps complex values.
    if (mcc["ddm_code"]["generate_phase_codes"]) {
      const std::string kind =
          mcc["ddm_code"]["generate_phase_codes"].as<std::string>();
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
  // READ-ONLY MIRROR of the cfar_detector operator block: peak_only stays a
  // native CFAR param (applied via from_config); the app reads it here only to
  // decide the graph shape (drop clustering when set).
  if (const YAML::Node cd = root["cfar_detector"];
      cd && cd["peak_only"] && !cd["peak_only"].IsNull()) {
    cfg.peak_only = cd["peak_only"].as<bool>();
  }

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
