// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MIMO FMCW sparse angle-then-cluster pipeline (TDM or DDM via YAML):
//
// TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
//        -> Compensation -> VirtualArrayAssemblyOp -> ...
// DDM: MimoDdmAdcSourceOp -> Range -> Doppler -> Compensation -> Clutter
//        -> VirtualArrayAssemblyOp -> ...
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_mimo/cpp/fmcw_mimo [config.yaml]

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sdk/yaml_converters.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

struct PlotsConfig {
  bool enable{true};
  int64_t window{120};
  // Optional fixed velocity-plot Y extent (m/s). NaN keeps the full-axis span.
  float velocity_ylim_lo{std::numeric_limits<float>::quiet_NaN()};
  float velocity_ylim_hi{std::numeric_limits<float>::quiet_NaN()};
};

struct MimoScene {
  std::vector<radar::sim::PointTarget> targets;
  radar::sim::SimWaveform waveform{};
  int64_t num_tx{2};
  int64_t num_chirps{64};
  int64_t samples_per_chirp{256};
  std::vector<radar::sim::Vec3> tx_positions;
  std::vector<radar::sim::Vec3> rx_positions;
  double propagation_speed{radar::sim::kSimSpeedOfLight};
  std::optional<double> snr_db{};
  // Fixed RNG seed. The ADC sources reseed with this on every frame, so the
  // additive-noise realization is deliberately identical frame to frame: the
  // demo trades frozen-speckle realism for reproducibility (deterministic
  // clusters and screenshots). Accepted 2026-07-16 as fine for a demo; a
  // frame-varying scene run should fold the frame index into the seed.
  uint64_t seed{0x5eed5eedULL};
};

struct AppConfig {
  radar::MultiplexingScheme scheme{radar::MultiplexingScheme::kTdm};
  MimoScene scene;
  radar::TXSchedule tx_schedule;
  radar::DDMCodeDescriptor ddm_code;
  std::vector<double> ddm_doppler_offsets;
  bool ddm_correlate{false};  // non-linear phase_codes → correlate demux
  radar::ArrayGeometry array_geometry;
  std::vector<double> tx_positions_flat;
  std::vector<double> rx_positions_flat;
  double wavelength{0.0};
  PlotsConfig plots;
  int64_t num_frames{1};
  double frame_period{0.1};
  bool meter_cadence{true};
  int64_t meter_log_every{0};
  bool visualize{false};
  bool skip_display_check{false};
  bool headless{false};
  double az_lo{-0.35}, az_hi{0.35}, el_lo{-0.1}, el_hi{0.1};
  radar::FMCWWaveformDescriptor report_waveform;
  int64_t report_num_doppler{0};
};

radar::sim::Vec3 scale_position(double iy, double iz, double spacing_wl,
                                double wavelength) {
  return radar::sim::Vec3{
      0.0, iy * spacing_wl * wavelength, iz * spacing_wl * wavelength};
}

std::vector<double> flatten_positions(
    const std::vector<radar::sim::Vec3>& positions) {
  std::vector<double> flat;
  flat.reserve(positions.size() * 3);
  for (const radar::sim::Vec3& p : positions) {
    flat.push_back(p[0]);
    flat.push_back(p[1]);
    flat.push_back(p[2]);
  }
  return flat;
}

radar::ArrayGeometry make_virtual_geometry(
    const std::vector<radar::sim::Vec3>& tx_positions,
    const std::vector<radar::sim::Vec3>& rx_positions, double wavelength) {
  radar::ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = radar::ArrayGeometryType::kConformal;
  const int64_t num_tx = static_cast<int64_t>(tx_positions.size());
  const int64_t num_rx = static_cast<int64_t>(rx_positions.size());
  geom.element_positions.reserve(
      static_cast<size_t>(num_tx * num_rx * 3));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const radar::sim::Vec3& tp = tx_positions[static_cast<size_t>(tx)];
      const radar::sim::Vec3& rp = rx_positions[static_cast<size_t>(rx)];
      geom.element_positions.push_back(tp[0] + rp[0]);
      geom.element_positions.push_back(tp[1] + rp[1]);
      geom.element_positions.push_back(tp[2] + rp[2]);
    }
  }
  geom.validate();
  return geom;
}

void advance_scene(MimoScene* scene, double dt) {
  for (radar::sim::PointTarget& target : scene->targets) {
    for (int axis = 0; axis < 3; ++axis) {
      target.position[axis] += target.velocity[axis] * dt;
    }
  }
}

// Demo helpers for generate_phase_codes: expand unit-magnitude codes without
// dumping N*M complex values into YAML. Codes must be *non-linear* so the
// operator takes the correlate path (linear-phase ramps reduce to gather).

// High-crosstalk recipe (matches unit-test / perf NL codes). Useful to stress
// demux leakage; not for a clean Holoviz scene.
void fill_sin_phase_codes(radar::DDMCodeDescriptor* code, int64_t num_chirps) {
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

// Low-crosstalk non-linear codes: distinct Zadoff-Chu roots (coprime to N).
// Quadratic phase fails the linear-phase classifier; distinct roots keep
// periodic cross-correlation low so CFAR sees the physical targets.
void fill_zadoff_chu_phase_codes(radar::DDMCodeDescriptor* code,
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
  constexpr double kPi = radar::kPi;
  for (int64_t m = 0; m < code->num_tx; ++m) {
    const double u = static_cast<double>(roots[static_cast<size_t>(m)]);
    for (int64_t k = 0; k < num_chirps; ++k) {
      const double kd = static_cast<double>(k);
      const double arg =
          odd_n ? (-kPi * u * kd * (kd + 1.0) /
                   static_cast<double>(num_chirps))
                : (-kPi * u * kd * kd / static_cast<double>(num_chirps));
      code->phase_codes[static_cast<size_t>(m * num_chirps + k)] =
          std::complex<double>(std::cos(arg), std::sin(arg));
    }
  }
}

radar::sim::SimResult simulate_mimo_tdm_interleaved(
    const MimoScene& scene, const radar::TXSchedule& schedule) {
  if (scene.tx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 TX element");
  }
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 RX element");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = scene.num_tx;
  const int64_t slow_time = num_tx * scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double physical_pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  radar::TXSchedule sched = schedule;
  sched.num_tx = num_tx;
  const std::vector<int64_t> pattern = sched.expand_chirp_tx_index(slow_time);

  radar::sim::SimResult result;
  result.shape = {num_rx, slow_time, samples};
  result.cube.assign(static_cast<size_t>(num_rx * slow_time * samples),
                     {0.0, 0.0});

  for (const radar::sim::PointTarget& target : scene.targets) {
    const radar::sim::TargetTruth truth = radar::sim::derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < slow_time; ++k) {
      const int64_t tx =
          pattern[static_cast<size_t>(k)];
      if (tx < 0 || tx >= num_tx) {
        throw std::runtime_error("simulate_mimo_tdm_interleaved: bad TX index");
      }
      const radar::sim::Vec3& tx_pos =
          scene.tx_positions[static_cast<size_t>(tx)];
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * physical_pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const double aperture =
            radar::sim::dot(tx_pos, truth.look_direction) +
            radar::sim::dot(scene.rx_positions[static_cast<size_t>(rx)],
                            truth.look_direction);
        const double tau = (2.0 * range_k - aperture) / c;
        const double beat = slope * tau;
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          const double phase =
              2.0 * radar::kPi * (beat * t) - 2.0 * radar::kPi * f_c * tau;
          const std::complex<double> sample =
              target.amplitude *
              std::complex<double>(std::cos(phase), std::sin(phase));
          const size_t idx =
              static_cast<size_t>((rx * slow_time + k) * samples + s);
          result.cube[idx] += sample;
        }
      }
    }
  }

  if (scene.snr_db.has_value()) {
    const double noise_power = std::pow(10.0, -(*scene.snr_db) / 10.0);
    const double sigma = std::sqrt(noise_power / 2.0);
    std::mt19937_64 rng(scene.seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (auto& v : result.cube) {
      v += std::complex<double>(gauss(rng), gauss(rng));
    }
  }
  return result;
}

radar::sim::SimResult simulate_mimo_ddm_coded(
    const MimoScene& scene, const std::vector<double>& offsets) {
  if (scene.tx_positions.empty() || scene.rx_positions.empty()) {
    throw std::invalid_argument("simulate_mimo_ddm_coded: need TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = scene.num_tx;
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  if (static_cast<int64_t>(offsets.size()) != num_tx) {
    throw std::invalid_argument(
        "simulate_mimo_ddm_coded: offsets length must equal num_tx");
  }

  radar::sim::SimResult result;
  result.shape = {num_rx, num_chirps, samples};
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples),
                     {0.0, 0.0});

  for (const radar::sim::PointTarget& target : scene.targets) {
    const radar::sim::TargetTruth truth = radar::sim::derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < num_chirps; ++k) {
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          std::complex<double> acc{0.0, 0.0};
          for (int64_t tx = 0; tx < num_tx; ++tx) {
            const radar::sim::Vec3& tx_pos =
                scene.tx_positions[static_cast<size_t>(tx)];
            const double aperture =
                radar::sim::dot(tx_pos, truth.look_direction) +
                radar::sim::dot(scene.rx_positions[static_cast<size_t>(rx)],
                                truth.look_direction);
            const double tau = (2.0 * range_k - aperture) / c;
            const double beat = slope * tau;
            const double code_phase =
                2.0 * radar::kPi * offsets[static_cast<size_t>(tx)] *
                static_cast<double>(k);
            const double phase =
                2.0 * radar::kPi * (beat * t) - 2.0 * radar::kPi * f_c * tau +
                code_phase;
            acc += target.amplitude *
                   std::complex<double>(std::cos(phase), std::sin(phase));
          }
          const size_t idx =
              static_cast<size_t>((rx * num_chirps + k) * samples + s);
          result.cube[idx] += acc;
        }
      }
    }
  }

  if (scene.snr_db.has_value()) {
    const double noise_power = std::pow(10.0, -(*scene.snr_db) / 10.0);
    const double sigma = std::sqrt(noise_power / 2.0);
    std::mt19937_64 rng(scene.seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (auto& v : result.cube) {
      v += std::complex<double>(gauss(rng), gauss(rng));
    }
  }
  return result;
}

// DDM with arbitrary per-(tx, chirp) unit-magnitude codes (non-linear demux).
radar::sim::SimResult simulate_mimo_ddm_phase_coded(
    const MimoScene& scene,
    const std::vector<std::complex<double>>& phase_codes) {
  if (scene.tx_positions.empty() || scene.rx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_ddm_phase_coded: need TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = scene.num_tx;
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  if (static_cast<int64_t>(phase_codes.size()) != num_tx * num_chirps) {
    throw std::invalid_argument(
        "simulate_mimo_ddm_phase_coded: phase_codes length must be "
        "num_tx * num_chirps");
  }

  radar::sim::SimResult result;
  result.shape = {num_rx, num_chirps, samples};
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples),
                     {0.0, 0.0});

  for (const radar::sim::PointTarget& target : scene.targets) {
    const radar::sim::TargetTruth truth = radar::sim::derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < num_chirps; ++k) {
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          std::complex<double> acc{0.0, 0.0};
          for (int64_t tx = 0; tx < num_tx; ++tx) {
            const radar::sim::Vec3& tx_pos =
                scene.tx_positions[static_cast<size_t>(tx)];
            const double aperture =
                radar::sim::dot(tx_pos, truth.look_direction) +
                radar::sim::dot(scene.rx_positions[static_cast<size_t>(rx)],
                                truth.look_direction);
            const double tau = (2.0 * range_k - aperture) / c;
            const double beat = slope * tau;
            const double phase =
                2.0 * radar::kPi * (beat * t) - 2.0 * radar::kPi * f_c * tau;
            const auto code =
                phase_codes[static_cast<size_t>(tx * num_chirps + k)];
            acc += target.amplitude * code *
                   std::complex<double>(std::cos(phase), std::sin(phase));
          }
          const size_t idx =
              static_cast<size_t>((rx * num_chirps + k) * samples + s);
          result.cube[idx] += acc;
        }
      }
    }
  }

  if (scene.snr_db.has_value()) {
    const double noise_power = std::pow(10.0, -(*scene.snr_db) / 10.0);
    const double sigma = std::sqrt(noise_power / 2.0);
    std::mt19937_64 rng(scene.seed);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (auto& v : result.cube) {
      v += std::complex<double>(gauss(rng), gauss(rng));
    }
  }
  return result;
}

void load_scene_geometry(AppConfig* cfg, const YAML::Node& root,
                         double wavelength) {
  const YAML::Node tx = root["tx_array"];
  const double tx_spacing = tx["spacing"].as<double>();
  cfg->scene.tx_positions.clear();
  cfg->scene.tx_positions.reserve(static_cast<size_t>(cfg->scene.num_tx));
  for (int64_t itx = 0; itx < cfg->scene.num_tx; ++itx) {
    cfg->scene.tx_positions.push_back(radar::sim::Vec3{
        0.0, static_cast<double>(itx) * tx_spacing * wavelength, 0.0});
  }

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
  cfg->scene.rx_positions.clear();
  cfg->scene.rx_positions.reserve(
      static_cast<size_t>(az_elements * el_elements));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      cfg->scene.rx_positions.push_back(
          scale_position(static_cast<double>(iy), static_cast<double>(iz),
                         rx_spacing, wavelength));
    }
  }

  cfg->array_geometry = make_virtual_geometry(
      cfg->scene.tx_positions, cfg->scene.rx_positions, wavelength);
}

AppConfig load_config(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  const YAML::Node root = YAML::LoadFile(path.string());
  AppConfig cfg;

  const auto waveform =
      root["waveform"].as<radar::FMCWWaveformDescriptor>();
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

  cfg.num_frames = root["run"]["num_frames"].as<int64_t>();
  cfg.frame_period = root["run"]["frame_period"].as<double>();
  if (root["run"]["meter_cadence"]) {
    cfg.meter_cadence = root["run"]["meter_cadence"].as<bool>();
  }
  if (root["run"]["meter_log_every"]) {
    cfg.meter_log_every = root["run"]["meter_log_every"].as<int64_t>();
  }
  if (cfg.meter_log_every < 0) {
    throw std::runtime_error("run.meter_log_every must be >= 0");
  }
  cfg.scene.samples_per_chirp = root["cube"]["samples_per_chirp"].as<int64_t>();

  const YAML::Node tx = root["tx_array"];
  if (!tx) {
    throw std::runtime_error("tx_array section is required for MIMO");
  }
  cfg.scene.num_tx = tx["num_tx"].as<int64_t>();
  if (cfg.scene.num_tx < 1) {
    throw std::runtime_error("tx_array.num_tx must be >= 1");
  }

  const double wavelength = cfg.scene.waveform.wavelength();
  cfg.wavelength = wavelength;
  load_scene_geometry(&cfg, root, wavelength);

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
    if (cfg.ddm_code.num_tx != cfg.scene.num_tx) {
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
    cfg.ddm_correlate =
        radar::detail::ddm_classify_codes(cfg.ddm_code) ==
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
      if (cfg.scene.num_chirps % cfg.scene.num_tx != 0) {
        throw std::runtime_error(
            "fmcw_mimo: cube.num_chirps must be divisible by tx_array.num_tx "
            "for linear DDM");
      }
      const int64_t band = cfg.scene.num_chirps / cfg.scene.num_tx;
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
    cfg.tx_schedule = root["mimo_deinterleave"]["tx_schedule"]
                          .as<radar::TXSchedule>();
    if (cfg.tx_schedule.num_tx != cfg.scene.num_tx) {
      throw std::runtime_error(
          "mimo_deinterleave.tx_schedule.num_tx must match tx_array.num_tx");
    }
    cfg.report_num_doppler = cfg.scene.num_chirps;
  }

  cfg.tx_positions_flat = flatten_positions(cfg.scene.tx_positions);
  cfg.rx_positions_flat = flatten_positions(cfg.scene.rx_positions);

  for (const YAML::Node& t : root["targets"]) {
    const double range = t["range"].as<double>();
    const double azimuth = t["azimuth"].as<double>();
    const double elevation = t["elevation"].as<double>();
    const double amplitude = t["amplitude"].as<double>();
    const YAML::Node vel = t["velocity"];
    if (!vel || !vel.IsSequence() || vel.size() != 3) {
      throw std::runtime_error(
          "fmcw_mimo: each target's velocity must be a 3-vector [vx, vy, vz]");
    }
    const radar::sim::Vec3 dir{std::cos(elevation) * std::cos(azimuth),
                               std::cos(elevation) * std::sin(azimuth),
                               std::sin(elevation)};
    cfg.scene.targets.push_back(
        {radar::sim::Vec3{range * dir[0], range * dir[1], range * dir[2]},
         radar::sim::Vec3{vel[0].as<double>(), vel[1].as<double>(),
                          vel[2].as<double>()},
         {amplitude, 0.0}});
  }

  const YAML::Node noise = root["noise_snr_db"];
  if (noise && !noise.IsNull()) {
    cfg.scene.snr_db = noise.as<double>();
  }
  if (root["visualize"]) {
    cfg.visualize = root["visualize"].as<bool>();
  }
  if (root["skip_display_check"]) {
    cfg.skip_display_check = root["skip_display_check"].as<bool>();
  }
  if (root["headless"]) {
    cfg.headless = root["headless"].as<bool>();
  }
  if (const YAML::Node plots = root["plots"]) {
    if (plots["enable"]) {
      cfg.plots.enable = plots["enable"].as<bool>();
    }
    if (plots["window"]) {
      cfg.plots.window = plots["window"].as<int64_t>();
    }
    if (const YAML::Node ylim = plots["velocity_ylim"]) {
      if (!ylim.IsSequence() || ylim.size() != 2) {
        throw std::runtime_error(
            "plots.velocity_ylim must be a [lo, hi] pair");
      }
      cfg.plots.velocity_ylim_lo = ylim[0].as<float>();
      cfg.plots.velocity_ylim_hi = ylim[1].as<float>();
    }
  }
  if (cfg.plots.window <= 1) {
    throw std::runtime_error("plots.window must be > 1");
  }

  const auto grid_extent = [](const YAML::Node& grid, double def_lo,
                              double def_hi, double* lo, double* hi) {
    if (grid && grid.IsSequence() && grid.size() > 0) {
      double mn = std::numeric_limits<double>::infinity();
      double mx = -std::numeric_limits<double>::infinity();
      for (const YAML::Node& v : grid) {
        const double x = v.as<double>();
        mn = std::min(mn, x);
        mx = std::max(mx, x);
      }
      *lo = mn;
      *hi = mx;
    } else {
      *lo = def_lo;
      *hi = def_hi;
    }
    if (*hi - *lo < 1e-6) {
      const double mid = 0.5 * (*lo + *hi);
      *lo = mid - 0.1;
      *hi = mid + 0.1;
    }
  };
  const YAML::Node be = root["beamscan_estimator"];
  grid_extent(be ? be["az_angles"] : YAML::Node(), -0.35, 0.35, &cfg.az_lo,
              &cfg.az_hi);
  grid_extent(be ? be["el_angles"] : YAML::Node(), -0.1, 0.1, &cfg.el_lo,
              &cfg.el_hi);
  return cfg;
}

// Post-DDM-demux velocity axis for reporting: CFAR / feature assembly run on
// VirtualArrayAssemblyOp rd_maps. Linear DDM keeps W = N/num_tx - guard bins at
// spacing PRF/N; non-linear correlate keeps the full N-bin Doppler axis at the
// same spacing.
class DdmReportVelocityBinsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DdmReportVelocityBinsOp)
  DdmReportVelocityBinsOp() = default;

  radar::FMCWWaveformDescriptor waveform;
  int64_t num_doppler{0};
  // The DDM demux keeps the full slow-time FFT bin resolution PRF/N (N = the
  // pre-demux Doppler length = num_chirps) but retains only W = N/num_tx minus
  // guard bins, re-centered. So the demuxed velocity axis has W bins spaced at
  // PRF/N (NOT PRF/W); using PRF/W would inflate velocities by ~num_tx and
  // fragment each target across Doppler in clustering.
  double bin_spacing_hz{0.0};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("sync");
    spec.output<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)radar::receive_single(op_input, "sync", "DdmReportVelocityBinsOp");
    // The axis is constant per run; build and upload it once, then re-emit the
    // cached device tensor each tick (the "sync" input just paces the emit).
    if (!velocity_bins_) {
      velocity_bins_ = build_axis();
    }
    holoscan::TensorMap out;
    out.insert({"velocity_bins", velocity_bins_});
    op_output.emit(out, "velocity_bins");
  }

 private:
  std::shared_ptr<holoscan::Tensor> build_axis() const {
    if (num_doppler <= 0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: num_doppler must be positive");
    }
    if (!waveform.is_valid()) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: waveform descriptor is invalid");
    }
    if (bin_spacing_hz <= 0.0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: bin_spacing_hz must be > 0");
    }
    const double wavelength = waveform.wavelength();
    // Center-zero axis: bin i maps to Doppler frequency (i - W/2) * df.
    std::vector<float> axis(static_cast<size_t>(num_doppler));
    for (int64_t i = 0; i < num_doppler; ++i) {
      const double f =
          static_cast<double>(i - num_doppler / 2) * bin_spacing_hz;
      axis[static_cast<size_t>(i)] =
          static_cast<float>(-f * wavelength / 2.0);
    }
    auto tensor = radar::make_device_tensor({num_doppler},
                                            radar::real_dtype<float>());
    if (cudaMemcpy(tensor->data(), axis.data(), sizeof(float) * axis.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: velocity_bins H2D upload failed");
    }
    return tensor;
  }

  std::shared_ptr<holoscan::Tensor> velocity_bins_;
};

class MimoTdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoTdmAdcSourceOp)
  MimoTdmAdcSourceOp() = default;

  MimoScene scene;
  radar::TXSchedule tx_schedule;
  double frame_period{0.1};
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const radar::sim::SimResult sim =
        simulate_mimo_tdm_interleaved(scene, tx_schedule);
    if (sim.shape.size() != 3) {
      throw std::runtime_error(
          "MimoTdmAdcSourceOp: expected rank-3 (N_rx, M*L, samples)");
    }

    std::vector<std::complex<float>> cube(sim.cube.size());
    for (size_t i = 0; i < sim.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(sim.cube[i].real()),
                                    static_cast<float>(sim.cube[i].imag()));
    }

    auto tensor =
        radar::make_device_tensor(sim.shape, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("MimoTdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    advance_scene(&scene, frame_period);
  }
};

class MimoDdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoDdmAdcSourceOp)
  MimoDdmAdcSourceOp() = default;

  MimoScene scene;
  std::vector<double> doppler_offsets;
  std::vector<std::complex<double>> phase_codes;
  bool use_phase_codes{false};
  double frame_period{0.1};
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    const radar::sim::SimResult sim =
        use_phase_codes ? simulate_mimo_ddm_phase_coded(scene, phase_codes)
                        : simulate_mimo_ddm_coded(scene, doppler_offsets);
    if (sim.shape.size() != 3) {
      throw std::runtime_error(
          "MimoDdmAdcSourceOp: expected rank-3 (N_rx, num_chirps, samples)");
    }

    std::vector<std::complex<float>> cube(sim.cube.size());
    for (size_t i = 0; i < sim.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(sim.cube[i].real()),
                                    static_cast<float>(sim.cube[i].imag()));
    }

    auto tensor =
        radar::make_device_tensor(sim.shape, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("MimoDdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    advance_scene(&scene, frame_period);
  }
};

class TensorMapDiscardOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorMapDiscardOp)
  TensorMapDiscardOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("input");
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)radar::receive_single(op_input, "input", "TensorMapDiscardOp");
  }
};

}  // namespace

class FmcwMimoApp : public holoscan::Application {
 public:
  AppConfig cfg;
  std::shared_ptr<CadenceMeter> cadence;

  void compose() override {
    const bool ddm = cfg.scheme == radar::MultiplexingScheme::kDdm;
    const std::string window_title =
        ddm ? "FMCW MIMO DDM - virtual RD + clusters"
            : "FMCW MIMO TDM - virtual RD + clusters";

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto compensation =
        make_operator<radar::MIMOCoherentCompensationOp<float>>(
            "mimo_coherent_compensation",
            from_config("mimo_coherent_compensation"),
            // Prefer the app-resolved descriptor (may expand
            // generate_phase_codes: sin into explicit phase_codes).
            holoscan::Arg("ddm_code") = cfg.ddm_code);
    auto vaa = make_operator<radar::VirtualArrayAssemblyOp<float>>(
        "virtual_array_assembly", from_config("virtual_array_assembly"),
        holoscan::Arg("tx_positions") = cfg.tx_positions_flat,
        holoscan::Arg("rx_positions") = cfg.rx_positions_flat,
        holoscan::Arg("wavelength") = cfg.wavelength);

    if (cfg.meter_cadence) {
      cadence = std::make_shared<CadenceMeter>();
      cadence->frame_period_s = cfg.frame_period;
      cadence->log_every = cfg.meter_log_every;
    }

    if (ddm) {
      auto source = make_operator<MimoDdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->use_phase_codes = cfg.ddm_correlate;
      if (cfg.ddm_correlate) {
        source->phase_codes = cfg.ddm_code.phase_codes;
      } else {
        source->doppler_offsets = cfg.ddm_doppler_offsets;
      }
      source->frame_period = cfg.frame_period;
      if (cadence) {
        source->cadence = cadence;
      }

      add_flow(source, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, clutter, {{"rd_maps", "range_profiles"}});
      add_flow(clutter, vaa, {{"range_profiles", "rd_maps"}});
    } else {
      auto source = make_operator<MimoTdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->tx_schedule = cfg.tx_schedule;
      source->frame_period = cfg.frame_period;
      if (cadence) {
        source->cadence = cadence;
      }

      auto deinterleave = make_operator<radar::MIMODeinterleaveOp<float>>(
          "mimo_deinterleave", from_config("mimo_deinterleave"));

      add_flow(source, deinterleave, {{"adc_cube", "adc_cube"}});
      add_flow(deinterleave, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
      add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, vaa, {{"rd_maps", "rd_maps"}});
    }

    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"));
    auto gather = make_operator<radar::SnapshotGatherOp<float>>(
        "snapshot_gather", from_config("snapshot_gather"));
    auto beamscan = make_operator<radar::BeamscanEstimatorOp<float>>(
        "beamscan_estimator", from_config("beamscan_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);

    auto det_report = make_operator<DetectionReportOp>("detection_report");
    if (cadence) {
      det_report->cadence = cadence;
    }
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");
    auto spectrum_sink =
        make_operator<TensorMapDiscardOp>("angle_spectrum_discard");

    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
        radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly", holoscan::Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
    auto reduction = make_operator<radar::DetectionReductionOp<float>>(
        "detection_reduction", from_config("reduction"));

    std::shared_ptr<DdmReportVelocityBinsOp> ddm_velocity_bins;
    if (ddm) {
      ddm_velocity_bins =
          make_operator<DdmReportVelocityBinsOp>("ddm_velocity_bins");
      ddm_velocity_bins->waveform = cfg.report_waveform;
      ddm_velocity_bins->num_doppler = cfg.report_num_doppler;
      ddm_velocity_bins->bin_spacing_hz =
          (1.0 / cfg.report_waveform.chirp_repetition_interval()) /
          static_cast<double>(cfg.scene.num_chirps);
      add_flow(range, ddm_velocity_bins, {{"range_bins", "sync"}});
    }

    add_flow(vaa, cfar, {{"rd_maps", "rd_maps"}});
    add_flow(vaa, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});
    add_flow(gather, beamscan, {{"snapshot", "snapshot"},
                                {"is_valid", "is_valid"}});
    add_flow(vaa, beamscan, {{"array_geometry", "array_geometry"}});
    add_flow(beamscan, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

    add_flow(cfar, det_report, {{"detection_idx", "detection_idx"},
                                  {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, det_report,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    }
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, feature_assembly,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, feature_assembly,
               {{"velocity_bins", "velocity_bins"}});
    }
    add_flow(feature_assembly, cluster, {{"points", "points"}});
    add_flow(feature_assembly, cluster_report, {{"points", "points"}});
    add_flow(cluster, cluster_report, {{"labels", "labels"},
                                      {"sizes", "sizes"}});
    add_flow(cluster, reduction, {{"labels", "labels"}});
    add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
    add_flow(feature_assembly, reduction, {{"points", "points"}});
    add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
    add_flow(cfar, cluster_report, {{"detection_idx", "detection_idx"}});
    add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, cluster_report,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, cluster_report,
               {{"velocity_bins", "velocity_bins"}});
    }

    if (cfg.visualize) {
      using InputSpec = holoscan::ops::HolovizOp::InputSpec;
      using InputType = holoscan::ops::HolovizOp::InputType;
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      std::vector<InputSpec> input_specs;
      input_specs.emplace_back("rd_map", InputType::COLOR_LUT);
      const auto palette = cluster_palette();
      const auto add_crosses = [&](const std::string& name,
                                   const std::vector<float>& color) {
        InputSpec spec(name, InputType::CROSSES);
        spec.color_ = color;
        spec.line_width_ = 2.0F;
        spec.priority_ = 1;
        input_specs.push_back(spec);
      };
      for (int c = 0; c < kNumClusterColors; ++c) {
        add_crosses(cluster_layer_name(c), palette[static_cast<size_t>(c)]);
      }
      add_crosses(kNoiseLayerName, {0.6F, 0.6F, 0.6F, 1.0F});

      const auto make_view = [](float ox, float oy, float w, float h) {
        InputSpec::View v;
        v.offset_x_ = ox;
        v.offset_y_ = oy;
        v.width_ = w;
        v.height_ = h;
        return v;
      };
      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        const auto left = make_view(0.0F, 0.0F, 0.60F, 1.0F);
        for (InputSpec& spec : input_specs) {
          spec.views_ = {left};
        }
        const auto add_plot = [&](const char* img_name, const char* title_name,
                                  const std::string& title, float oy,
                                  float h) {
          InputSpec image(img_name, InputType::COLOR);
          image.views_ = {make_view(0.63F, oy, 0.35F, h)};
          input_specs.push_back(image);
          InputSpec text(title_name, InputType::TEXT);
          text.text_ = {title};
          text.color_ = {1.0F, 1.0F, 1.0F, 1.0F};
          text.priority_ = 2;
          text.views_ = {make_view(0.63F, oy, 0.35F, h)};
          input_specs.push_back(text);
        };
        constexpr float kPanelH = 0.30F;
        add_plot(kRangePlotLayer, kRangeTitleLayer, "Range (m) vs time (s)",
                 0.02F, kPanelH);
        add_plot(kVelocityPlotLayer, kVelocityTitleLayer,
                 "Velocity (m/s) vs time (s)", 0.35F, kPanelH);
        add_plot(kBoresightPlotLayer, kBoresightTitleLayer,
                 "Boresight az/el (deg)", 0.68F, kPanelH);
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        plot->velocity_ylim_lo = cfg.plots.velocity_ylim_lo;
        plot->velocity_ylim_hi = cfg.plots.velocity_ylim_hi;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
      }

      auto visualizer = make_operator<holoscan::ops::HolovizOp>(
          "holoviz", holoscan::Arg("tensors", input_specs),
          holoscan::Arg("color_lut", make_turbo_lut()),
          holoscan::Arg("window_title", window_title),
          holoscan::Arg("width", cfg.plots.enable ? 1200U : 800U),
          holoscan::Arg("height", cfg.plots.enable ? 760U : 600U),
          holoscan::Arg("headless", cfg.headless));

      add_flow(vaa, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        if (ddm) {
          add_flow(ddm_velocity_bins, plot,
                   {{"velocity_bins", "velocity_bins"}});
        } else {
          add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        }
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
      if (boresight) {
        add_flow(beamscan, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(cluster, boresight, {{"labels", "labels"}});
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
      }
    }
  }
};

std::filesystem::path resolve_config_path(int argc, char** argv) {
  const auto bin_dir = std::filesystem::path(argv[0]).parent_path();
  if (argc > 1) {
    std::filesystem::path given(argv[1]);
    if (std::filesystem::is_regular_file(given)) {
      return given;
    }
    // Bare names like fmcw_mimo_ddm_phase_codes.yaml: try next to the binary
    // (same place CMake stages the packaged YAMLs).
    const auto beside_bin = bin_dir / given.filename();
    if (std::filesystem::is_regular_file(beside_bin)) {
      return beside_bin;
    }
    return given;  // load_config will report the original path
  }
  return bin_dir / "fmcw_mimo.yaml";
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<FmcwMimoApp>();
  try {
    const std::filesystem::path config_path =
        resolve_config_path(argc, argv);
    app->cfg = load_config(config_path);
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "fmcw_mimo: visualize enabled but no display; running without GUI");
      app->cfg.visualize = false;
    }
    app->config(config_path.string());
    app->run();
    if (app->cadence) {
      app->cadence->log_summary();
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("fmcw_mimo: {}", e.what());
    return 1;
  }
  return 0;
}
