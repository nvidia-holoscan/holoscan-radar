// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MIMO scene model + analytic TDM/DDM simulators for fmcw_mimo. Colocated with
// the app: multiplexing needs a bespoke simulator that does not fit the shared
// point-target simulate(SimScene) shape, so it stays local rather than in
// radar_common.

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sim/point_target_sim.hpp"

namespace radar = holoscan::radar;

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

inline radar::sim::Vec3 scale_position(double iy, double iz, double spacing_wl,
                                double wavelength) {
  return radar::sim::Vec3{
      0.0, iy * spacing_wl * wavelength, iz * spacing_wl * wavelength};
}

inline std::vector<double> flatten_positions(
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

inline std::vector<double> uniform_ddm_offsets(int64_t num_tx) {
  std::vector<double> offsets(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  return offsets;
}

inline std::vector<double> resolve_ddm_offsets(const radar::DDMCodeDescriptor& code) {
  code.validate();
  if (!code.doppler_offsets.empty()) {
    return code.doppler_offsets;
  }
  return uniform_ddm_offsets(code.num_tx);
}

inline radar::sim::SimResult simulate_mimo_tdm_interleaved(
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

inline radar::sim::SimResult simulate_mimo_ddm_coded(
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
