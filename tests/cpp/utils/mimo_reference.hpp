// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Analytic MIMO FMCW simulators (CPU) — the multi-TX generalization of the
// single-TX reference in sim_reference.hpp, sharing the same physics (far-field
// two-way delay, beat frequency, slow-time carrier phase). The three schemes —
// TDM interleaving, DDM Doppler coding, DDM phase coding — are the ground truth
// the unified GPU core (simulator.hpp::simulate_mimo_gpu)
// is validated against, and
// the tautology guard for the deinterleave / coherent-compensation /
// virtual-array operators.
//
// A tests-only reference (like sim_reference.hpp): the runtime MIMO source
// operators synthesize on the device via simulate_mimo_gpu; this lives under
// tests/ so the oracle stays implementation-independent. Builds on the shared
// scene model radar_common/sim/scene.hpp.

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::sim {

// Uniform DDM per-TX Doppler offsets: TX m rides at m/num_tx of the PRF.
inline std::vector<double> uniform_ddm_offsets(int64_t num_tx) {
  std::vector<double> offsets(static_cast<size_t>(num_tx));
  for (int64_t m = 0; m < num_tx; ++m) {
    offsets[static_cast<size_t>(m)] =
        static_cast<double>(m) / static_cast<double>(num_tx);
  }
  return offsets;
}

// Resolve a DDMCodeDescriptor to explicit per-TX Doppler offsets: its own
// doppler_offsets when set, otherwise the uniform code.
inline std::vector<double> resolve_ddm_offsets(const DDMCodeDescriptor& code) {
  code.validate();
  if (!code.doppler_offsets.empty()) {
    return code.doppler_offsets;
  }
  return uniform_ddm_offsets(code.num_tx);
}

// TDM: one TX is active per interleaved chirp slot per `schedule`, so the raw
// cube is (num_rx, num_tx * num_chirps, samples_per_chirp) along an interleaved
// slow-time axis. num_tx = scene.tx_positions.size(); the physical PRI (the
// real chirp-to-chirp spacing) is scene.waveform.chirp_repetition_interval.
inline SimResult simulate_mimo_tdm_interleaved(const SimScene& scene,
                                               const TXSchedule& schedule) {
  if (scene.tx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 TX element");
  }
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_tdm_interleaved: need >= 1 RX element");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t slow_time = num_tx * scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double physical_pri = scene.waveform.chirp_repetition_interval;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double c = scene.propagation_speed;

  TXSchedule sched = schedule;
  sched.num_tx = num_tx;
  const std::vector<int64_t> pattern = sched.expand_chirp_tx_index(slow_time);

  SimResult result;
  result.shape = {num_rx, slow_time, samples};
  result.cube.assign(static_cast<size_t>(num_rx * slow_time * samples),
                     {0.0, 0.0});

  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < slow_time; ++k) {
      const int64_t tx = pattern[static_cast<size_t>(k)];
      if (tx < 0 || tx >= num_tx) {
        throw std::runtime_error("simulate_mimo_tdm_interleaved: bad TX index");
      }
      const Vec3& tx_pos = scene.tx_positions[static_cast<size_t>(tx)];
      const double range_k =
          truth.range +
          truth.radial_velocity * static_cast<double>(k) * physical_pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const double aperture =
            dot(tx_pos, truth.look_direction) +
            dot(scene.rx_positions[static_cast<size_t>(rx)],
                truth.look_direction);
        const double tau = (2.0 * range_k - aperture) / c;
        const double beat = slope * tau;
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          const double phase = 2.0 * kPi * (beat * t) - 2.0 * kPi * f_c * tau;
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

  // Clean scatterer response; the caller
  // applies sim::add_thermal_noise (and any
  // other I/Q additives) to the cube.
  return result;
}

// DDM: all TX fire every chirp simultaneously, each carrying a per-TX Doppler
// code phase (2*pi*offset_m*k over slow time), so the cube is
// (num_rx, num_chirps, samples_per_chirp) and the TX separate along Doppler.
// `offsets` are the per-TX offsets (fraction of the PRF); length must equal
// scene.tx_positions.size().
inline SimResult simulate_mimo_ddm_coded(const SimScene& scene,
                                         const std::vector<double>& offsets) {
  if (scene.tx_positions.empty() || scene.rx_positions.empty()) {
    throw std::invalid_argument("simulate_mimo_ddm_coded: need TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
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

  SimResult result;
  result.shape = {num_rx, num_chirps, samples};
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples),
                     {0.0, 0.0});

  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < num_chirps; ++k) {
      const double range_k =
          truth.range + truth.radial_velocity * static_cast<double>(k) * pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          std::complex<double> acc{0.0, 0.0};
          for (int64_t tx = 0; tx < num_tx; ++tx) {
            const Vec3& tx_pos = scene.tx_positions[static_cast<size_t>(tx)];
            const double aperture =
                dot(tx_pos, truth.look_direction) +
                dot(scene.rx_positions[static_cast<size_t>(rx)],
                    truth.look_direction);
            const double tau = (2.0 * range_k - aperture) / c;
            const double beat = slope * tau;
            const double code_phase = 2.0 * kPi *
                                      offsets[static_cast<size_t>(tx)] *
                                      static_cast<double>(k);
            const double phase = 2.0 * kPi * (beat * t) -
                                 2.0 * kPi * f_c * tau + code_phase;
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

  // Clean scatterer response; the caller
  // applies sim::add_thermal_noise (and any
  // other I/Q additives) to the cube.
  return result;
}

// DDM with arbitrary per-(tx, chirp) unit-magnitude codes (non-linear demux).
// `phase_codes` length must be num_tx * num_chirps, row-major (tx, chirp).
inline SimResult simulate_mimo_ddm_phase_coded(
    const SimScene& scene,
    const std::vector<std::complex<double>>& phase_codes) {
  if (scene.tx_positions.empty() || scene.rx_positions.empty()) {
    throw std::invalid_argument(
        "simulate_mimo_ddm_phase_coded: need TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
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

  SimResult result;
  result.shape = {num_rx, num_chirps, samples};
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples),
                     {0.0, 0.0});

  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t k = 0; k < num_chirps; ++k) {
      const double range_k =
          truth.range + truth.radial_velocity * static_cast<double>(k) * pri;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          std::complex<double> acc{0.0, 0.0};
          for (int64_t tx = 0; tx < num_tx; ++tx) {
            const Vec3& tx_pos = scene.tx_positions[static_cast<size_t>(tx)];
            const double aperture =
                dot(tx_pos, truth.look_direction) +
                dot(scene.rx_positions[static_cast<size_t>(rx)],
                    truth.look_direction);
            const double tau = (2.0 * range_k - aperture) / c;
            const double beat = slope * tau;
            const double phase =
                2.0 * kPi * (beat * t) - 2.0 * kPi * f_c * tau;
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

  // Clean scatterer response; the caller
  // applies sim::add_thermal_noise (and any
  // other I/Q additives) to the cube.
  return result;
}

}  // namespace holoscan::radar::sim
