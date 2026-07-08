// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Analytic point-target FMCW signal simulator. A signal model, not an EM
// solver: each target's
// closed-form contribution is summed over (rx, chirp, fast-time sample),
// then noise is added.
//
// A shared radar_common utility: applications use it to synthesize input
// cubes, and tests use it as the tautology guard (asserting operator outputs
// against its analytic ground truth). Deliberately independent of the operator
// implementations (STL-only; no operator or MatX includes) so that the ground
// truth it produces is derived from the signal model, not from the code under
// test.
//
// Modeling assumptions: stop-and-hop (target frozen within a
// chirp, advances between chirps), ego motion folded into target relative
// velocity by the caller, far-field plane wave, monostatic, single-slope
// sweeps, ideal hardware. Aliasing folds naturally through the phase
// arithmetic. Fixed seed → bit-identical output.
//
// Sign conventions (must match the operators' locked semantics):
//  - Dechirped sample: s(k, t) = A * exp(+j*2*pi*f_b(k)*t - j*2*pi*f_c*tau_k)
//    with beat frequency f_b(k) = slope * tau_k, so a forward FFT along
//    fast-time peaks at the positive beat frequency (bin 0 = zero range).
//  - The slow-time carrier phase -2*pi*f_c*tau_k advances by
//    2*pi*f_d*PRI per chirp with f_d = -2*v_r/lambda, so radial velocity
//    v_r (positive = receding) satisfies the contract v = -f_d*lambda/2.
//  - Baseband reference chirp: r(t) = exp(+j*pi*slope*t^2). The pre-mixer
//    output is s(k, t) * r(t); hardware/software dechirp is multiplication
//    by conj(r). Operators synthesizing a dechirp reference must use this
//    same convention.

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace holoscan::radar::sim {

inline constexpr double kSimPi = 3.14159265358979323846;
inline constexpr double kSimSpeedOfLight = 299792458.0;

using Vec3 = std::array<double, 3>;

inline double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

struct PointTarget {
  Vec3 position{};                        // m, array-local Cartesian frame
  Vec3 velocity{};                        // m/s, relative (ego motion folded)
  std::complex<double> amplitude{1.0, 0.0};
};

// Analytic waveform fields (mirrors FMCWWaveformDescriptor).
struct SimWaveform {
  double carrier_frequency{77e9};          // Hz
  double bandwidth{150e6};                 // Hz (slope = bandwidth/duration)
  double chirp_duration{20e-6};            // s
  double sample_rate{12.8e6};              // Hz
  double chirp_repetition_interval{25e-6};  // s (PRI)

  double slope() const { return bandwidth / chirp_duration; }
  double wavelength(double c = kSimSpeedOfLight) const {
    return c / carrier_frequency;
  }
};

struct SimScene {
  std::vector<PointTarget> targets;
  SimWaveform waveform{};
  int64_t num_chirps{64};
  int64_t samples_per_chirp{256};
  // Element positions (m, array-local frame). Single TX for SISO/SIMO; the
  // API is shaped so TDM/DDM multiplexing can be added later.
  Vec3 tx_position{0.0, 0.0, 0.0};
  std::vector<Vec3> rx_positions{Vec3{0.0, 0.0, 0.0}};
  double propagation_speed{kSimSpeedOfLight};
  // Additive complex white Gaussian noise, relative to a unit-amplitude
  // target (SNR in dB). Unset → noise-free.
  std::optional<double> snr_db{};
  uint64_t seed{0x5eed5eedULL};
  // true → dechirped (IF) cube; false → pre-mixer cube (for apply_dechirp
  // tests): pre_mixer(k, t) = dechirped(k, t) * reference(t).
  bool dechirped{true};
};

// Per-target derived physical ground truth (tests assert in the
// physical domain, not on bin indices).
struct TargetTruth {
  double range{};             // m
  double radial_velocity{};   // m/s, positive = receding
  Vec3 look_direction{};      // unit vector, position / |position|
};

struct SimResult {
  // Complex IF cube, compact row-major:
  //   num_rx == 1 → (num_chirps, samples_per_chirp)
  //   num_rx  > 1 → (num_rx, num_chirps, samples_per_chirp)
  std::vector<std::complex<double>> cube;
  std::vector<int64_t> shape;
  std::vector<TargetTruth> truth;
};

// Baseband reference chirp sample r[s] = exp(+j*pi*slope*t_s^2).
inline std::vector<std::complex<double>> reference_chirp(
    const SimWaveform& waveform, int64_t samples_per_chirp) {
  std::vector<std::complex<double>> ref(samples_per_chirp);
  const double slope = waveform.slope();
  for (int64_t s = 0; s < samples_per_chirp; ++s) {
    const double t = static_cast<double>(s) / waveform.sample_rate;
    const double phase = kSimPi * slope * t * t;
    ref[s] = std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return ref;
}

inline TargetTruth derive_truth(const PointTarget& target) {
  TargetTruth truth;
  truth.range = norm(target.position);
  if (truth.range <= 0.0) {
    throw std::invalid_argument(
        "point_target_sim: target at the origin has no defined look "
        "direction");
  }
  truth.look_direction = {target.position[0] / truth.range,
                          target.position[1] / truth.range,
                          target.position[2] / truth.range};
  truth.radial_velocity = dot(target.velocity, truth.look_direction);
  return truth;
}

// Synthesize the raw ADC cube plus ground truth for a scene.
inline SimResult simulate(const SimScene& scene) {
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument("point_target_sim: need >= 1 RX element");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;
  const double c = scene.propagation_speed;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double pri = scene.waveform.chirp_repetition_interval;

  SimResult result;
  result.cube.assign(
      static_cast<size_t>(num_rx * num_chirps * samples), {0.0, 0.0});
  if (num_rx == 1) {
    result.shape = {num_chirps, samples};
  } else {
    result.shape = {num_rx, num_chirps, samples};
  }

  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    result.truth.push_back(truth);

    for (int64_t rx = 0; rx < num_rx; ++rx) {
      // Far-field two-way delay: 2*R_k minus the (tx+rx) aperture
      // projection onto the look direction.
      const double aperture =
          dot(scene.tx_position, truth.look_direction) +
          dot(scene.rx_positions[static_cast<size_t>(rx)],
              truth.look_direction);
      for (int64_t k = 0; k < num_chirps; ++k) {
        const double range_k =
            truth.range +
            truth.radial_velocity * static_cast<double>(k) * pri;
        const double tau = (2.0 * range_k - aperture) / c;
        const double beat = slope * tau;  // Hz
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          const double phase =
              2.0 * kSimPi * (beat * t) - 2.0 * kSimPi * f_c * tau;
          const std::complex<double> sample =
              target.amplitude *
              std::complex<double>(std::cos(phase), std::sin(phase));
          result.cube[static_cast<size_t>((rx * num_chirps + k) * samples +
                                          s)] += sample;
        }
      }
    }
  }

  // Pre-mixer output: multiply the dechirped model back up by the
  // reference chirp (dechirp = conj(reference) multiplication).
  if (!scene.dechirped) {
    const auto ref = reference_chirp(scene.waveform, samples);
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t k = 0; k < num_chirps; ++k) {
        for (int64_t s = 0; s < samples; ++s) {
          result.cube[static_cast<size_t>((rx * num_chirps + k) * samples +
                                          s)] *= ref[static_cast<size_t>(s)];
        }
      }
    }
  }

  // Additive complex white Gaussian noise (seeded, deterministic).
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

// Real-IF variant: the real part of the complex IF cube (exercises the
// operators' RFFT front-end path).
inline std::vector<double> to_real_if(const SimResult& sim) {
  std::vector<double> real_cube(sim.cube.size());
  for (size_t i = 0; i < sim.cube.size(); ++i) {
    real_cube[i] = sim.cube[i].real();
  }
  return real_cube;
}

}  // namespace holoscan::radar::sim
