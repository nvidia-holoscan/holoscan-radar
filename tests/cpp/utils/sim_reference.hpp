// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Analytic CPU point-target FMCW simulator — the closed-form ground truth used
// by the tests. This is the reference the
// runtime GPU simulator (simulator.hpp) is
// validated against; it deliberately lives under tests/ (not in the operators
// tree) and stays STL-only, so it is an implementation-independent oracle for
// both the GPU sim and the processing operators.
//
// The scene model + geometry it builds on (SimScene, PointTarget, derive_truth,
// …) is the shared, production header radar_common/sim/scene.hpp.
//
// Sign conventions and modeling assumptions are documented on that header; this
// file is the single-TX synthesis loop that was factored out of it. (The MIMO
// reference solvers are the sibling utils/mimo_reference.hpp.)

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/math/constants.hpp"
#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::sim {

namespace reference_detail {

struct FoldedDelay {
  int64_t ambiguity_order;
  double residual;
};

inline void validate_scene(const SimScene& scene, const char* caller) {
  const SimWaveform& waveform = scene.waveform;
  const auto finite_positive = [](double value) {
    return std::isfinite(value) && value > 0.0;
  };
  if (!finite_positive(waveform.carrier_frequency) || !finite_positive(waveform.bandwidth) ||
      !finite_positive(waveform.chirp_duration) || !finite_positive(waveform.sample_rate) ||
      !finite_positive(waveform.chirp_repetition_interval) ||
      !finite_positive(scene.propagation_speed)) {
    throw std::invalid_argument(std::string(caller) +
                                ": waveform and propagation fields must be "
                                "finite and positive");
  }
  if (scene.num_chirps <= 0 || scene.samples_per_chirp <= 0) {
    throw std::invalid_argument(std::string(caller) + ": cube dimensions must be positive");
  }
  const double sample_window = static_cast<double>(scene.samples_per_chirp) / waveform.sample_rate;
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                           std::max(waveform.chirp_duration, waveform.chirp_repetition_interval);
  if (sample_window > waveform.chirp_duration + tolerance) {
    throw std::invalid_argument(std::string(caller) + ": sample window exceeds chirp duration");
  }
  if (waveform.chirp_repetition_interval + tolerance < waveform.chirp_duration) {
    throw std::invalid_argument(std::string(caller) + ": CRI must be at least chirp duration");
  }
  const AnalogIfConfig& analog_if = scene.analog_if;
  if (!std::isfinite(analog_if.passband_fraction) || !std::isfinite(analog_if.stopband_fraction) ||
      analog_if.passband_fraction <= 0.0 ||
      analog_if.passband_fraction >= analog_if.stopband_fraction ||
      analog_if.stopband_fraction > 1.0) {
    throw std::invalid_argument(std::string(caller) +
                                ": analog IF fractions must satisfy 0 < passband < stopband <= 1");
  }
  if (scene.range_loss.model == RangeLossModel::kMonostaticFreeSpace &&
      !finite_positive(scene.range_loss.reference_range)) {
    throw std::invalid_argument(
        std::string(caller) +
        ": free-space range-loss reference range must be finite and positive");
  }
}

inline FoldedDelay fold_delay(double delay, double cri) {
  const double order_value = std::floor(delay / cri);
  int64_t order = static_cast<int64_t>(order_value);
  double residual = delay - order_value * cri;
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * std::max(std::abs(delay), cri);
  if (std::abs(residual) <= tolerance) {
    residual = 0.0;
  } else if (std::abs(cri - residual) <= tolerance) {
    ++order;
    residual = 0.0;
  }
  return FoldedDelay{order, residual};
}

inline double analog_if_gain(const SimScene& scene, double beat_frequency) {
  if (scene.analog_if.response == AnalogIfResponseType::kNone) {
    return 1.0;
  }
  const double boundary = scene.analog_if.sampling_mode == IfSamplingMode::kComplexIq
                              ? scene.waveform.sample_rate
                              : 0.5 * scene.waveform.sample_rate;
  const double passband = scene.analog_if.passband_fraction * boundary;
  const double stopband = scene.analog_if.stopband_fraction * boundary;
  const double frequency = std::abs(beat_frequency);
  if (frequency <= passband)
    return 1.0;
  if (frequency >= stopband)
    return 0.0;
  const double x = (frequency - passband) / (stopband - passband);
  return 0.5 * (1.0 + std::cos(kPi * x));
}

inline double range_loss_gain(const SimScene& scene, double range) {
  if (scene.range_loss.model == RangeLossModel::kNone) {
    return 1.0;
  }
  if (!std::isfinite(range) || range <= 0.0) {
    throw std::invalid_argument("range_loss_gain: modeled range must be finite and positive");
  }
  const double ratio = scene.range_loss.reference_range / range;
  return ratio * ratio;
}

}  // namespace reference_detail

// Baseband reference chirp sample r[s] = exp(+j*pi*slope*t_s^2).
inline std::vector<std::complex<double>> reference_chirp(const SimWaveform& waveform,
                                                         int64_t samples_per_chirp) {
  std::vector<std::complex<double>> ref(samples_per_chirp);
  const double slope = waveform.slope();
  for (int64_t s = 0; s < samples_per_chirp; ++s) {
    const double t = static_cast<double>(s) / waveform.sample_rate;
    const double phase = kPi * slope * t * t;
    ref[s] = std::complex<double>(std::cos(phase), std::sin(phase));
  }
  return ref;
}

// Synthesize the raw ADC cube plus ground truth for a single-TX scene — the CPU
// reference the GPU simulate_gpu() is checked against.
inline SimResult simulate(const SimScene& scene) {
  reference_detail::validate_scene(scene, "simulate");
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument("simulate: need >= 1 RX element");
  }
  if (scene.tx_positions.size() != 1) {
    throw std::invalid_argument(
        "simulate: models a single TX; use the MIMO "
        "simulators (mimo_sim.hpp) for multi-TX scenes");
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
  result.sampling_mode = scene.analog_if.sampling_mode;
  result.cube.assign(static_cast<size_t>(num_rx * num_chirps * samples), {0.0, 0.0});
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
          dot(scene.tx_positions[0], truth.look_direction) +
          dot(scene.rx_positions[static_cast<size_t>(rx)], truth.look_direction);
      for (int64_t k = 0; k < num_chirps; ++k) {
        const double range_k = truth.range + truth.radial_velocity * static_cast<double>(k) * pri;
        const double tau = (2.0 * range_k - aperture) / c;
        const auto folded = reference_detail::fold_delay(tau, pri);
        const double beat = slope * folded.residual;
        const double gain = reference_detail::range_loss_gain(scene, range_k) *
                            reference_detail::analog_if_gain(scene, beat);
        for (int64_t s = 0; s < samples; ++s) {
          const double t = static_cast<double>(s) / fs;
          const double phase = 2.0 * kPi * (beat * t) - 2.0 * kPi * f_c * tau;
          const std::complex<double> sample =
              gain * target.amplitude * std::complex<double>(std::cos(phase), std::sin(phase));
          result.cube[static_cast<size_t>((rx * num_chirps + k) * samples + s)] += sample;
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
          result.cube[static_cast<size_t>((rx * num_chirps + k) * samples + s)] *=
              ref[static_cast<size_t>(s)];
        }
      }
    }
  }

  // Clean scatterer response — thermal noise and other I/Q additives are a
  // separate stage the caller applies (radar_common/sim/additives.hpp).
  return result;
}

// Real-IF variant: the real part of the complex IF cube (exercises the
// operators' RFFT front-end path).
inline std::vector<double> to_real_if(const SimResult& sim) {
  if (sim.sampling_mode != IfSamplingMode::kReal) {
    throw std::invalid_argument("to_real_if: result was not synthesized in real-IF mode");
  }
  std::vector<double> real_cube(sim.cube.size());
  for (size_t i = 0; i < sim.cube.size(); ++i) {
    real_cube[i] = sim.cube[i].real();
  }
  return real_cube;
}

}  // namespace holoscan::radar::sim
