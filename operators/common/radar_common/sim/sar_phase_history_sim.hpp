// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Analytic SAR phase-history simulator: the closed-form signal a set of ideal
// point scatterers writes into a CPHD signal array. A signal model, not an EM
// solver, and the SAR counterpart to point_target_sim.hpp — the tautology
// guard tests assert range compression and image formation against.
//
// STL-only by design (no operator, MatX or CUDA includes), so the ground truth
// it produces comes from the signal model rather than from the code under
// test.
//
// **The model.** CPHD is *compensated* phase history: the signal has already
// been deramped and motion-compensated to a reference point, so what is stored
// for pulse v and frequency sample k is each scatterer's residual phase
// relative to that reference. For a scatterer whose differential range is
//
//     dR(v) = |apc(v) - target| - |apc(v) - srp(v)|
//
// the stored sample is
//
//     s(v, k) = sum_targets  a * exp(-j * 4 * pi * f_k * dR(v) / c)
//
// with f_k the transmit frequency of sample k.
//
// **The sign is a convention, and it is the convention.** It is the same one
// tools/sar_sim/sar_sim/signal.py writes, paired with `Global/SGN = -1` in the
// CPHD XML: an *inverse* transform along the frequency axis (the exp(+2 pi i)
// kernel, numpy's ifft and MatX's ifft) puts a target at positive dR at a
// positive delay, and so at a range bin above centre. SARRangeCompressionOp
// and BackprojectionOp both depend on this; disagreeing with it mirrors the
// image about the reference point, which is quiet — a mirrored image of a
// symmetric scene looks fine.
//
// **The frequency axis is centred on the carrier and spans the bandwidth
// first-sample-to-last**, so N samples divide the band into N-1 steps:
// df = bandwidth / (num_samples - 1). This matches CPHD's `FxBand` against the
// per-vector `SC0` / `SCSS`, and matches the in-repo simulator. The other
// spelling (B/N) differs by one part in N, which is a range *scale* error — it
// vanishes at the reference range and grows toward the swath edges, so a
// single target at scene centre cannot see it.
//
// Scope: this header models the fast-time (range) axis, which is what range
// compression needs, and takes the per-pulse differential range as an input
// rather than deriving it from a trajectory. Image formation needs the
// geometry too, and will extend this file rather than restate the phase model.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar::sim {

// One ideal point scatterer, described by the quantity range compression
// actually resolves: its range relative to the reference point.
struct SarPointTarget {
  // Differential range in metres, positive for a target farther than the
  // reference point. Constant across the aperture unless
  // `differential_range_per_pulse` is supplied below.
  double differential_range{0.0};

  // Per-pulse differential range, overriding the constant above when
  // non-empty. Length must equal num_pulses. This is how an aperture with
  // real range walk is described without giving this header a trajectory.
  std::vector<double> differential_range_per_pulse{};

  std::complex<double> amplitude{1.0, 0.0};

  double range_at(int64_t pulse) const {
    if (differential_range_per_pulse.empty()) {
      return differential_range;
    }
    // Only sar_phase_history() has checked the length against num_pulses, so
    // a direct caller reaching past the end would otherwise read whatever
    // sits after the vector and hand it back as ground truth.
    if (pulse < 0 ||
        static_cast<size_t>(pulse) >= differential_range_per_pulse.size()) {
      throw std::invalid_argument(
          "sar_phase_history_sim: pulse index is outside "
          "differential_range_per_pulse");
    }
    return differential_range_per_pulse[static_cast<size_t>(pulse)];
  }
};

// The collection's fast-time sampling, mirroring the fields of
// LFMWaveformDescriptor that the frequency axis depends on.
struct SarPhaseHistoryConfig {
  double carrier_frequency{9.6e9};  // Hz, band centre
  double bandwidth{600e6};          // Hz, first sample to last
  int64_t num_pulses{16};
  int64_t num_samples{256};
  double propagation_speed{kSpeedOfLight};
};

// Transmit frequency of fast-time sample k, in Hz.
inline std::vector<double> sar_frequency_axis(
    const SarPhaseHistoryConfig& cfg) {
  if (cfg.num_samples < 2) {
    throw std::invalid_argument(
        "sar_phase_history_sim: num_samples must be >= 2 for the frequency "
        "axis to have a spacing");
  }
  const double df =
      cfg.bandwidth / static_cast<double>(cfg.num_samples - 1);
  const double f0 = cfg.carrier_frequency - 0.5 * cfg.bandwidth;
  std::vector<double> freqs(static_cast<size_t>(cfg.num_samples));
  for (int64_t k = 0; k < cfg.num_samples; ++k) {
    freqs[static_cast<size_t>(k)] = f0 + df * static_cast<double>(k);
  }
  return freqs;
}

// Range bin spacing (m) of the compressed profile for a transform of length
// `nfft`. Derived here from the frequency axis this header generates, so a
// test can predict where a target lands without asking the operator.
inline double sar_range_bin_spacing(const SarPhaseHistoryConfig& cfg,
                                    int64_t nfft) {
  if (cfg.num_samples < 2) {
    throw std::invalid_argument(
        "sar_phase_history_sim: num_samples must be >= 2 for the frequency "
        "axis to have a spacing");
  }
  if (nfft <= 0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: nfft must be positive");
  }
  const double df =
      cfg.bandwidth / static_cast<double>(cfg.num_samples - 1);
  return cfg.propagation_speed / (2.0 * static_cast<double>(nfft) * df);
}

// Compressed range resolution (m), c / (2B): the width of the impulse
// response, which zero-padding interpolates but does not improve.
inline double sar_range_resolution(const SarPhaseHistoryConfig& cfg) {
  return cfg.propagation_speed / (2.0 * cfg.bandwidth);
}

// Synthesize the (num_pulses, num_samples) phase-history array, row-major.
// Phase is accumulated in double and returned in double: the argument reaches
// tens of thousands of radians at X band, where a float accumulation loses the
// fractional part that carries all the information. The caller casts once, on
// the way to the device.
inline std::vector<std::complex<double>> sar_phase_history(
    const SarPhaseHistoryConfig& cfg,
    const std::vector<SarPointTarget>& targets) {
  if (cfg.num_pulses <= 0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: num_pulses must be positive");
  }
  if (cfg.num_samples <= 0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: num_samples must be positive");
  }
  if (!std::isfinite(cfg.bandwidth) || cfg.bandwidth <= 0.0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: bandwidth must be finite and > 0");
  }
  if (!std::isfinite(cfg.carrier_frequency) || cfg.carrier_frequency <= 0.0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: carrier_frequency must be finite and > 0");
  }
  if (!std::isfinite(cfg.propagation_speed) || cfg.propagation_speed <= 0.0) {
    throw std::invalid_argument(
        "sar_phase_history_sim: propagation_speed must be finite and > 0");
  }
  const std::vector<double> freqs = sar_frequency_axis(cfg);
  for (const auto& target : targets) {
    if (!target.differential_range_per_pulse.empty() &&
        static_cast<int64_t>(target.differential_range_per_pulse.size()) !=
            cfg.num_pulses) {
      throw std::invalid_argument(
          "sar_phase_history_sim: differential_range_per_pulse length does "
          "not match num_pulses");
    }
  }

  const double scale = -4.0 * kPi / cfg.propagation_speed;
  std::vector<std::complex<double>> signal(
      static_cast<size_t>(cfg.num_pulses * cfg.num_samples));
  for (int64_t v = 0; v < cfg.num_pulses; ++v) {
    for (const auto& target : targets) {
      const double delta_r = target.range_at(v);
      for (int64_t k = 0; k < cfg.num_samples; ++k) {
        const double phase = scale * delta_r * freqs[static_cast<size_t>(k)];
        signal[static_cast<size_t>(v * cfg.num_samples + k)] +=
            target.amplitude *
            std::complex<double>(std::cos(phase), std::sin(phase));
      }
    }
  }
  return signal;
}

}  // namespace holoscan::radar::sim
