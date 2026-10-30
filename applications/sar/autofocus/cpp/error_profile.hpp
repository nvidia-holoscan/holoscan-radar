// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The line-of-sight error shapes this app can inject, and the per-pulse metre
// sequence they generate. Split out of position_error_op.hpp so the config
// loader can validate a profile without pulling in Holoscan, and kept STL-only
// so it can be checked on the host.
//
// Real platform error is a trend plus an oscillation plus a drift rather than
// any one of them, so the three components are additive rather than exclusive.
// A pure quadratic -- the only shape injected before 2026-10-29 -- is the
// smoothest, most symmetric error there is, and the one a geometrically
// shrinking PGA window suits best, which is exactly why conclusions drawn from
// it alone are suspect.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar::app::sar {

inline double poly_eval(const std::vector<double>& coeffs, double s) {
  double acc = 0.0;
  for (std::size_t i = coeffs.size(); i > 0; --i) {
    acc = acc * s + coeffs[i - 1];
  }
  return acc;
}

// PGA is blind to a constant and a linear ramp: they translate the image
// rather than blur it. Scoring an estimate against a truth is only meaningful
// once both have had an affine fit removed.
inline void remove_affine(std::vector<double>* phi) {
  const int64_t n = static_cast<int64_t>(phi->size());
  if (n < 2) {
    return;
  }
  double sum = 0.0;
  double sumk = 0.0;
  for (int64_t k = 0; k < n; ++k) {
    sum += (*phi)[static_cast<std::size_t>(k)];
    sumk += static_cast<double>(k) * (*phi)[static_cast<std::size_t>(k)];
  }
  const double xmean = 0.5 * static_cast<double>(n - 1);
  const double denom =
      static_cast<double>(n) * (static_cast<double>(n) * static_cast<double>(n) - 1.0) / 12.0;
  const double slope = (sumk - xmean * sum) / denom;
  const double intercept = sum / static_cast<double>(n) - slope * xmean;
  for (int64_t k = 0; k < n; ++k) {
    (*phi)[static_cast<std::size_t>(k)] -= intercept + slope * static_cast<double>(k);
  }
}

// `cycles` oscillations of `amplitude` metres across the whole aperture. One
// cycle is the lowest frequency a window can see as curvature rather than a
// ramp; many cycles is vibration, which the geometric window schedule has no
// reason to handle and which the quadratic sweeps never probed.
struct ErrorSinusoid {
  double amplitude{0.0};  // metres
  double cycles{0.0};     // oscillations over the aperture
  double phase_deg{0.0};
};

struct ErrorProfile {
  std::vector<double> polynomial;        // metres vs s in [-1, 1]
  std::vector<ErrorSinusoid> sinusoids;  // metres vs pulse fraction
  // RMS of the walk *after* its affine part is removed. A raw random walk is
  // dominated by the trend it happens to take, and a trend is the one thing
  // neither PGA nor the scorer can see, so normalizing the raw walk would put
  // most of the requested magnitude somewhere unmeasurable and make this axis
  // incomparable with the polynomial one. Normalizing the observable part
  // instead means `random_walk_rms` is the error PGA actually has to estimate.
  double random_walk_rms{0.0};  // metres
  std::uint64_t random_walk_seed{1};

  [[nodiscard]] bool empty() const {
    return polynomial.empty() && sinusoids.empty() && random_walk_rms == 0.0;
  }
};

// Per-pulse line-of-sight displacement in metres. Deterministic in the seed, so
// a sweep row is reproducible and the walk is identical across the ticks that
// read it batch by batch.
inline std::vector<double> profile_metres(const ErrorProfile& profile, int64_t num_pulses) {
  if (num_pulses < 1) {
    throw std::invalid_argument("profile_metres: num_pulses must be >= 1");
  }
  const auto n = static_cast<std::size_t>(num_pulses);
  std::vector<double> metres(n, 0.0);
  const double denom = static_cast<double>(std::max(num_pulses - 1, int64_t{1}));

  if (!profile.polynomial.empty()) {
    for (std::size_t p = 0; p < n; ++p) {
      const double s = 2.0 * static_cast<double>(p) / denom - 1.0;
      metres[p] += poly_eval(profile.polynomial, s);
    }
  }

  for (const auto& sine : profile.sinusoids) {
    const double phase = sine.phase_deg * kPi / 180.0;
    const double rate = 2.0 * kPi * sine.cycles;
    for (std::size_t p = 0; p < n; ++p) {
      const double frac = static_cast<double>(p) / denom;
      metres[p] += sine.amplitude * std::sin(rate * frac + phase);
    }
  }

  if (profile.random_walk_rms != 0.0 && num_pulses >= 2) {
    std::mt19937_64 gen{profile.random_walk_seed};
    std::normal_distribution<double> step{0.0, 1.0};
    std::vector<double> walk(n, 0.0);
    double acc = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      acc += step(gen);
      walk[p] = acc;
    }
    remove_affine(&walk);
    double sq = 0.0;
    for (const double v : walk) {
      sq += v * v;
    }
    const double rms = std::sqrt(sq / static_cast<double>(n));
    if (rms > 0.0) {
      const double scale = profile.random_walk_rms / rms;
      for (std::size_t p = 0; p < n; ++p) {
        metres[p] += walk[p] * scale;
      }
    }
  }

  return metres;
}

// How the shape reads in a log line, so a sweep row says what it injected.
inline std::string describe(const ErrorProfile& profile) {
  std::string out;
  if (!profile.polynomial.empty()) {
    out += "poly deg " + std::to_string(profile.polynomial.size() - 1);
  }
  if (!profile.sinusoids.empty()) {
    if (!out.empty()) {
      out += " + ";
    }
    out += std::to_string(profile.sinusoids.size()) + " sinusoid(s)";
  }
  if (profile.random_walk_rms != 0.0) {
    if (!out.empty()) {
      out += " + ";
    }
    out += "random walk seed " + std::to_string(profile.random_walk_seed);
  }
  return out.empty() ? "none" : out;
}

}  // namespace holoscan::radar::app::sar
