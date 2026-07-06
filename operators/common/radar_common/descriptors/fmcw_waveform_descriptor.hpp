// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace holoscan::radar {

// Default wave propagation speed (vacuum speed of light, m/s).
inline constexpr double kDefaultPropagationSpeed = 299792458.0;

// Shared FMCW chirp descriptor, one of the module's shared config-plane
// descriptor types (see operators/common/README.md). Precision-agnostic:
// all scalars are double.
//
// Exactly one of `bandwidth` / `chirp_slope` should be provided; the other
// is derived (slope = bandwidth / chirp_duration). Providing both is an
// error unless they are consistent.
struct FMCWWaveformDescriptor {
  // Carrier frequency f_c in Hz. Sets the wavelength for Doppler/velocity.
  double carrier_frequency{0.0};

  // Swept bandwidth in Hz over one chirp. Define this or chirp_slope.
  std::optional<double> bandwidth{};

  // Sweep slope in Hz/s. Define this or bandwidth.
  std::optional<double> chirp_slope{};

  // Active sweep duration of one chirp in seconds.
  double chirp_duration{0.0};

  // ADC sample rate in Hz (fast-time).
  double sample_rate{0.0};

  // Chirp repetition interval (PRI) in seconds. For MIMO this is the
  // per-TX effective interval.
  double chirp_repetition_interval{0.0};

  // Optional measured dechirp reference, length samples_per_chirp. When
  // empty, the reference chirp is synthesized from the analytic fields.
  std::vector<std::complex<double>> reference_samples{};

  friend bool operator==(const FMCWWaveformDescriptor& a,
                         const FMCWWaveformDescriptor& b) {
    return a.carrier_frequency == b.carrier_frequency &&
           a.bandwidth == b.bandwidth && a.chirp_slope == b.chirp_slope &&
           a.chirp_duration == b.chirp_duration &&
           a.sample_rate == b.sample_rate &&
           a.chirp_repetition_interval == b.chirp_repetition_interval &&
           a.reference_samples == b.reference_samples;
  }
  friend bool operator!=(const FMCWWaveformDescriptor& a,
                         const FMCWWaveformDescriptor& b) {
    return !(a == b);
  }

  // Sweep slope in Hz/s, from chirp_slope or bandwidth / chirp_duration.
  // Throws std::invalid_argument when neither is derivable or the two
  // are inconsistent.
  double slope() const {
    if (chirp_slope.has_value()) {
      if (bandwidth.has_value() && chirp_duration > 0.0) {
        const double derived = *bandwidth / chirp_duration;
        const double tol = 1e-9 * std::max(std::abs(derived), 1.0);
        if (std::abs(derived - *chirp_slope) > tol) {
          throw std::invalid_argument(
              "FMCWWaveformDescriptor: bandwidth/chirp_duration and "
              "chirp_slope are both set but inconsistent");
        }
      }
      return *chirp_slope;
    }
    if (bandwidth.has_value()) {
      if (chirp_duration <= 0.0) {
        throw std::invalid_argument(
            "FMCWWaveformDescriptor: chirp_duration must be > 0 to derive "
            "the sweep slope from bandwidth");
      }
      return *bandwidth / chirp_duration;
    }
    throw std::invalid_argument(
        "FMCWWaveformDescriptor: define bandwidth or chirp_slope");
  }

  // Carrier wavelength in meters (propagation speed / carrier frequency).
  double wavelength(double propagation_speed = kDefaultPropagationSpeed) const {
    if (carrier_frequency <= 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: carrier_frequency must be > 0");
    }
    return propagation_speed / carrier_frequency;
  }
};

}  // namespace holoscan::radar
