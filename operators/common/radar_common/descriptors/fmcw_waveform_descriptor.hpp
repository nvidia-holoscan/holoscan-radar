// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace holoscan::radar {

// Default wave propagation speed (vacuum speed of light, m/s).
inline constexpr double kDefaultPropagationSpeed = 299792458.0;

// Construction input for FMCWWaveformDescriptor. Transient: fill the fields
// (designated initializers work), hand it to the descriptor constructor, and
// let validation/derivation happen there. Exactly one of `bandwidth` /
// `chirp_slope` should be provided; providing both is an error unless they
// are consistent (slope = bandwidth / chirp_duration).
struct FMCWWaveformSpec {
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
};

// Shared FMCW chirp descriptor, one of the module's shared config-plane
// descriptor types (see operators/common/README.md). Precision-agnostic:
// all scalars are double.
//
// Immutable value type: every derived quantity (slope <-> bandwidth,
// wavelength) is computed and validated once at construction, so all
// accessors are plain non-throwing reads and a constructed descriptor can
// never hold mutually inconsistent fields. To change a value, construct a
// new descriptor from an updated FMCWWaveformSpec.
//
// The default constructor exists only for SDK plumbing (Parameter<T>
// requires default-constructible types) and yields an empty descriptor with
// is_valid() == false; consumers must reject it before use.
class FMCWWaveformDescriptor {
 public:
  FMCWWaveformDescriptor() = default;

  // Validates the spec and derives all redundant quantities. Throws
  // std::invalid_argument naming the offending field when the spec is
  // incomplete or inconsistent.
  explicit FMCWWaveformDescriptor(FMCWWaveformSpec spec) {
    if (spec.carrier_frequency <= 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: carrier_frequency must be > 0");
    }
    if (spec.chirp_duration <= 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: chirp_duration must be > 0");
    }
    if (spec.sample_rate <= 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: sample_rate must be > 0");
    }
    if (spec.chirp_repetition_interval <= 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: chirp_repetition_interval must be > 0");
    }
    if (!spec.bandwidth.has_value() && !spec.chirp_slope.has_value()) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: define bandwidth or chirp_slope");
    }
    if (spec.bandwidth.has_value() && spec.chirp_slope.has_value()) {
      const double derived = *spec.bandwidth / spec.chirp_duration;
      const double tol = 1e-9 * std::max(std::abs(derived), 1.0);
      if (std::abs(derived - *spec.chirp_slope) > tol) {
        throw std::invalid_argument(
            "FMCWWaveformDescriptor: bandwidth/chirp_duration and "
            "chirp_slope are both set but inconsistent");
      }
    }
    chirp_slope_ = spec.chirp_slope.has_value()
                       ? *spec.chirp_slope
                       : *spec.bandwidth / spec.chirp_duration;
    if (chirp_slope_ == 0.0) {
      throw std::invalid_argument(
          "FMCWWaveformDescriptor: sweep slope must be non-zero (bandwidth "
          "or chirp_slope is 0)");
    }
    bandwidth_ = spec.bandwidth.has_value()
                     ? *spec.bandwidth
                     : chirp_slope_ * spec.chirp_duration;
    carrier_frequency_ = spec.carrier_frequency;
    chirp_duration_ = spec.chirp_duration;
    sample_rate_ = spec.sample_rate;
    chirp_repetition_interval_ = spec.chirp_repetition_interval;
    wavelength_ = kDefaultPropagationSpeed / carrier_frequency_;
    reference_samples_ = std::move(spec.reference_samples);
    valid_ = true;
  }

  // false only for a default-constructed (empty) descriptor.
  bool is_valid() const { return valid_; }

  // Carrier frequency f_c in Hz.
  double carrier_frequency() const { return carrier_frequency_; }

  // Swept bandwidth in Hz over one chirp (derived at construction when the
  // spec provided chirp_slope).
  double bandwidth() const { return bandwidth_; }

  // Sweep slope in Hz/s (derived at construction when the spec provided
  // bandwidth). Non-zero for any valid descriptor.
  double slope() const { return chirp_slope_; }

  // Active sweep duration of one chirp in seconds.
  double chirp_duration() const { return chirp_duration_; }

  // ADC sample rate in Hz (fast-time).
  double sample_rate() const { return sample_rate_; }

  // Chirp repetition interval (PRI) in seconds.
  double chirp_repetition_interval() const {
    return chirp_repetition_interval_;
  }

  // Carrier wavelength in meters at the default propagation speed
  // (precomputed at construction).
  double wavelength() const { return wavelength_; }

  // Carrier wavelength in meters for a non-default propagation speed.
  // Computed on call (single division); invalid descriptors yield inf.
  double wavelength(double propagation_speed) const {
    return propagation_speed / carrier_frequency_;
  }

  // Optional measured dechirp reference; empty when synthesized.
  const std::vector<std::complex<double>>& reference_samples() const {
    return reference_samples_;
  }

  friend bool operator==(const FMCWWaveformDescriptor& a,
                         const FMCWWaveformDescriptor& b) {
    return a.valid_ == b.valid_ &&
           a.carrier_frequency_ == b.carrier_frequency_ &&
           a.chirp_slope_ == b.chirp_slope_ &&
           a.chirp_duration_ == b.chirp_duration_ &&
           a.sample_rate_ == b.sample_rate_ &&
           a.chirp_repetition_interval_ == b.chirp_repetition_interval_ &&
           a.reference_samples_ == b.reference_samples_;
  }
  friend bool operator!=(const FMCWWaveformDescriptor& a,
                         const FMCWWaveformDescriptor& b) {
    return !(a == b);
  }

 private:
  bool valid_{false};
  double carrier_frequency_{0.0};
  double bandwidth_{0.0};
  double chirp_slope_{0.0};
  double chirp_duration_{0.0};
  double sample_rate_{0.0};
  double chirp_repetition_interval_{0.0};
  double wavelength_{0.0};
  std::vector<std::complex<double>> reference_samples_{};
};

}  // namespace holoscan::radar
