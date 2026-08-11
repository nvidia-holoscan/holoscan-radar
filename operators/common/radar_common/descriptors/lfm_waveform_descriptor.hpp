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

// kDefaultPropagationSpeed lives in the FMCW descriptor header rather than in
// constants.hpp; included here so this header compiles standalone. It moves
// with the D24 merge of the two descriptors.
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/math/constants.hpp"

namespace holoscan::radar {

// Construction input for LFMWaveformDescriptor. Transient: fill the fields
// (designated initializers work), hand it to the descriptor constructor, and
// let validation/derivation happen there. Exactly one of `bandwidth` /
// `chirp_slope` should be provided; providing both is an error unless they
// are consistent (slope = bandwidth / chirp_duration).
struct LFMWaveformSpec {
  // Carrier frequency f_c in Hz. Sets the wavelength.
  double carrier_frequency{0.0};

  // Swept bandwidth in Hz over one chirp. Define this or chirp_slope.
  std::optional<double> bandwidth{};

  // Sweep slope in Hz/s. Define this or bandwidth.
  std::optional<double> chirp_slope{};

  // Active sweep duration of one chirp in seconds. Named chirp_duration
  // rather than SAR's usual "pulse duration" to mirror
  // FMCWWaveformDescriptor exactly — see the convergence note below.
  double chirp_duration{0.0};

  // ADC sample rate in Hz (fast-time).
  double sample_rate{0.0};

  // Chirp repetition interval (PRI) in seconds. For SAR this is the
  // inter-pulse time along the aperture.
  double chirp_repetition_interval{0.0};

  // Optional measured matched-filter replica. When empty, the replica is
  // synthesized from the analytic fields.
  std::vector<std::complex<double>> reference_samples{};
};

// Shared linear-FM chirp descriptor for the SAR chain, one of the module's
// shared config-plane descriptor types (see operators/common/README.md).
// Precision-agnostic: all scalars are double.
//
// CONVERGENCE NOTE (D24). This type and FMCWWaveformDescriptor carry the
// same fields, because FMCW and SAR both transmit linear-FM chirps; that
// type is FMCW only in its name. They are deliberately kept separate for
// now: renaming the FMCW type would be a breaking change to shipped FMCW
// public API made to satisfy a SAR contract, so SAR adds rather than
// renames. The two are intended to merge, and the field names here mirror
// FMCWWaveformDescriptor exactly so that merging later is one sweep
// (migrate FMCW's call sites onto this type, delete the old one) rather
// than two.
//
// The accepted cost is silent drift: a field added to one type will not
// appear in the other, and nothing fails to compile to say so. If you add a
// field here — a sweep direction flag, a chirp shape — add it to
// FMCWWaveformDescriptor too, or record why it is SAR-only.
//
// Two fields read differently between the modalities, in comments only.
// chirp_repetition_interval is SAR's inter-pulse time along the aperture
// where FMCW describes per-TX effective spacing for MIMO; reference_samples
// is SAR's measured matched-filter replica where FMCW's is the measured
// dechirp reference. Same arrays, consumed by different stages.
//
// Immutable value type: every derived quantity (slope <-> bandwidth,
// wavelength) is computed and validated once at construction, so all
// accessors are plain non-throwing reads and a constructed descriptor can
// never hold mutually inconsistent fields. To change a value, construct a
// new descriptor from an updated LFMWaveformSpec.
//
// The default constructor exists only for SDK plumbing (Parameter<T>
// requires default-constructible types) and yields an empty descriptor with
// is_valid() == false; consumers must reject it before use.
class LFMWaveformDescriptor {
 public:
  LFMWaveformDescriptor() = default;

  // Validates the spec and derives all redundant quantities. Throws
  // std::invalid_argument naming the offending field when the spec is
  // incomplete or inconsistent.
  explicit LFMWaveformDescriptor(LFMWaveformSpec spec) {
    // NaN compares false against every bound, so each check must require
    // finiteness explicitly — otherwise non-finite values would flow into
    // the derived fields of a descriptor reporting is_valid().
    if (!std::isfinite(spec.carrier_frequency) ||
        spec.carrier_frequency <= 0.0) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: carrier_frequency must be finite and > 0");
    }
    if (!std::isfinite(spec.chirp_duration) || spec.chirp_duration <= 0.0) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: chirp_duration must be finite and > 0");
    }
    if (!std::isfinite(spec.sample_rate) || spec.sample_rate <= 0.0) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: sample_rate must be finite and > 0");
    }
    if (!std::isfinite(spec.chirp_repetition_interval) ||
        spec.chirp_repetition_interval <= 0.0) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: chirp_repetition_interval must be finite "
          "and > 0");
    }
    if (spec.bandwidth.has_value() && !std::isfinite(*spec.bandwidth)) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: bandwidth must be finite");
    }
    if (spec.chirp_slope.has_value() && !std::isfinite(*spec.chirp_slope)) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: chirp_slope must be finite");
    }
    if (!spec.bandwidth.has_value() && !spec.chirp_slope.has_value()) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: define bandwidth or chirp_slope");
    }
    if (spec.bandwidth.has_value() && spec.chirp_slope.has_value()) {
      const double derived = *spec.bandwidth / spec.chirp_duration;
      const double tol = 1e-9 * std::max(std::abs(derived), 1.0);
      if (std::abs(derived - *spec.chirp_slope) > tol) {
        throw std::invalid_argument(
            "LFMWaveformDescriptor: bandwidth/chirp_duration and "
            "chirp_slope are both set but inconsistent");
      }
    }
    chirp_slope_ = spec.chirp_slope.has_value()
                       ? *spec.chirp_slope
                       : *spec.bandwidth / spec.chirp_duration;
    if (chirp_slope_ == 0.0) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: sweep slope must be non-zero (bandwidth "
          "or chirp_slope is 0)");
    }
    bandwidth_ = spec.bandwidth.has_value()
                     ? *spec.bandwidth
                     : chirp_slope_ * spec.chirp_duration;
    // Finite inputs can still overflow the derivation (bandwidth / tiny
    // duration, slope * huge duration).
    if (!std::isfinite(chirp_slope_) || !std::isfinite(bandwidth_)) {
      throw std::invalid_argument(
          "LFMWaveformDescriptor: derived sweep slope / bandwidth is not "
          "finite (bandwidth, chirp_slope, and chirp_duration are "
          "mutually inconsistent in magnitude)");
    }
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
  // spec provided chirp_slope). Sets the compressed range resolution.
  double bandwidth() const { return bandwidth_; }

  // Sweep slope in Hz/s (derived at construction when the spec provided
  // bandwidth). Non-zero for any valid descriptor.
  double slope() const { return chirp_slope_; }

  // Active sweep duration of one chirp in seconds.
  double chirp_duration() const { return chirp_duration_; }

  // ADC sample rate in Hz (fast-time).
  double sample_rate() const { return sample_rate_; }

  // Pulse repetition interval in seconds.
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

  // Optional measured matched-filter replica; empty when synthesized.
  const std::vector<std::complex<double>>& reference_samples() const {
    return reference_samples_;
  }

  friend bool operator==(const LFMWaveformDescriptor& a,
                         const LFMWaveformDescriptor& b) {
    return a.valid_ == b.valid_ &&
           a.carrier_frequency_ == b.carrier_frequency_ &&
           a.chirp_slope_ == b.chirp_slope_ &&
           a.chirp_duration_ == b.chirp_duration_ &&
           a.sample_rate_ == b.sample_rate_ &&
           a.chirp_repetition_interval_ == b.chirp_repetition_interval_ &&
           a.reference_samples_ == b.reference_samples_;
  }
  friend bool operator!=(const LFMWaveformDescriptor& a,
                         const LFMWaveformDescriptor& b) {
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
