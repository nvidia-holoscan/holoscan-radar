// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Construction input for CoordinateFrame. Transient: fill the fields
// (designated initializers work), hand it to the descriptor constructor, and
// let validation happen there.
struct CoordinateFrameSpec {
  // Frame family. Required: kUnspecified is rejected at construction.
  CoordinateFrameType frame{CoordinateFrameType::kUnspecified};

  // Geodetic anchor for the local frames as (latitude deg, longitude deg,
  // height m above the ellipsoid). Geodetic, not geocentric, latitude.
  //
  // Not required here: the formation kernels ignore it, and a scene_local
  // frame without one is legal for simulation and unit tests. GeocodeOp and
  // the writers require it and reject a frame that omits it.
  std::optional<std::array<double, 3>> origin_llh{};

  // Seconds since a stated reference; the instant the frame is realized at.
  // Meaningful for ecef over long apertures, where the frame rotates
  // relative to inertial space.
  std::optional<double> epoch{};

  // Reference ellipsoid that origin_llh is expressed against.
  Ellipsoid ellipsoid{Ellipsoid::kWgs84};
};

// Declares the Cartesian frame that positions and sample geometry are
// expressed in, one of the module's shared config-plane descriptor types
// (see operators/common/README.md).
//
// Applied only as a boundary declaration: it fixes what an operator's
// positions *mean*, and no kernel reads it. Its value is that a frame is
// stated at every boundary rather than assumed, which makes a mismatch
// findable by inspection instead of only by looking at the image.
//
// Immutable value type validated once at construction, following the
// FMCWWaveformDescriptor pattern. The default constructor exists only for
// SDK plumbing (Parameter<T> requires default-constructible types) and
// yields an empty descriptor with is_valid() == false; consumers must
// reject it before use.
class CoordinateFrame {
 public:
  CoordinateFrame() = default;

  // Validates the spec. Throws std::invalid_argument naming the offending
  // field when the spec is incomplete or out of range.
  explicit CoordinateFrame(CoordinateFrameSpec spec) {
    if (spec.frame == CoordinateFrameType::kUnspecified) {
      throw std::invalid_argument(
          "CoordinateFrame: frame is required (ecef, enu, ned, scene_local)");
    }
    if (spec.origin_llh.has_value()) {
      const std::array<double, 3>& llh = *spec.origin_llh;
      for (double component : llh) {
        if (!std::isfinite(component)) {
          throw std::invalid_argument("CoordinateFrame: origin_llh components must be finite");
        }
      }
      if (llh[0] < -90.0 || llh[0] > 90.0) {
        throw std::invalid_argument(
            "CoordinateFrame: origin_llh latitude must be within [-90, 90] "
            "degrees (geodetic, not geocentric)");
      }
      if (llh[1] < -180.0 || llh[1] > 180.0) {
        throw std::invalid_argument(
            "CoordinateFrame: origin_llh longitude must be within "
            "[-180, 180] degrees");
      }
    }
    if (spec.epoch.has_value() && !std::isfinite(*spec.epoch)) {
      throw std::invalid_argument("CoordinateFrame: epoch must be finite");
    }
    frame_ = spec.frame;
    origin_llh_ = spec.origin_llh;
    epoch_ = spec.epoch;
    ellipsoid_ = spec.ellipsoid;
    valid_ = true;
  }

  // false only for a default-constructed (empty) descriptor.
  bool is_valid() const { return valid_; }

  CoordinateFrameType frame() const { return frame_; }

  // Geodetic anchor (latitude deg, longitude deg, height m); unset where the
  // frame carries no geodetic meaning.
  const std::optional<std::array<double, 3>>& origin_llh() const { return origin_llh_; }

  std::optional<double> epoch() const { return epoch_; }

  Ellipsoid ellipsoid() const { return ellipsoid_; }

  // True when both frames describe the same spatial reference, so positions
  // expressed in one may be consumed as the other. An empty descriptor is
  // compatible with nothing, itself included: two default-constructed frames
  // agree on every field while stating no reference at all, which is the
  // mismatch this check exists to catch.
  //
  // epoch is compared only for ecef, and only where either side sets it.
  // Comparing it unconditionally would reject perfectly valid local and
  // scene_local configurations, where the field carries no meaning and is
  // routinely left unset on one side.
  //
  // origin_llh is compared only where both sides state one, for the same
  // reason: omitting the anchor is legal (see CoordinateFrameSpec), so a
  // simulation frame is routinely paired with an anchored one describing the
  // same reference.
  bool compatible_with(const CoordinateFrame& other) const {
    if (!valid_ || !other.valid_) {
      return false;
    }
    if (frame_ != other.frame_ || ellipsoid_ != other.ellipsoid_) {
      return false;
    }
    if (origin_llh_.has_value() && other.origin_llh_.has_value() &&
        !anchors_agree(*origin_llh_, *other.origin_llh_)) {
      return false;
    }
    if (frame_ == CoordinateFrameType::kEcef && (epoch_.has_value() || other.epoch_.has_value())) {
      return epoch_ == other.epoch_;
    }
    return true;
  }

  friend bool operator==(const CoordinateFrame& a, const CoordinateFrame& b) {
    return a.valid_ == b.valid_ && a.frame_ == b.frame_ && a.origin_llh_ == b.origin_llh_ &&
           a.epoch_ == b.epoch_ && a.ellipsoid_ == b.ellipsoid_;
  }
  friend bool operator!=(const CoordinateFrame& a, const CoordinateFrame& b) { return !(a == b); }

 private:
  // Two anchors agree when they differ by less than the smallest difference
  // that could describe a different reference point. 1e-9 degrees is about
  // 0.1 mm of ground distance and 1e-6 m is a micrometre — well below any
  // surveyed anchor's own accuracy, and well above the last-bit disagreement
  // an anchor picks up from a YAML round trip or a degree/radian conversion.
  // Exact equality would call those distinct frames.
  static bool anchors_agree(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    constexpr double kAngleToleranceDeg = 1e-9;
    constexpr double kHeightToleranceM = 1e-6;
    return std::abs(a[0] - b[0]) <= kAngleToleranceDeg &&
           std::abs(a[1] - b[1]) <= kAngleToleranceDeg &&
           std::abs(a[2] - b[2]) <= kHeightToleranceM;
  }

  bool valid_{false};
  CoordinateFrameType frame_{CoordinateFrameType::kUnspecified};
  std::optional<std::array<double, 3>> origin_llh_{};
  std::optional<double> epoch_{};
  Ellipsoid ellipsoid_{Ellipsoid::kWgs84};
};

}  // namespace holoscan::radar
