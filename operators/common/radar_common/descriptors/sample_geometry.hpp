// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace holoscan::radar {

// Construction input for SampleGeometry. Transient: fill the fields
// (designated initializers work), hand it to the descriptor constructor, and
// let validation and axis normalization happen there.
struct SampleGeometrySpec {
  // Position of sample (0, 0) in the declared CoordinateFrame, in meters.
  std::array<double, 3> origin{};

  // Vectors along the two lattice axes. They need not be axis-aligned,
  // horizontal, or mutually orthogonal — that is what lets a caller form
  // onto a slant plane, a tilted facet, or a sheared lattice. Any non-zero
  // vector is accepted and normalized at construction; a zero-length axis
  // is an error. All the metric scale lives in `spacing`.
  std::array<double, 3> axis_u{};
  std::array<double, 3> axis_v{};

  // Meters per sample, (v, u) in shape order.
  std::array<double, 2> spacing{};

  // (num_v, num_u), which is the image shape.
  std::array<int64_t, 2> counts{};
};

// Describes where in space each image pixel sits: a uniform lattice given by
// an origin, two unit axes, a spacing and a count. One of the module's
// shared config-plane descriptor types (see operators/common/README.md).
//
// Bundling these four fields into one descriptor is a correctness
// requirement rather than tidiness. As independent [live] override ports, a
// caller moving to a differently-sized scene sends several messages and
// nothing guarantees they arrive on the same tick, so formation could reset
// repeatedly for one logical change or — worse — form pulses against a new
// origin with an old extent. One value makes the update atomic by
// construction.
//
// Explicit per-pixel lattices are NOT this type: a DEM-conformal lattice is
// (num_v, num_u, 3) and can reach hundreds of megabytes, which is not
// something a config-plane value type should carry. Those stay a tensor on
// BackprojectionOp::sample_points.
//
// Immutable value type validated once at construction. The default
// constructor exists only for SDK plumbing (Parameter<T> requires
// default-constructible types) and yields an empty descriptor with
// is_valid() == false; consumers must reject it before use.
class SampleGeometry {
 public:
  SampleGeometry() = default;

  // Validates the spec and normalizes both axes. Throws
  // std::invalid_argument naming the offending field.
  explicit SampleGeometry(SampleGeometrySpec spec) {
    for (double component : spec.origin) {
      if (!std::isfinite(component)) {
        throw std::invalid_argument(
            "SampleGeometry: origin components must be finite");
      }
    }
    axis_u_ = normalize(spec.axis_u, "axis_u");
    axis_v_ = normalize(spec.axis_v, "axis_v");
    for (int i = 0; i < 2; ++i) {
      if (!std::isfinite(spec.spacing[i]) || spec.spacing[i] <= 0.0) {
        throw std::invalid_argument(
            "SampleGeometry: spacing[" + std::to_string(i) +
            "] must be finite and > 0 (spacing is (v, u) in shape order)");
      }
      if (spec.counts[i] <= 0) {
        throw std::invalid_argument(
            "SampleGeometry: counts[" + std::to_string(i) +
            "] must be > 0 (counts is (num_v, num_u))");
      }
    }
    origin_ = spec.origin;
    spacing_ = spec.spacing;
    counts_ = spec.counts;
    valid_ = true;
  }

  // false only for a default-constructed (empty) descriptor.
  bool is_valid() const { return valid_; }

  const std::array<double, 3>& origin() const { return origin_; }

  // Unit vector along the u (column) axis.
  const std::array<double, 3>& axis_u() const { return axis_u_; }

  // Unit vector along the v (row) axis.
  const std::array<double, 3>& axis_v() const { return axis_v_; }

  // Meters per sample, (v, u).
  const std::array<double, 2>& spacing() const { return spacing_; }

  // (num_v, num_u).
  const std::array<int64_t, 2>& counts() const { return counts_; }

  int64_t num_v() const { return counts_[0]; }
  int64_t num_u() const { return counts_[1]; }

  // Total sample count, num_v * num_u.
  int64_t num_samples() const { return counts_[0] * counts_[1]; }

  // Position of sample (iv, iu):
  //   origin + iv * spacing[0] * v_hat + iu * spacing[1] * u_hat
  //
  // Stated as an accessor because BackprojectionOp and SICDWriterOp must
  // agree on it pixel for pixel. This is also SICD's own convention, where
  // the Row/Col unit vectors carry direction and spacing carries length.
  std::array<double, 3> sample_position(int64_t iv, int64_t iu) const {
    const double dv = static_cast<double>(iv) * spacing_[0];
    const double du = static_cast<double>(iu) * spacing_[1];
    return {origin_[0] + dv * axis_v_[0] + du * axis_u_[0],
            origin_[1] + dv * axis_v_[1] + du * axis_u_[1],
            origin_[2] + dv * axis_v_[2] + du * axis_u_[2]};
  }

  friend bool operator==(const SampleGeometry& a, const SampleGeometry& b) {
    return a.valid_ == b.valid_ && a.origin_ == b.origin_ &&
           a.axis_u_ == b.axis_u_ && a.axis_v_ == b.axis_v_ &&
           a.spacing_ == b.spacing_ && a.counts_ == b.counts_;
  }
  friend bool operator!=(const SampleGeometry& a, const SampleGeometry& b) {
    return !(a == b);
  }

 private:
  static std::array<double, 3> normalize(const std::array<double, 3>& axis,
                                         const char* name) {
    double norm_squared = 0.0;
    for (double component : axis) {
      if (!std::isfinite(component)) {
        throw std::invalid_argument(std::string("SampleGeometry: ") + name +
                                    " components must be finite");
      }
      norm_squared += component * component;
    }
    const double norm = std::sqrt(norm_squared);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      throw std::invalid_argument(std::string("SampleGeometry: ") + name +
                                  " must be a non-zero vector");
    }
    return {axis[0] / norm, axis[1] / norm, axis[2] / norm};
  }

  bool valid_{false};
  std::array<double, 3> origin_{};
  std::array<double, 3> axis_u_{};
  std::array<double, 3> axis_v_{};
  std::array<double, 2> spacing_{};
  std::array<int64_t, 2> counts_{};
};

}  // namespace holoscan::radar
