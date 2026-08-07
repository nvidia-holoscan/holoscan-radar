// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/angle_convention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar {

namespace {

CanonicalDirection from_canonical_angles(double az, double el) {
  const double cel = std::cos(el);
  return {cel * std::cos(az), cel * std::sin(az), std::sin(el)};
}

void to_canonical_angles(const CanonicalDirection& d, double* az, double* el) {
  const double r_xy = std::hypot(d.x, d.y);
  *az = std::atan2(d.y, d.x);
  *el = std::atan2(d.z, r_xy);
}

CanonicalDirection rotate_z(const CanonicalDirection& d, double delta) {
  const double c = std::cos(delta);
  const double s = std::sin(delta);
  return {c * d.x - s * d.y, s * d.x + c * d.y, d.z};
}

// Unit vector in the canonical frame for a signed principal axis.
CanonicalDirection axis_vec(SignedAxis axis) {
  switch (axis) {
    case SignedAxis::kPlusX:
      return {1.0, 0.0, 0.0};
    case SignedAxis::kMinusX:
      return {-1.0, 0.0, 0.0};
    case SignedAxis::kPlusY:
      return {0.0, 1.0, 0.0};
    case SignedAxis::kMinusY:
      return {0.0, -1.0, 0.0};
    case SignedAxis::kPlusZ:
      return {0.0, 0.0, 1.0};
    case SignedAxis::kMinusZ:
      return {0.0, 0.0, -1.0};
  }
  return {1.0, 0.0, 0.0};  // unreachable; every SignedAxis is handled
}

double dot(const CanonicalDirection& p, const CanonicalDirection& q) {
  return p.x * q.x + p.y * q.y + p.z * q.z;
}

// The custom-convention orthonormal frame: boresight (az=0,el=0), azimuth-
// positive, and elevation-positive axes. They must be three distinct principal
// axes (mutually orthogonal) — otherwise the (az, el) parameterization is
// degenerate.
struct CustomFrame {
  CanonicalDirection boresight;
  CanonicalDirection azimuth;
  CanonicalDirection elevation;
};

CustomFrame custom_frame(const AngleConvention& conv) {
  const CustomFrame f{axis_vec(conv.boresight_axis),
                      axis_vec(conv.azimuth_positive_axis),
                      axis_vec(conv.elevation_positive_axis)};
  constexpr double kOrthoEps = 1e-9;
  if (std::abs(dot(f.boresight, f.azimuth)) > kOrthoEps ||
      std::abs(dot(f.boresight, f.elevation)) > kOrthoEps ||
      std::abs(dot(f.azimuth, f.elevation)) > kOrthoEps) {
    throw std::invalid_argument(
        "AngleConvention: preset=custom requires boresight, azimuth, and "
        "elevation on three distinct (mutually orthogonal) principal axes");
  }
  return f;
}

}  // namespace

CanonicalDirection boundary_to_canonical(const AngleConvention& conv, double az,
                                         double el) {
  if (conv.uses_direction_cosine()) {
    const double u = az;
    const double v = el;
    if (!is_physical_direction_cosine(u, v)) {
      return {0.0, 0.0, 0.0};
    }
    const double w = std::sqrt(std::max(0.0, 1.0 - u * u - v * v));
    return {w, u, v};
  }

  switch (conv.preset) {
    case AngleConventionPreset::kRadarBoresight:
    case AngleConventionPreset::kRosRep103:
    case AngleConventionPreset::kIso8855Flu:
      return from_canonical_angles(az, el);
    case AngleConventionPreset::kAerospaceNed:
    case AngleConventionPreset::kGeodeticNed:
      // CW azimuth and positive-up elevation maps to −z in NED/FRD.
      return from_canonical_angles(-az, -el);
    case AngleConventionPreset::kGeodeticEnu:
      // Azimuth clockwise from North (+y): rotate boundary az by −π/2.
      return rotate_z(from_canonical_angles(az, el), -kHalfPi);
    case AngleConventionPreset::kPhysicsSpherical: {
      // Zenith reference: boundary el is polar angle from +z.
      const double polar = el;
      const double horizon_el = kHalfPi - polar;
      return from_canonical_angles(az, horizon_el);
    }
    case AngleConventionPreset::kCustom: {
      const CustomFrame f = custom_frame(conv);
      // Zenith reference: boundary el is the polar angle from the elevation
      // axis; convert to elevation-from-the-azimuth-plane before building.
      const bool zenith =
          conv.elevation_reference == ElevationReference::kZenith;
      const double el_h = zenith ? kHalfPi - el : el;
      const double ce = std::cos(el_h);
      const double se = std::sin(el_h);
      const double ca = std::cos(az);
      const double sa = std::sin(az);
      return {
          ce * ca * f.boresight.x + ce * sa * f.azimuth.x + se * f.elevation.x,
          ce * ca * f.boresight.y + ce * sa * f.azimuth.y + se * f.elevation.y,
          ce * ca * f.boresight.z + ce * sa * f.azimuth.z + se * f.elevation.z,
      };
    }
    case AngleConventionPreset::kDirectionCosine:
      break;
  }
  throw std::invalid_argument("AngleConvention: unknown preset");
}

void canonical_to_boundary(const AngleConvention& conv,
                           const CanonicalDirection& dir, double* out0,
                           double* out1) {
  if (conv.uses_direction_cosine()) {
    *out0 = dir.y;
    *out1 = dir.z;
    return;
  }

  double az = 0.0;
  double el = 0.0;
  to_canonical_angles(dir, &az, &el);

  switch (conv.preset) {
    case AngleConventionPreset::kRadarBoresight:
    case AngleConventionPreset::kRosRep103:
    case AngleConventionPreset::kIso8855Flu:
      break;
    case AngleConventionPreset::kAerospaceNed:
    case AngleConventionPreset::kGeodeticNed:
      az = -az;
      el = -el;
      break;
    case AngleConventionPreset::kGeodeticEnu: {
      CanonicalDirection rotated = rotate_z(dir, kHalfPi);
      to_canonical_angles(rotated, &az, &el);
      break;
    }
    case AngleConventionPreset::kPhysicsSpherical:
      el = kHalfPi - el;
      break;
    case AngleConventionPreset::kCustom: {
      // Recover (az, el) by projecting the canonical direction onto the custom
      // frame: cos(el)cos(az) = dir·boresight, cos(el)sin(az) = dir·azimuth,
      // sin(el) = dir·elevation.
      const CustomFrame f = custom_frame(conv);
      const double cb = dot(dir, f.boresight);
      const double ca = dot(dir, f.azimuth);
      const double ce = dot(dir, f.elevation);
      el = std::asin(std::clamp(ce, -1.0, 1.0));
      az = std::atan2(ca, cb);
      if (conv.elevation_reference == ElevationReference::kZenith) {
        el = kHalfPi - el;
      }
      break;
    }
    case AngleConventionPreset::kDirectionCosine:
      break;
  }
  *out0 = az;
  *out1 = el;
}

}  // namespace holoscan::radar
