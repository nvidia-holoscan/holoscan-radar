// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/angle_convention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace holoscan::radar {

namespace {

constexpr double kPi = 3.141592653589793;

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
      return rotate_z(from_canonical_angles(az, el), -kPi / 2.0);
    case AngleConventionPreset::kPhysicsSpherical: {
      // Zenith reference: boundary el is polar angle from +z.
      const double polar = el;
      const double horizon_el = kPi / 2.0 - polar;
      return from_canonical_angles(az, horizon_el);
    }
    case AngleConventionPreset::kCustom:
      throw std::invalid_argument(
          "AngleConvention: preset=custom boundary mapping is not implemented "
          "yet");
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
      CanonicalDirection rotated = rotate_z(dir, kPi / 2.0);
      to_canonical_angles(rotated, &az, &el);
      break;
    }
    case AngleConventionPreset::kPhysicsSpherical:
      el = kPi / 2.0 - el;
      break;
    case AngleConventionPreset::kCustom:
      throw std::invalid_argument(
          "AngleConvention: preset=custom boundary mapping is not implemented "
          "yet");
    case AngleConventionPreset::kDirectionCosine:
      break;
  }
  *out0 = az;
  *out1 = el;
}

}  // namespace holoscan::radar
