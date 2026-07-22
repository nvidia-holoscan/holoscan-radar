// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// AngleConvention custom-preset mapping tests. The strong anchor is
// preset-equivalence: every built-in preset is expressible as a `custom` axis
// configuration, and the generic mapping must reproduce that preset's hardcoded
// boundary<->canonical transform (both directions).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/angle_convention.hpp"

namespace {
constexpr double kPi = 3.141592653589793;

using holoscan::radar::AngleConvention;
using holoscan::radar::AngleConventionPreset;
using holoscan::radar::boundary_to_canonical;
using holoscan::radar::canonical_to_boundary;
using holoscan::radar::CanonicalDirection;
using holoscan::radar::ElevationReference;
using holoscan::radar::SignedAxis;

AngleConvention preset(AngleConventionPreset p) {
  AngleConvention c;
  c.preset = p;
  return c;
}

AngleConvention custom(SignedAxis b, SignedAxis a, SignedAxis e,
                       ElevationReference ref = ElevationReference::kHorizon) {
  AngleConvention c;
  c.preset = AngleConventionPreset::kCustom;
  c.boresight_axis = b;
  c.azimuth_positive_axis = a;
  c.elevation_positive_axis = e;
  c.elevation_reference = ref;
  return c;
}

// az/el sample grid; el kept away from the ±π/2 poles where az is undefined.
const std::vector<double>& az_grid() {
  static const std::vector<double> g{-2.0, -1.0, -0.3, 0.0, 0.5, 1.5};
  return g;
}
const std::vector<double>& el_grid() {
  static const std::vector<double> g{-1.2, -0.5, 0.0, 0.4, 1.0};
  return g;
}

void expect_forward_matches(const AngleConvention& a, const AngleConvention& b,
                            const std::string& label) {
  for (double az : az_grid()) {
    for (double el : el_grid()) {
      const CanonicalDirection da = boundary_to_canonical(a, az, el);
      const CanonicalDirection db = boundary_to_canonical(b, az, el);
      EXPECT_NEAR(da.x, db.x, 1e-12) << label << " az=" << az << " el=" << el;
      EXPECT_NEAR(da.y, db.y, 1e-12) << label << " az=" << az << " el=" << el;
      EXPECT_NEAR(da.z, db.z, 1e-12) << label << " az=" << az << " el=" << el;
    }
  }
}

void expect_inverse_matches(const AngleConvention& a, const AngleConvention& b,
                            const std::string& label) {
  for (double az : az_grid()) {
    for (double el : el_grid()) {
      const double ce = std::cos(el);
      // A canonical unit direction to invert (built independently of a/b).
      const CanonicalDirection dir{ce * std::cos(az), ce * std::sin(az),
                                   std::sin(el)};
      double az_a = 0.0, el_a = 0.0, az_b = 0.0, el_b = 0.0;
      canonical_to_boundary(a, dir, &az_a, &el_a);
      canonical_to_boundary(b, dir, &az_b, &el_b);
      EXPECT_NEAR(az_a, az_b, 1e-12) << label << " dir az=" << az;
      EXPECT_NEAR(el_a, el_b, 1e-12) << label << " dir el=" << el;
    }
  }
}

TEST(AngleConventionCustomTest, ReproducesEachPreset) {
  struct Case {
    AngleConventionPreset preset;
    AngleConvention equivalent;
    const char* name;
  };
  const std::vector<Case> cases{
      {AngleConventionPreset::kRadarBoresight,
       custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ),
       "radar_boresight"},
      {AngleConventionPreset::kRosRep103,
       custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ),
       "ros_rep103"},
      {AngleConventionPreset::kIso8855Flu,
       custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ),
       "iso8855_flu"},
      {AngleConventionPreset::kAerospaceNed,
       custom(SignedAxis::kPlusX, SignedAxis::kMinusY, SignedAxis::kMinusZ),
       "aerospace_ned"},
      {AngleConventionPreset::kGeodeticNed,
       custom(SignedAxis::kPlusX, SignedAxis::kMinusY, SignedAxis::kMinusZ),
       "geodetic_ned"},
      {AngleConventionPreset::kGeodeticEnu,
       custom(SignedAxis::kMinusY, SignedAxis::kPlusX, SignedAxis::kPlusZ),
       "geodetic_enu"},
      {AngleConventionPreset::kPhysicsSpherical,
       custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ,
              ElevationReference::kZenith),
       "physics_spherical"},
  };
  for (const Case& c : cases) {
    const AngleConvention p = preset(c.preset);
    expect_forward_matches(p, c.equivalent, std::string(c.name) + " forward");
    expect_inverse_matches(p, c.equivalent, std::string(c.name) + " inverse");
  }
}

void expect_round_trips(const AngleConvention& conv,
                        const std::vector<double>& els) {
  for (double az : az_grid()) {
    for (double el : els) {
      const CanonicalDirection dir = boundary_to_canonical(conv, az, el);
      double az_rt = 0.0, el_rt = 0.0;
      canonical_to_boundary(conv, dir, &az_rt, &el_rt);
      EXPECT_NEAR(az, az_rt, 1e-9) << "az=" << az << " el=" << el;
      EXPECT_NEAR(el, el_rt, 1e-9) << "az=" << az << " el=" << el;
    }
  }
}

TEST(AngleConventionCustomTest, RoundTripsBoundaryCanonicalBoundary) {
  // Horizon frames (non-trivial valid axis permutations): el ∈ (−π/2, π/2).
  expect_round_trips(
      custom(SignedAxis::kPlusY, SignedAxis::kPlusZ, SignedAxis::kPlusX),
      el_grid());
  expect_round_trips(
      custom(SignedAxis::kMinusZ, SignedAxis::kPlusX, SignedAxis::kMinusY),
      el_grid());
  // Zenith frame: el is the polar angle in (0, π), so round-trip over a polar
  // grid (negative/horizon el is out of domain for a zenith reference).
  expect_round_trips(
      custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ,
             ElevationReference::kZenith),
      {0.3, 0.8, 1.4, 2.0, 2.6});
}

// Principal azimuth in (−π, π] for the value `a` (period 2π).
double wrap_to_pi(double a) {
  const double two_pi = 2.0 * kPi;
  while (a > kPi) {
    a -= two_pi;
  }
  while (a <= -kPi) {
    a += two_pi;
  }
  return a;
}

TEST(AngleConventionCustomTest, ForwardPeriodicInAzimuth) {
  // boundary_to_canonical is built from cos/sin, so az and az ± 2π yield the
  // identical unit vector — any real azimuth is valid input.
  const AngleConvention conv =
      custom(SignedAxis::kMinusY, SignedAxis::kPlusX, SignedAxis::kPlusZ);
  const double two_pi = 2.0 * kPi;
  for (double az : az_grid()) {
    for (double el : el_grid()) {
      const CanonicalDirection d0 = boundary_to_canonical(conv, az, el);
      const CanonicalDirection d1 =
          boundary_to_canonical(conv, az + two_pi, el);
      const CanonicalDirection d2 =
          boundary_to_canonical(conv, az - two_pi, el);
      for (auto* d : {&d1, &d2}) {
        EXPECT_NEAR(d0.x, d->x, 1e-12) << "az=" << az << " el=" << el;
        EXPECT_NEAR(d0.y, d->y, 1e-12) << "az=" << az << " el=" << el;
        EXPECT_NEAR(d0.z, d->z, 1e-12) << "az=" << az << " el=" << el;
      }
    }
  }
}

TEST(AngleConventionCustomTest, InverseAzimuthWrapsToPrincipalRange) {
  // canonical_to_boundary uses atan2/asin, so it always returns the principal
  // azimuth in (−π, π]. An out-of-range input azimuth round-trips to its
  // wrapped principal value (not the original), and an in-range elevation is
  // preserved.
  const AngleConvention conv =
      custom(SignedAxis::kPlusX, SignedAxis::kPlusY, SignedAxis::kPlusZ);
  const std::vector<double> wrapped_az{-6.0, -4.0, 3.0, 4.0, 5.5};
  for (double az : wrapped_az) {
    for (double el : el_grid()) {
      const CanonicalDirection dir = boundary_to_canonical(conv, az, el);
      double az_out = 0.0, el_out = 0.0;
      canonical_to_boundary(conv, dir, &az_out, &el_out);
      EXPECT_GT(az_out, -kPi - 1e-9) << "az=" << az;
      EXPECT_LE(az_out, kPi + 1e-9) << "az=" << az;
      EXPECT_NEAR(az_out, wrap_to_pi(az), 1e-9) << "az=" << az;
      EXPECT_NEAR(el_out, el, 1e-9) << "el=" << el;
    }
  }
}

TEST(AngleConventionCustomTest, RejectsDegenerateAxes) {
  // Two roles on the same principal axis (repeated, or same unsigned axis).
  const AngleConvention repeated =
      custom(SignedAxis::kPlusX, SignedAxis::kPlusX, SignedAxis::kPlusZ);
  const AngleConvention same_axis =
      custom(SignedAxis::kPlusX, SignedAxis::kMinusX, SignedAxis::kPlusZ);
  EXPECT_THROW(boundary_to_canonical(repeated, 0.1, 0.2),
               std::invalid_argument);
  EXPECT_THROW(boundary_to_canonical(same_axis, 0.1, 0.2),
               std::invalid_argument);
  double a = 0.0, b = 0.0;
  EXPECT_THROW(canonical_to_boundary(repeated, {1.0, 0.0, 0.0}, &a, &b),
               std::invalid_argument);
}

}  // namespace
