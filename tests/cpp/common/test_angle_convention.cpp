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
