// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SAR's three shared config-plane descriptors (radar_common/descriptors):
// CoordinateFrame's required frame family, range checks and the ecef-scoped
// epoch compatibility rule; SampleGeometry's axis normalization, positive
// spacing/counts, and sample_position convention; and
// LFMWaveformDescriptor's derivation and validation, which must stay
// identical to FMCWWaveformDescriptor's while the two types are converging
// (D24).

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"

namespace {

using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameSpec;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::Ellipsoid;
using holoscan::radar::LFMWaveformDescriptor;
using holoscan::radar::LFMWaveformSpec;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;

CoordinateFrameSpec test_frame_spec() {
  CoordinateFrameSpec spec;
  spec.frame = CoordinateFrameType::kEnu;
  spec.origin_llh = std::array<double, 3>{33.6407, -84.4277, 313.0};
  return spec;
}

SampleGeometrySpec test_geometry_spec() {
  SampleGeometrySpec spec;
  spec.origin = {100.0, 200.0, 0.0};
  // Deliberately un-normalized, to exercise normalization at construction.
  spec.axis_u = {2.0, 0.0, 0.0};
  spec.axis_v = {0.0, 5.0, 0.0};
  spec.spacing = {0.5, 0.25};
  spec.counts = {8, 4};
  return spec;
}

LFMWaveformSpec test_lfm_spec() {
  LFMWaveformSpec spec;
  spec.carrier_frequency = 9.6e9;
  spec.bandwidth = 600e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 750e6;
  spec.chirp_repetition_interval = 1.0 / 2000.0;
  return spec;
}

TEST(CoordinateFrameDescriptor, RequiresFrameFamilyAndDefaultsEllipsoid) {
  const CoordinateFrame frame(test_frame_spec());
  EXPECT_TRUE(frame.is_valid());
  EXPECT_EQ(frame.frame(), CoordinateFrameType::kEnu);
  EXPECT_EQ(frame.ellipsoid(), Ellipsoid::kWgs84);
  EXPECT_FALSE(frame.epoch().has_value());

  const CoordinateFrame empty;
  EXPECT_FALSE(empty.is_valid());

  CoordinateFrameSpec unspecified;
  EXPECT_THROW(CoordinateFrame{unspecified}, std::invalid_argument);
}

// A scene_local frame with no geodetic anchor is legal here — simulation and
// unit tests use exactly that. Rejecting it is the writers' job, not the
// descriptor's.
TEST(CoordinateFrameDescriptor, AnchorIsOptionalButRangeChecked) {
  CoordinateFrameSpec anchorless;
  anchorless.frame = CoordinateFrameType::kSceneLocal;
  const CoordinateFrame frame(anchorless);
  EXPECT_TRUE(frame.is_valid());
  EXPECT_FALSE(frame.origin_llh().has_value());

  auto bad_latitude = test_frame_spec();
  bad_latitude.origin_llh = std::array<double, 3>{90.5, 0.0, 0.0};
  EXPECT_THROW(CoordinateFrame{bad_latitude}, std::invalid_argument);

  auto bad_longitude = test_frame_spec();
  bad_longitude.origin_llh = std::array<double, 3>{0.0, -180.5, 0.0};
  EXPECT_THROW(CoordinateFrame{bad_longitude}, std::invalid_argument);

  auto not_finite = test_frame_spec();
  not_finite.origin_llh =
      std::array<double, 3>{0.0, 0.0, std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(CoordinateFrame{not_finite}, std::invalid_argument);

  auto bad_epoch = test_frame_spec();
  bad_epoch.epoch = std::numeric_limits<double>::infinity();
  EXPECT_THROW(CoordinateFrame{bad_epoch}, std::invalid_argument);
}

// epoch is compared only for ecef, and only where either side sets it.
// Comparing it unconditionally would reject valid local configurations where
// the field carries no meaning and is routinely left unset on one side.
TEST(CoordinateFrameDescriptor, EpochCompatibilityIsScopedToEcef) {
  auto local = test_frame_spec();
  const CoordinateFrame no_epoch(local);
  local.epoch = 1234.0;
  const CoordinateFrame with_epoch(local);
  EXPECT_TRUE(no_epoch.compatible_with(with_epoch));
  EXPECT_TRUE(with_epoch.compatible_with(no_epoch));
  EXPECT_NE(no_epoch, with_epoch);

  CoordinateFrameSpec ecef;
  ecef.frame = CoordinateFrameType::kEcef;
  const CoordinateFrame ecef_no_epoch(ecef);
  ecef.epoch = 1234.0;
  const CoordinateFrame ecef_epoch(ecef);
  ecef.epoch = 5678.0;
  const CoordinateFrame ecef_other_epoch(ecef);
  EXPECT_FALSE(ecef_no_epoch.compatible_with(ecef_epoch));
  EXPECT_FALSE(ecef_epoch.compatible_with(ecef_other_epoch));
  EXPECT_TRUE(ecef_epoch.compatible_with(ecef_epoch));

  auto different_anchor = test_frame_spec();
  different_anchor.origin_llh = std::array<double, 3>{0.0, 0.0, 0.0};
  EXPECT_FALSE(no_epoch.compatible_with(CoordinateFrame(different_anchor)));

  CoordinateFrameSpec ned = test_frame_spec();
  ned.frame = CoordinateFrameType::kNed;
  EXPECT_FALSE(no_epoch.compatible_with(CoordinateFrame(ned)));
}

TEST(SampleGeometryDescriptor, NormalizesAxesAndKeepsScaleInSpacing) {
  const SampleGeometry geometry(test_geometry_spec());
  EXPECT_TRUE(geometry.is_valid());
  EXPECT_DOUBLE_EQ(geometry.axis_u()[0], 1.0);
  EXPECT_DOUBLE_EQ(geometry.axis_v()[1], 1.0);
  EXPECT_EQ(geometry.num_v(), 8);
  EXPECT_EQ(geometry.num_u(), 4);
  EXPECT_EQ(geometry.num_samples(), 32);

  // Scaling an axis changes nothing: all the metric scale is in spacing.
  auto scaled = test_geometry_spec();
  scaled.axis_u = {1000.0, 0.0, 0.0};
  EXPECT_EQ(geometry, SampleGeometry(scaled));

  const SampleGeometry empty;
  EXPECT_FALSE(empty.is_valid());
}

// sample (iv, iu) sits at origin + iv*spacing[0]*v_hat + iu*spacing[1]*u_hat.
// BackprojectionOp and SICDWriterOp must agree on this pixel for pixel.
TEST(SampleGeometryDescriptor, SamplePositionFollowsTheSpacingOrder) {
  const SampleGeometry geometry(test_geometry_spec());
  const auto corner = geometry.sample_position(0, 0);
  EXPECT_DOUBLE_EQ(corner[0], 100.0);
  EXPECT_DOUBLE_EQ(corner[1], 200.0);

  const auto along_v = geometry.sample_position(2, 0);
  EXPECT_DOUBLE_EQ(along_v[0], 100.0);
  EXPECT_DOUBLE_EQ(along_v[1], 200.0 + 2 * 0.5);

  const auto along_u = geometry.sample_position(0, 3);
  EXPECT_DOUBLE_EQ(along_u[0], 100.0 + 3 * 0.25);
  EXPECT_DOUBLE_EQ(along_u[1], 200.0);
}

// A sheared, tilted lattice is legal — that is what lets a caller form onto
// a slant plane or a tilted facet. Only a zero-length axis is an error.
TEST(SampleGeometryDescriptor, AcceptsShearedAxesAndRejectsDegenerateOnes) {
  auto sheared = test_geometry_spec();
  sheared.axis_u = {1.0, 1.0, 0.0};
  sheared.axis_v = {0.0, 1.0, 1.0};
  EXPECT_NO_THROW(SampleGeometry{sheared});

  for (auto axis : {&SampleGeometrySpec::axis_u,
                    &SampleGeometrySpec::axis_v}) {
    auto zero = test_geometry_spec();
    zero.*axis = {0.0, 0.0, 0.0};
    EXPECT_THROW(SampleGeometry{zero}, std::invalid_argument);

    auto not_finite = test_geometry_spec();
    not_finite.*axis = {std::numeric_limits<double>::infinity(), 0.0, 0.0};
    EXPECT_THROW(SampleGeometry{not_finite}, std::invalid_argument);
  }
}

TEST(SampleGeometryDescriptor, RejectsNonPositiveSpacingAndCounts) {
  for (int i = 0; i < 2; ++i) {
    auto zero_spacing = test_geometry_spec();
    zero_spacing.spacing[i] = 0.0;
    EXPECT_THROW(SampleGeometry{zero_spacing}, std::invalid_argument);

    auto negative_spacing = test_geometry_spec();
    negative_spacing.spacing[i] = -1.0;
    EXPECT_THROW(SampleGeometry{negative_spacing}, std::invalid_argument);

    auto zero_count = test_geometry_spec();
    zero_count.counts[i] = 0;
    EXPECT_THROW(SampleGeometry{zero_count}, std::invalid_argument);

    auto negative_count = test_geometry_spec();
    negative_count.counts[i] = -4;
    EXPECT_THROW(SampleGeometry{negative_count}, std::invalid_argument);
  }

  auto bad_origin = test_geometry_spec();
  bad_origin.origin[2] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(SampleGeometry{bad_origin}, std::invalid_argument);
}

TEST(LFMWaveformDescriptorTest, DerivesSlopeBandwidthAndWavelength) {
  const LFMWaveformDescriptor by_bandwidth(test_lfm_spec());
  EXPECT_TRUE(by_bandwidth.is_valid());
  EXPECT_DOUBLE_EQ(by_bandwidth.slope(), 600e6 / 20e-6);
  EXPECT_DOUBLE_EQ(by_bandwidth.bandwidth(), 600e6);
  EXPECT_NEAR(by_bandwidth.wavelength(), 0.031228, 1e-6);

  auto slope_spec = test_lfm_spec();
  slope_spec.bandwidth.reset();
  slope_spec.chirp_slope = 600e6 / 20e-6;
  const LFMWaveformDescriptor by_slope(slope_spec);
  EXPECT_DOUBLE_EQ(by_slope.bandwidth(), 600e6);
  EXPECT_EQ(by_bandwidth, by_slope);

  const LFMWaveformDescriptor empty;
  EXPECT_FALSE(empty.is_valid());
}

TEST(LFMWaveformDescriptorTest, RejectsIncompleteOrInconsistentSpecs) {
  auto neither = test_lfm_spec();
  neither.bandwidth.reset();
  EXPECT_THROW(LFMWaveformDescriptor{neither}, std::invalid_argument);

  auto inconsistent = test_lfm_spec();
  inconsistent.chirp_slope = 1e12;
  EXPECT_THROW(LFMWaveformDescriptor{inconsistent}, std::invalid_argument);

  for (auto field : {&LFMWaveformSpec::carrier_frequency,
                     &LFMWaveformSpec::chirp_duration,
                     &LFMWaveformSpec::sample_rate,
                     &LFMWaveformSpec::chirp_repetition_interval}) {
    auto zero = test_lfm_spec();
    zero.*field = 0.0;
    EXPECT_THROW(LFMWaveformDescriptor{zero}, std::invalid_argument);

    auto not_finite = test_lfm_spec();
    not_finite.*field = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(LFMWaveformDescriptor{not_finite}, std::invalid_argument);
  }

  auto overflow = test_lfm_spec();
  overflow.bandwidth = std::numeric_limits<double>::max();
  overflow.chirp_duration = 1e-12;
  EXPECT_THROW(LFMWaveformDescriptor{overflow}, std::invalid_argument);
}

// D24 keeps the two chirp descriptors separate but intends them to merge, so
// they must agree field for field on the same inputs. A divergence here is
// the drift the convergence note warns about, caught rather than silent.
TEST(LFMWaveformDescriptorTest, MatchesFMCWWaveformDescriptorFieldForField) {
  const LFMWaveformDescriptor lfm(test_lfm_spec());

  holoscan::radar::FMCWWaveformSpec fmcw_spec;
  fmcw_spec.carrier_frequency = test_lfm_spec().carrier_frequency;
  fmcw_spec.bandwidth = test_lfm_spec().bandwidth;
  fmcw_spec.chirp_duration = test_lfm_spec().chirp_duration;
  fmcw_spec.sample_rate = test_lfm_spec().sample_rate;
  fmcw_spec.chirp_repetition_interval =
      test_lfm_spec().chirp_repetition_interval;
  const holoscan::radar::FMCWWaveformDescriptor fmcw(fmcw_spec);

  EXPECT_DOUBLE_EQ(lfm.carrier_frequency(), fmcw.carrier_frequency());
  EXPECT_DOUBLE_EQ(lfm.bandwidth(), fmcw.bandwidth());
  EXPECT_DOUBLE_EQ(lfm.slope(), fmcw.slope());
  EXPECT_DOUBLE_EQ(lfm.chirp_duration(), fmcw.chirp_duration());
  EXPECT_DOUBLE_EQ(lfm.sample_rate(), fmcw.sample_rate());
  EXPECT_DOUBLE_EQ(lfm.chirp_repetition_interval(),
                   fmcw.chirp_repetition_interval());
  EXPECT_DOUBLE_EQ(lfm.wavelength(), fmcw.wavelength());
}

}  // namespace
