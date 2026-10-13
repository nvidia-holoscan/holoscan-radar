// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FMCW waveform descriptor (radar_common/descriptors): derivation of slope,
// bandwidth, and wavelength at construction, plus validation of incomplete,
// inconsistent, and non-finite specs.

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/dsp/shift_invariant_subarray.hpp"

namespace {

using holoscan::radar::FMCWWaveformDescriptor;
using holoscan::radar::ShiftInvariantSubarrayPair;
using holoscan::radar::ArrayGeometry;
using holoscan::radar::ArrayGeometryType;

holoscan::radar::FMCWWaveformSpec test_waveform_spec() {
  holoscan::radar::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = 150e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  return spec;
}

TEST(WaveformDescriptor, DerivesSlopeAndBandwidthAtConstruction) {
  const FMCWWaveformDescriptor by_bandwidth(test_waveform_spec());
  EXPECT_TRUE(by_bandwidth.is_valid());
  EXPECT_DOUBLE_EQ(by_bandwidth.slope(), 150e6 / 20e-6);
  EXPECT_DOUBLE_EQ(by_bandwidth.bandwidth(), 150e6);

  auto slope_spec = test_waveform_spec();
  slope_spec.bandwidth.reset();
  slope_spec.chirp_slope = 150e6 / 20e-6;
  const FMCWWaveformDescriptor by_slope(slope_spec);
  EXPECT_DOUBLE_EQ(by_slope.slope(), 150e6 / 20e-6);
  EXPECT_DOUBLE_EQ(by_slope.bandwidth(), 150e6);
  // Bandwidth- and slope-spelled specs describe the same waveform, so the
  // canonical descriptors compare equal (config-cache semantics).
  EXPECT_EQ(by_bandwidth, by_slope);
}

TEST(WaveformDescriptor, ConstructionRejectsIncompleteOrInconsistentSpecs) {
  auto neither = test_waveform_spec();
  neither.bandwidth.reset();
  EXPECT_THROW(FMCWWaveformDescriptor{neither}, std::invalid_argument);

  auto inconsistent = test_waveform_spec();
  inconsistent.chirp_slope = 1e12;  // != bandwidth / chirp_duration
  EXPECT_THROW(FMCWWaveformDescriptor{inconsistent}, std::invalid_argument);

  for (auto field : {&holoscan::radar::FMCWWaveformSpec::carrier_frequency,
                     &holoscan::radar::FMCWWaveformSpec::chirp_duration,
                     &holoscan::radar::FMCWWaveformSpec::sample_rate,
                     &holoscan::radar::FMCWWaveformSpec::chirp_repetition_interval}) {
    auto spec = test_waveform_spec();
    spec.*field = 0.0;
    EXPECT_THROW(FMCWWaveformDescriptor{spec}, std::invalid_argument);
  }
}

TEST(WaveformDescriptor, ConstructionRejectsNonFiniteValues) {
  // NaN compares false against every bound, so finiteness is enforced
  // explicitly; both spellings of the sweep and every scalar are covered.
  for (double bad :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    for (auto field : {&holoscan::radar::FMCWWaveformSpec::carrier_frequency,
                       &holoscan::radar::FMCWWaveformSpec::chirp_duration,
                       &holoscan::radar::FMCWWaveformSpec::sample_rate,
                       &holoscan::radar::FMCWWaveformSpec::chirp_repetition_interval}) {
      auto spec = test_waveform_spec();
      spec.*field = bad;
      EXPECT_THROW(FMCWWaveformDescriptor{spec}, std::invalid_argument);
    }
    auto bad_bandwidth = test_waveform_spec();
    bad_bandwidth.bandwidth = bad;
    EXPECT_THROW(FMCWWaveformDescriptor{bad_bandwidth}, std::invalid_argument);
    auto bad_slope = test_waveform_spec();
    bad_slope.bandwidth.reset();
    bad_slope.chirp_slope = bad;
    EXPECT_THROW(FMCWWaveformDescriptor{bad_slope}, std::invalid_argument);
  }

  // Finite inputs whose derivation overflows are rejected too.
  auto overflow = test_waveform_spec();
  overflow.bandwidth = std::numeric_limits<double>::max();
  overflow.chirp_duration = 1e-12;
  EXPECT_THROW(FMCWWaveformDescriptor{overflow}, std::invalid_argument);
}

TEST(WaveformDescriptor, WavelengthPrecomputedAndDefaultInvalid) {
  const FMCWWaveformDescriptor wf(test_waveform_spec());
  EXPECT_NEAR(wf.wavelength(), 3.8934e-3, 1e-6);
  EXPECT_NEAR(wf.wavelength(2.0 * 299792458.0), 2.0 * wf.wavelength(), 1e-9);

  const FMCWWaveformDescriptor empty;
  EXPECT_FALSE(empty.is_valid());
}

TEST(ShiftInvariantSubarrayPair, AutoAndExplicitValidation) {
  const ShiftInvariantSubarrayPair automatic;
  EXPECT_TRUE(automatic.is_auto());
  EXPECT_NO_THROW(automatic.validate(8));

  ShiftInvariantSubarrayPair pair;
  pair.reference_indices = {0, 1, 2};
  pair.shifted_indices = {1, 2, 3};
  EXPECT_FALSE(pair.is_auto());
  EXPECT_NO_THROW(pair.validate());
  EXPECT_NO_THROW(pair.validate(4));
}

TEST(ShiftInvariantSubarrayPair, RejectsMalformedIndices) {
  ShiftInvariantSubarrayPair pair;
  pair.reference_indices = {0, 1};
  EXPECT_THROW(pair.validate(), std::invalid_argument);

  pair.shifted_indices = {1};
  EXPECT_THROW(pair.validate(), std::invalid_argument);

  pair.shifted_indices = {1, 2};
  pair.reference_indices = {0, 0};
  EXPECT_THROW(pair.validate(), std::invalid_argument);

  pair.reference_indices = {0, 1};
  pair.shifted_indices = {1, 1};
  EXPECT_THROW(pair.validate(), std::invalid_argument);

  pair.shifted_indices = {1, 4};
  EXPECT_THROW(pair.validate(4), std::invalid_argument);
}

ArrayGeometry make_test_ula(int64_t channels) {
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  geom.geometry_type = ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geom.element_positions.insert(geom.element_positions.end(),
                                  {0.0, 0.5 * static_cast<double>(channel), 0.0});
  }
  return geom;
}

TEST(ShiftInvariantSubarrayPlan, AutoDerivesMaximumOverlapUlaPair) {
  const auto geom = make_test_ula(6);
  const auto plan =
      holoscan::radar::build_shift_invariant_subarray_plan(geom, ShiftInvariantSubarrayPair{});
  ASSERT_EQ(plan.pair.reference_indices.size(), 5u);
  ASSERT_EQ(plan.pair.shifted_indices.size(), 5u);
  EXPECT_EQ(plan.pair.reference_indices.front(), 0);
  EXPECT_EQ(plan.pair.shifted_indices.back(), 5);
  EXPECT_NEAR(plan.spacing, 0.5, 1e-9);
}

TEST(ShiftInvariantSubarrayPlan, RejectsParallelToBoresightUla) {
  ArrayGeometry geom;
  geom.wavelength = 1.0;
  geom.geometry_type = ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < 4; ++channel) {
    geom.element_positions.insert(geom.element_positions.end(),
                                  {0.5 * static_cast<double>(channel), 0.0, 0.0});
  }
  EXPECT_THROW(
      holoscan::radar::build_shift_invariant_subarray_plan(geom, ShiftInvariantSubarrayPair{}),
      std::invalid_argument);
}

}  // namespace
