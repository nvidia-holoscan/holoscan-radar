// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host unit tests for the EFT source-count helper (radar_common/dsp/
// source_count.hpp). See that header for the algorithm and its fidelity
// note (closed-form profile prediction, Gaussian-approximated threshold).

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "radar_common/dsp/source_count.hpp"

namespace {
using holoscan::radar::eft_source_count;

TEST(SourceCountTest, PureNoiseFloorsAtOneSource) {
  // Ascending eigenvalues with modest (+/-10%) jitter around a common noise
  // floor - no eigenvalue breaks from the profile, so the whole array should
  // be classified noise and the result floors at the contract minimum of 1.
  const std::vector<double> eig = {0.90, 0.95, 1.00, 1.05, 1.10};
  EXPECT_EQ(eft_source_count(eig, /*num_snapshots=*/20, /*pfa=*/0.1,
                             /*max_sources=*/4),
           1);
}

TEST(SourceCountTest, TwoDominantEigenvaluesDetectedAsTwoSources) {
  // Two eigenvalues far above the noise floor (50x, 100x) - unambiguous at
  // any reasonable threshold.
  const std::vector<double> eig = {0.90, 0.95, 1.00, 1.05, 50.0, 100.0};
  EXPECT_EQ(eft_source_count(eig, /*num_snapshots=*/20, /*pfa=*/0.1,
                             /*max_sources=*/4),
           2);
}

TEST(SourceCountTest, MaxSourcesClamps) {
  const std::vector<double> eig = {0.90, 0.95, 1.00, 1.05, 50.0, 100.0};
  EXPECT_EQ(eft_source_count(eig, /*num_snapshots=*/20, /*pfa=*/0.1,
                             /*max_sources=*/1),
           1);
}

TEST(SourceCountTest, LowerPfaIsAtLeastAsConservative) {
  // A tighter (smaller) Pfa budget raises every eta_P, so the test should
  // never report MORE sources than a looser Pfa on the same data.
  const std::vector<double> eig = {0.90, 0.95, 1.00, 1.05, 3.0, 100.0};
  const int64_t loose =
      eft_source_count(eig, /*num_snapshots=*/20, /*pfa=*/0.5, 4);
  const int64_t tight =
      eft_source_count(eig, /*num_snapshots=*/20, /*pfa=*/1e-4, 4);
  EXPECT_LE(tight, loose);
}

TEST(SourceCountTest, MoreSnapshotsSharpensDetection) {
  // A marginal eigenvalue bump should be easier to detect (more likely
  // classified signal) with more snapshots (tighter eta_P), all else equal.
  const std::vector<double> eig = {0.90, 0.95, 1.00, 1.05, 1.3};
  const int64_t few = eft_source_count(eig, /*num_snapshots=*/4, 0.1, 4);
  const int64_t many = eft_source_count(eig, /*num_snapshots=*/2000, 0.1, 4);
  EXPECT_GE(many, few);
}

TEST(SourceCountTest, DegenerateAllZeroDoesNotThrow) {
  const std::vector<double> eig = {0.0, 0.0, 0.0, 0.0};
  EXPECT_NO_THROW({
    const int64_t d = eft_source_count(eig, 10, 0.1, 4);
    EXPECT_GE(d, 1);
    EXPECT_LE(d, 4);
  });
}

TEST(SourceCountTest, TrivialArraysFloorAtOne) {
  EXPECT_EQ(eft_source_count({1.0}, 10, 0.1, 4), 1);
  EXPECT_EQ(eft_source_count({1.0, 2.0}, 10, 0.1, 4), 1);
}

TEST(SourceCountTest, RejectsEmptyEigenvalues) {
  EXPECT_THROW(eft_source_count({}, 10, 0.1, 4), std::invalid_argument);
}

TEST(SourceCountTest, RejectsNonPositiveSnapshots) {
  const std::vector<double> eig = {1.0, 1.0, 1.0};
  EXPECT_THROW(eft_source_count(eig, 0, 0.1, 4), std::invalid_argument);
  EXPECT_THROW(eft_source_count(eig, -1, 0.1, 4), std::invalid_argument);
}

TEST(SourceCountTest, RejectsOutOfRangePfa) {
  const std::vector<double> eig = {1.0, 1.0, 1.0};
  EXPECT_THROW(eft_source_count(eig, 10, 0.0, 4), std::invalid_argument);
  EXPECT_THROW(eft_source_count(eig, 10, 1.0, 4), std::invalid_argument);
  EXPECT_THROW(eft_source_count(eig, 10, -0.1, 4), std::invalid_argument);
}

TEST(SourceCountTest, RejectsNonPositiveMaxSources) {
  const std::vector<double> eig = {1.0, 1.0, 1.0};
  EXPECT_THROW(eft_source_count(eig, 10, 0.1, 0), std::invalid_argument);
  EXPECT_THROW(eft_source_count(eig, 10, 0.1, -1), std::invalid_argument);
}

}  // namespace
