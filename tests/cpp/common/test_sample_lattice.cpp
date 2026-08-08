// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Uniform-lattice expansion (radar_common/dsp/sample_lattice.hpp): the
// row-major (num_v, num_u, 3) layout the formation kernels consume, its
// agreement with SampleGeometry::sample_position, and the overflow and
// validity guards.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/dsp/sample_lattice.hpp"

namespace {

using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::expand_sample_lattice;
using holoscan::radar::sample_lattice_size;

SampleGeometry test_geometry() {
  SampleGeometrySpec spec;
  spec.origin = {10.0, -5.0, 2.0};
  spec.axis_u = {1.0, 0.0, 0.0};
  spec.axis_v = {0.0, 0.0, 1.0};  // A vertical v axis: no horizontal
                                  // assumption is baked in anywhere.
  spec.spacing = {2.0, 0.5};
  spec.counts = {3, 4};
  return SampleGeometry(spec);
}

TEST(SampleLattice, ExpandsRowMajorAndAgreesWithSamplePosition) {
  const SampleGeometry geometry = test_geometry();
  EXPECT_EQ(sample_lattice_size(geometry), 3 * 4 * 3);

  const std::vector<double> points = expand_sample_lattice(geometry);
  ASSERT_EQ(points.size(), static_cast<size_t>(3 * 4 * 3));

  for (int64_t iv = 0; iv < geometry.num_v(); ++iv) {
    for (int64_t iu = 0; iu < geometry.num_u(); ++iu) {
      const std::array<double, 3> expected = geometry.sample_position(iv, iu);
      const size_t base =
          static_cast<size_t>((iv * geometry.num_u() + iu) * 3);
      EXPECT_DOUBLE_EQ(points[base + 0], expected[0]);
      EXPECT_DOUBLE_EQ(points[base + 1], expected[1]);
      EXPECT_DOUBLE_EQ(points[base + 2], expected[2]);
    }
  }

  // Spot-check the convention itself rather than only its self-consistency:
  // the fastest-varying axis is u, and v advances by spacing[0].
  EXPECT_DOUBLE_EQ(points[3], 10.0 + 0.5);   // (0, 1) moves along u
  EXPECT_DOUBLE_EQ(points[4 * 3 + 2], 2.0 + 2.0);  // (1, 0) moves along v
}

// The pointer overload is what a hot path uses: it casts per element rather
// than building a double buffer, which for a DEM-scale lattice is the
// difference between one allocation and two.
TEST(SampleLattice, PointerOverloadCastsToTheCallerPrecision) {
  const SampleGeometry geometry = test_geometry();
  const std::vector<double> reference = expand_sample_lattice(geometry);

  std::vector<float> single(reference.size());
  expand_sample_lattice(geometry, single.data());
  for (size_t i = 0; i < reference.size(); ++i) {
    EXPECT_FLOAT_EQ(single[i], static_cast<float>(reference[i]));
  }

  EXPECT_THROW(expand_sample_lattice<float>(geometry, nullptr),
               std::invalid_argument);
}

TEST(SampleLattice, RejectsInvalidGeometryAndOverflowingCounts) {
  const SampleGeometry empty;
  EXPECT_THROW(sample_lattice_size(empty), std::invalid_argument);
  EXPECT_THROW(expand_sample_lattice(empty), std::invalid_argument);

  // A count pair can overflow the point-list size long before the
  // allocation would fail, so the guard is on the product, not on malloc.
  SampleGeometrySpec huge;
  huge.origin = {0.0, 0.0, 0.0};
  huge.axis_u = {1.0, 0.0, 0.0};
  huge.axis_v = {0.0, 1.0, 0.0};
  huge.spacing = {1.0, 1.0};
  huge.counts = {std::numeric_limits<int64_t>::max() / 2, 4};
  EXPECT_THROW(sample_lattice_size(SampleGeometry(huge)),
               std::invalid_argument);
}

}  // namespace
