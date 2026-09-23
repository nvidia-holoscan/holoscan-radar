// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host unit tests for the spatial-smoothing planner (lattice detection,
// sub-array enumeration, reduced geometry, forward-backward flip, and the
// rejection of non-uniform / holey / mis-specified geometries).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"

namespace {
using holoscan::radar::ArrayGeometry;
using holoscan::radar::ArrayGeometryType;
using holoscan::radar::build_spatial_smoothing_plan;

ArrayGeometry make_ula(int64_t n, double spacing = 0.5, double wl = 1.0) {
  ArrayGeometry g;
  g.wavelength = wl;
  g.geometry_type = ArrayGeometryType::kUla;
  for (int64_t m = 0; m < n; ++m) {
    g.element_positions.push_back(0.0);
    g.element_positions.push_back(static_cast<double>(m) * spacing);
    g.element_positions.push_back(0.0);
  }
  return g;
}

// URA in the y-z plane, element order (iz outer, iy inner) — the ordering the
// example builders and MVDR tests use; the planner must be robust to it.
ArrayGeometry make_ura(int64_t ny, int64_t nz, double spacing = 0.5, double wl = 1.0) {
  ArrayGeometry g;
  g.wavelength = wl;
  g.geometry_type = ArrayGeometryType::kUra;
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      g.element_positions.push_back(0.0);
      g.element_positions.push_back(static_cast<double>(iy) * spacing);
      g.element_positions.push_back(static_cast<double>(iz) * spacing);
    }
  }
  return g;
}

TEST(SpatialSmoothingPlan, UlaForwardEnumeration) {
  const auto g = make_ula(8);
  const auto plan = build_spatial_smoothing_plan(g, {3}, /*fb=*/false);
  ASSERT_TRUE(plan.enabled);
  EXPECT_FALSE(plan.forward_backward);
  EXPECT_EQ(plan.num_channels, 8);
  EXPECT_EQ(plan.num_subarrays, 3);  // P = 8 - 6 + 1
  EXPECT_EQ(plan.subarray_size, 6);  // L = 8 - 3 + 1
  EXPECT_EQ(plan.looks_per_snapshot(), 3);

  // Three maximally-overlapping windows: [0..5], [1..6], [2..7].
  const std::vector<int32_t> expect = {0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(plan.subarray_index, expect);

  // Reduced geometry is the reference sub-array (elements 0..5).
  ASSERT_EQ(plan.reduced_geometry.num_elements(), 6);
  EXPECT_EQ(plan.reduced_geometry.geometry_type, ArrayGeometryType::kUla);
  for (int64_t l = 0; l < 6; ++l) {
    EXPECT_DOUBLE_EQ(plan.reduced_geometry.element_positions[l * 3 + 1],
                     static_cast<double>(l) * 0.5);
  }
}

TEST(SpatialSmoothingPlan, UlaForwardBackwardFlip) {
  const auto g = make_ula(8);
  const auto plan = build_spatial_smoothing_plan(g, {3}, /*fb=*/true);
  ASSERT_TRUE(plan.enabled);
  EXPECT_TRUE(plan.forward_backward);
  EXPECT_EQ(plan.looks_per_snapshot(), 6);  // 2P
  // 1-D flip reverses the L=6 local order.
  const std::vector<int32_t> expect = {5, 4, 3, 2, 1, 0};
  EXPECT_EQ(plan.fb_flip, expect);
}

TEST(SpatialSmoothingPlan, PassThroughWhenAllOnesNoFb) {
  const auto g = make_ula(8);
  const auto plan = build_spatial_smoothing_plan(g, {1}, /*fb=*/false);
  EXPECT_FALSE(plan.enabled);
}

TEST(SpatialSmoothingPlan, PureForwardBackwardNoReduction) {
  const auto g = make_ula(8);
  const auto plan = build_spatial_smoothing_plan(g, {1}, /*fb=*/true);
  ASSERT_TRUE(plan.enabled);
  EXPECT_EQ(plan.num_subarrays, 1);
  EXPECT_EQ(plan.subarray_size, 8);
  EXPECT_EQ(plan.looks_per_snapshot(), 2);  // forward + backward on full array
  const std::vector<int32_t> flip = {7, 6, 5, 4, 3, 2, 1, 0};
  EXPECT_EQ(plan.fb_flip, flip);
}

TEST(SpatialSmoothingPlan, UraForwardStructure) {
  const auto g = make_ura(4, 4);  // 16 elements
  const auto plan = build_spatial_smoothing_plan(g, {2, 2}, /*fb=*/false);
  ASSERT_TRUE(plan.enabled);
  EXPECT_EQ(plan.num_subarrays, 4);  // 2 * 2
  EXPECT_EQ(plan.subarray_size, 9);  // 3 * 3
  EXPECT_EQ(plan.looks_per_snapshot(), 4);
  ASSERT_EQ(plan.reduced_geometry.num_elements(), 9);

  // Every sub-array references 9 distinct valid channels.
  for (int64_t p = 0; p < plan.num_subarrays; ++p) {
    std::set<int32_t> uniq;
    for (int64_t l = 0; l < plan.subarray_size; ++l) {
      const int32_t ch = plan.subarray_index[p * plan.subarray_size + l];
      EXPECT_GE(ch, 0);
      EXPECT_LT(ch, 16);
      uniq.insert(ch);
    }
    EXPECT_EQ(uniq.size(), 9u);
  }

  // Reduced geometry is a filled 3x3 sub-grid (distinct positions).
  std::set<std::pair<double, double>> pos;
  for (int64_t l = 0; l < 9; ++l) {
    pos.insert({plan.reduced_geometry.element_positions[l * 3 + 1],
                plan.reduced_geometry.element_positions[l * 3 + 2]});
  }
  EXPECT_EQ(pos.size(), 9u);
}

TEST(SpatialSmoothingPlan, UraForwardBackward2DFlipIsInvolution) {
  const auto g = make_ura(4, 3);  // 12 elements
  const auto plan = build_spatial_smoothing_plan(g, {2, 2}, /*fb=*/true);
  ASSERT_TRUE(plan.enabled);
  ASSERT_EQ(static_cast<int64_t>(plan.fb_flip.size()), plan.subarray_size);
  // J is its own inverse: fb_flip[fb_flip[l]] == l.
  for (int64_t l = 0; l < plan.subarray_size; ++l) {
    EXPECT_EQ(plan.fb_flip[static_cast<size_t>(plan.fb_flip[l])], l);
  }
}

TEST(SpatialSmoothingPlan, RejectsNonUniformSpacing) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  const double ys[4] = {0.0, 0.5, 1.0, 2.0};  // gap at 1.5
  for (double y : ys) {
    g.element_positions.insert(g.element_positions.end(), {0.0, y, 0.0});
  }
  EXPECT_THROW(build_spatial_smoothing_plan(g, {2}, false), std::invalid_argument);
}

TEST(SpatialSmoothingPlan, RejectsHoleyLattice) {
  // 3x3 grid missing the last element → 8 elements, not a filled 3x3.
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (int iz = 0; iz < 3; ++iz) {
    for (int iy = 0; iy < 3; ++iy) {
      if (iz == 2 && iy == 2) {
        continue;
      }
      g.element_positions.insert(g.element_positions.end(), {0.0, iy * 0.5, iz * 0.5});
    }
  }
  EXPECT_THROW(build_spatial_smoothing_plan(g, {2, 2}, false), std::invalid_argument);
}

TEST(SpatialSmoothingPlan, RejectsCountLengthMismatch) {
  const auto g = make_ula(8);
  EXPECT_THROW(build_spatial_smoothing_plan(g, {2, 2}, false), std::invalid_argument);
  const auto u = make_ura(4, 4);
  EXPECT_THROW(build_spatial_smoothing_plan(u, {2}, false), std::invalid_argument);
}

TEST(SpatialSmoothingPlan, RejectsCountOutOfRange) {
  const auto g = make_ula(8);
  EXPECT_THROW(build_spatial_smoothing_plan(g, {9}, false), std::invalid_argument);
  EXPECT_THROW(build_spatial_smoothing_plan(g, {0}, false), std::invalid_argument);
}

TEST(SpatialSmoothingPlan, RejectsEmptyCounts) {
  const auto g = make_ula(8);
  EXPECT_THROW(build_spatial_smoothing_plan(g, {}, false), std::invalid_argument);
}

}  // namespace
