// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host unit tests for the S3 sub-array selection helper (holey-lattice
// detection, largest filled ULA/URA sub-block, explicit selection, rejections).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/subarray_selection.hpp"

namespace {
using holoscan::radar::ArrayGeometry;
using holoscan::radar::ArrayGeometryType;
using holoscan::radar::build_subarray_selection;

constexpr double kD = 0.5;  // element spacing (m)

// A linear array from integer lattice positions along y (holes = skipped ints).
ArrayGeometry make_linear(const std::vector<int>& iy) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (int i : iy) {
    g.element_positions.insert(g.element_positions.end(),
                               {0.0, i * kD, 0.0});
  }
  return g;
}

// A planar array from integer (iy, iz) lattice cells (holes = omitted cells).
ArrayGeometry make_planar(const std::vector<std::pair<int, int>>& cells) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (auto [iy, iz] : cells) {
    g.element_positions.insert(g.element_positions.end(),
                               {0.0, iy * kD, iz * kD});
  }
  return g;
}

std::set<std::pair<double, double>> yz_set(const ArrayGeometry& g) {
  std::set<std::pair<double, double>> s;
  for (int64_t e = 0; e < g.num_elements(); ++e) {
    s.insert({g.element_positions[e * 3 + 1], g.element_positions[e * 3 + 2]});
  }
  return s;
}

TEST(SubarraySelection, FilledUlaNoReduction) {
  const auto g = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  const auto sel = build_subarray_selection(g, {}, 2);
  EXPECT_FALSE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 8);
  const std::vector<int32_t> id = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(sel.channel_index, id);
  EXPECT_EQ(sel.reduced_geometry, g);
}

TEST(SubarraySelection, FilledUnorderedUlaRequiresReorder) {
  const auto g = make_linear({2, 0, 1, 3});
  const auto sel = build_subarray_selection(g, {}, 2);
  EXPECT_TRUE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 4);
  const std::vector<int32_t> expected = {1, 2, 0, 3};
  EXPECT_EQ(sel.channel_index, expected);
  EXPECT_EQ(sel.reduced_geometry.element_positions,
            make_linear({0, 1, 2, 3}).element_positions);
}

TEST(SubarraySelection, HoleyUlaLongestRun) {
  // Filled at lattice 0,1,2,3, hole at 4, filled 5,6 → longest run is 0..3.
  const auto g = make_linear({0, 1, 2, 3, 5, 6});
  const auto sel = build_subarray_selection(g, {}, 2);
  EXPECT_TRUE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 4);
  const std::vector<int32_t> id = {0, 1, 2, 3};
  EXPECT_EQ(sel.channel_index, id);
  EXPECT_EQ(sel.reduced_geometry.geometry_type, ArrayGeometryType::kUla);
}

TEST(SubarraySelection, HoleyUlaNonUnitGaps) {
  // Gaps {2,1,1,3} → gcd unit 1; lattice 0,2,3,4,7. Longest run 2,3,4.
  const auto g = make_linear({0, 2, 3, 4, 7});
  const auto sel = build_subarray_selection(g, {}, 2);
  EXPECT_TRUE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 3);
  const std::vector<int32_t> id = {1, 2, 3};  // input indices of y=2,3,4
  EXPECT_EQ(sel.channel_index, id);
}

TEST(SubarraySelection, HoleyUraLargestRectangle) {
  // Full 3x3 minus corner (iy=2, iz=2). Largest all-filled rectangle is the
  // 3(az) x 2(el) block iz in {0,1} (area 6), preferred over 2x3 by tie-break.
  std::vector<std::pair<int, int>> cells;
  for (int iz = 0; iz < 3; ++iz) {
    for (int iy = 0; iy < 3; ++iy) {
      if (iy == 2 && iz == 2) {
        continue;
      }
      cells.push_back({iy, iz});
    }
  }
  const auto g = make_planar(cells);
  const auto sel = build_subarray_selection(g, {}, 2);
  EXPECT_TRUE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 6);
  EXPECT_EQ(sel.reduced_geometry.geometry_type, ArrayGeometryType::kUra);
  // Selected block: all iy in {0,1,2}, iz in {0,1}.
  std::set<std::pair<double, double>> expect;
  for (int iz = 0; iz < 2; ++iz) {
    for (int iy = 0; iy < 3; ++iy) {
      expect.insert({iy * kD, iz * kD});
    }
  }
  EXPECT_EQ(yz_set(sel.reduced_geometry), expect);
}

TEST(SubarraySelection, ExplicitSelectionReordersToLattice) {
  const auto g = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  // Scrambled order — must come back in lattice order 2,3,4,5.
  const auto sel = build_subarray_selection(g, {5, 2, 4, 3}, 2);
  EXPECT_TRUE(sel.reduced);
  EXPECT_EQ(sel.subarray_size, 4);
  const std::vector<int32_t> id = {2, 3, 4, 5};
  EXPECT_EQ(sel.channel_index, id);
}

TEST(SubarraySelection, ExplicitRejectsHoleySet) {
  const auto g = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  // {0,1,3} is holey (gap at lattice 2) → not a filled ULA.
  EXPECT_THROW(build_subarray_selection(g, {0, 1, 3}, 2),
               std::invalid_argument);
}

TEST(SubarraySelection, ExplicitRejectsOutOfRangeAndDuplicate) {
  const auto g = make_linear({0, 1, 2, 3});
  EXPECT_THROW(build_subarray_selection(g, {0, 9}, 2), std::invalid_argument);
  EXPECT_THROW(build_subarray_selection(g, {1, 1}, 2), std::invalid_argument);
}

TEST(SubarraySelection, RejectsWhenBelowMinSize) {
  const auto g = make_linear({0, 1, 2, 3});
  EXPECT_THROW(build_subarray_selection(g, {}, 5), std::invalid_argument);
}

}  // namespace
