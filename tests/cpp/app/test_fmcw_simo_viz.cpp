// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "viz/simo_viz_ops.hpp"

namespace {

namespace viz = holoscan::radar::app::simo::viz;

TEST(SimoRdHeatmapTest, MapsPeakAndDynamicRange) {
  const std::vector<std::uint8_t> rgba =
      viz::detail::make_heatmap_rgba_reference(
          {1.0F, 1000000.0F}, {1, 2}, 60.0F);
  ASSERT_EQ(rgba.size(), 8U);
  EXPECT_EQ(rgba[0], 0U);
  EXPECT_EQ(rgba[1], 0U);
  EXPECT_EQ(rgba[2], 255U);
  EXPECT_EQ(rgba[3], 255U);
  EXPECT_EQ(rgba[4], 255U);
  EXPECT_EQ(rgba[5], 0U);
  EXPECT_EQ(rgba[6], 0U);
  EXPECT_EQ(rgba[7], 255U);
}

TEST(SimoRdHeatmapTest, DenseLeadingAxesUseMaxProjection) {
  const std::vector<std::uint8_t> dense =
      viz::detail::make_heatmap_rgba_reference(
          {1.0F, 10.0F, 100.0F, 1.0F}, {2, 1, 1, 2}, 20.0F);
  const std::vector<std::uint8_t> projected =
      viz::detail::make_heatmap_rgba_reference(
          {100.0F, 10.0F}, {1, 2}, 20.0F);
  EXPECT_EQ(dense, projected);
}

TEST(SimoClusterGeometryTest, UsesTrailingSparseAndDenseIndices) {
  const std::vector<float> sparse = viz::detail::make_cluster_ovals(
      {1, 2}, {0}, 2, 4, 8, 0.01F);
  const std::vector<float> dense = viz::detail::make_cluster_ovals(
      {9, 7, 1, 2}, {3}, 4, 4, 8, 0.01F);
  EXPECT_EQ(sparse, dense);
  ASSERT_EQ(sparse.size(), 4U);
  EXPECT_FLOAT_EQ(sparse[0], 0.3125F);
  EXPECT_FLOAT_EQ(sparse[1], 0.375F);
}

TEST(SimoClusterGeometryTest, FiltersNoiseLabels) {
  const std::vector<float> ovals = viz::detail::make_cluster_ovals(
      {0, 0, 1, 1, 2, 2}, {0, -1, 1}, 2, 4, 4, 0.02F);
  ASSERT_EQ(ovals.size(), 8U);
  EXPECT_FLOAT_EQ(ovals[0], 0.125F);
  EXPECT_FLOAT_EQ(ovals[4], 0.625F);
}

TEST(SimoClusterGeometryTest, BuildsRepresentativeCrosses) {
  const std::vector<float> crosses =
      viz::detail::make_representative_crosses(
          {4, 5, 2, 3}, 2, 8, 10, 0.03F);
  ASSERT_EQ(crosses.size(), 6U);
  EXPECT_FLOAT_EQ(crosses[0], 0.55F);
  EXPECT_FLOAT_EQ(crosses[1], 0.5625F);
  EXPECT_FLOAT_EQ(crosses[2], 0.03F);
}

}  // namespace
