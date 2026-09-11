// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
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

TEST(SimoDashboardTest, ComputesClusterCentroids) {
  const std::vector<viz::ClusterCentroid> centroids =
      viz::detail::compute_cluster_centroids(
          {0, 2, 2, 4, 3, 6},
          {10.0F, 1.0F, 0.1F, 0.2F,
           14.0F, 3.0F, 0.3F, 0.4F,
           30.0F, -2.0F, -0.2F, -0.3F},
          {0, 0, 1}, {2, 1}, 2, 4);
  ASSERT_EQ(centroids.size(), 2U);
  EXPECT_FLOAT_EQ(centroids[0].doppler_bin, 1.0F);
  EXPECT_FLOAT_EQ(centroids[0].range_bin, 3.0F);
  EXPECT_FLOAT_EQ(centroids[0].features[0], 12.0F);
  EXPECT_FLOAT_EQ(centroids[0].features[3], 0.3F);
  EXPECT_EQ(centroids[0].color_bucket, 0);
  EXPECT_EQ(centroids[1].color_bucket, 1);
}

TEST(SimoDashboardTest, BoundsHistoryAndMapsPanelCoordinates) {
  std::deque<viz::HistorySample> history;
  const viz::ClusterCentroid centroid{
      .features = {10.0F, 0.0F, 0.25F, -0.25F},
      .color_bucket = 3};
  for (std::int64_t frame = 0; frame < 5; ++frame) {
    viz::detail::append_history(&history, {centroid}, frame, 3);
  }
  ASSERT_EQ(history.size(), 3U);
  EXPECT_EQ(history.front().frame, 2);
  const auto position = viz::detail::history_plot_position(
      history.back(), 4, 3, 2, {-0.5F, 0.5F});
  EXPECT_FLOAT_EQ(position[0], 1.0F);
  EXPECT_FLOAT_EQ(position[1], 0.75F);
}

}  // namespace
