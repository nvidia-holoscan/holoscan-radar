// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

TEST(SimoRdHeatmapTest, MapsDopplerToXAndRangeUpward) {
  const auto low_range =
      viz::detail::rd_plot_position(1.0F, 0.0F, 4, 8);
  const auto high_range =
      viz::detail::rd_plot_position(1.0F, 7.0F, 4, 8);
  const auto later_doppler =
      viz::detail::rd_plot_position(2.0F, 0.0F, 4, 8);
  EXPECT_LT(low_range[0], later_doppler[0]);
  EXPECT_GT(low_range[1], high_range[1]);
}

TEST(SimoDashboardTest, ComputesClusterCentroids) {
  const std::vector<viz::ClusterCentroid> centroids =
      viz::detail::compute_cluster_centroids(
          {0, 2, 2, 4, 3, 6},
          {10.0F, 1.0F, 0.1F, 0.2F, 1000.0F,
           14.0F, 3.0F, 0.3F, 0.4F, -1000.0F,
           30.0F, -2.0F, -0.2F, -0.3F, 7.0F},
          {0, 0, 1}, {2, 1}, 2, 5);
  ASSERT_EQ(centroids.size(), 2U);
  EXPECT_FLOAT_EQ(centroids[0].doppler_bin, 1.0F);
  EXPECT_FLOAT_EQ(centroids[0].range_bin, 3.0F);
  EXPECT_FLOAT_EQ(centroids[0].features[0], 12.0F);
  EXPECT_FLOAT_EQ(centroids[0].features[3], 0.3F);
  EXPECT_EQ(centroids[0].color_bucket, 0);
  EXPECT_EQ(centroids[1].color_bucket, 1);
}

TEST(SimoDashboardTest, UsesReducedFeaturesForHistory) {
  std::vector<viz::ClusterCentroid> centroids(2);
  viz::detail::apply_reduced_features(
      &centroids,
      {10.0F, 1.0F, 0.1F, 0.2F, 3.0F,
       20.0F, 2.0F, 0.3F, 0.4F, 4.0F},
      5);
  EXPECT_EQ(centroids[0].features,
            (std::array<float, 4>{10.0F, 1.0F, 0.1F, 0.2F}));
  EXPECT_EQ(centroids[1].features,
            (std::array<float, 4>{20.0F, 2.0F, 0.3F, 0.4F}));
}

TEST(SimoDashboardTest, BoundsHistoryAndMapsPanelCoordinates) {
  std::deque<viz::HistorySample> history;
  const viz::ClusterCentroid centroid{
      .doppler_bin = 7.0F,
      .range_bin = 11.0F,
      .features = {10.0F, 0.0F, 0.25F, -0.25F},
      .color_bucket = 3};
  for (std::int64_t frame = 0; frame < 5; ++frame) {
    viz::detail::append_history(&history, {centroid}, frame, 3);
  }
  ASSERT_EQ(history.size(), 3U);
  EXPECT_EQ(history.front().frame, 2);
  EXPECT_FLOAT_EQ(history.front().doppler_bin, 7.0F);
  EXPECT_FLOAT_EQ(history.front().range_bin, 11.0F);
  const auto position = viz::detail::history_plot_position(
      history.back(), 4, 3, 2, {-0.5F, 0.5F});
  EXPECT_FLOAT_EQ(position[0], 1.0F);
  EXPECT_FLOAT_EQ(position[1], 0.75F);
}

TEST(SimoDashboardTest, ExpiresDetectionAfterConfiguredFrames) {
  std::deque<viz::HistorySample> history;
  const viz::ClusterCentroid centroid{
      .doppler_bin = 7.0F,
      .range_bin = 11.0F,
      .features = {10.0F, 0.0F, 0.25F, -0.25F},
  };
  viz::detail::append_history(&history, {centroid}, 0, 3);
  viz::detail::append_history(&history, {}, 1, 3);
  viz::detail::append_history(&history, {}, 2, 3);
  ASSERT_EQ(history.size(), 1U);
  viz::detail::append_history(&history, {}, 3, 3);
  EXPECT_TRUE(history.empty());
}

TEST(SimoDashboardTest, DerivesMovingTargetTruthFeatures) {
  const viz::TruthTarget target{
      .position = {1.0, 0.0, 0.0},
      .velocity = {0.0, 1.0, 1.0},
  };
  const auto features = viz::detail::truth_features_at_time(target, 1.0);
  EXPECT_NEAR(features[0], std::sqrt(3.0F), 1.0e-6F);
  EXPECT_NEAR(features[1], 2.0F / std::sqrt(3.0F), 1.0e-6F);
  EXPECT_NEAR(features[2], std::acos(-1.0F) / 4.0F, 1.0e-6F);
  EXPECT_NEAR(features[3], std::asin(1.0F / std::sqrt(3.0F)), 1.0e-6F);
}

}  // namespace
