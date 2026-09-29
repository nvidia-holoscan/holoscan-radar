// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/app/angle_presentation.hpp"
#include "radar_app/viz/fmcw_dashboard_op.hpp"
#include "radar_app/viz/rgba_image_bridge_op.hpp"

namespace {

namespace viz = holoscan::radar::app::viz;
namespace radar = holoscan::radar;

class RgbaGraphSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(rgba).produces_tensor(holoscan::TensorOutputSpec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = DLDataType{kDLUInt, 8U, 1U},
                .rank = 3U,
                .strides = holoscan::TensorStridePolicy::kContiguousOnly,
            },
        .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = 16U},
    });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::Tensor> rgba{"rgba"};
};

TEST(FmcwDashboardTest, ConnectsRgbaTensorToHsdk5HolovizImageLayer) {
  holoscan::Graph graph{"dashboard-image-bridge-test"};
  auto source = graph.op<RgbaGraphSourceOp>("source");
  auto bridge = graph.op<viz::RgbaImageBridgeOp>("bridge", 2, 2);
  holoscan::visualization::LayerStack layers;
  layers.add_image_layer("image",
                         holoscan::visualization::ImageLayerDesc{
                             .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm,
                             .max_width = 2U,
                             .max_height = 2U});
  auto sink = graph.op<holoscan::visualization::HolovizOp>(
      "holoviz", std::move(layers),
      holoscan::visualization::WindowSinkDesc{.title = "test", .width = 2U, .height = 2U});
  graph.partition("application").add(source).add(bridge);
  graph.partition("visualization").add(sink);
  graph.add_flow(source->rgba, bridge->rgba);
  graph.add_flow(bridge->image, sink->image_layer_input(0));
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, holoscan::CompileOptions{});
  EXPECT_TRUE(plan.ok()) << plan.json();
}

TEST(AnglePresentationTest, ConvertsDirectionCosinesToPhysicalAngles) {
  holoscan::radar::AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kDirectionCosine;
  const auto angles = holoscan::radar::app::to_physical_angles(convention, 0.5, 0.25);
  EXPECT_NEAR(angles.az, std::atan2(0.5, std::sqrt(0.6875)), 1e-12);
  EXPECT_NEAR(angles.el, std::asin(0.25), 1e-12);
}

TEST(AnglePresentationTest, RejectsNonPhysicalDirectionCosinePair) {
  holoscan::radar::AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kDirectionCosine;
  EXPECT_THROW(holoscan::radar::app::to_physical_angles(convention, 0.9, 0.9),
               std::invalid_argument);
}

TEST(AnglePresentationTest, ComputesBoundsFromValidGridCombinations) {
  holoscan::radar::AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kDirectionCosine;
  const auto bounds = holoscan::radar::app::physical_scan_grid_bounds(convention, {-1.0, 0.0, 1.0},
                                                                      {-1.0, 0.0, 1.0});
  EXPECT_NEAR(bounds.az[0], -1.5707963267948966, 1e-12);
  EXPECT_NEAR(bounds.az[1], 1.5707963267948966, 1e-12);
  EXPECT_NEAR(bounds.el[0], -1.5707963267948966, 1e-12);
  EXPECT_NEAR(bounds.el[1], 1.5707963267948966, 1e-12);
}

TEST(AnglePresentationTest, ConvertsPhysicsSphericalToRadarBoresight) {
  holoscan::radar::AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kPhysicsSpherical;
  const auto angles = holoscan::radar::app::to_physical_angles(convention, 0.3, 1.2);
  EXPECT_NEAR(angles.az, 0.3, 1e-12);
  EXPECT_NEAR(angles.el, 1.5707963267948966 - 1.2, 1e-12);
}

TEST(AnglePresentationTest, CentroidHandlesAzimuthWrap) {
  holoscan::radar::app::PhysicalAngleCentroid centroid;
  centroid.add(holoscan::radar::AngleConvention{}, 3.13, 0.1);
  centroid.add(holoscan::radar::AngleConvention{}, -3.13, 0.1);
  const auto angles = centroid.mean();
  EXPECT_GT(std::abs(angles.az), 3.0);
  EXPECT_NEAR(angles.el, 0.1, 1e-5);
}

TEST(FmcwDashboardTest, MatchesExactSelectedWaveformAndRejectsMissing) {
  const holoscan::radar::FMCWWaveformDescriptor near(
      holoscan::radar::FMCWWaveformSpec{.carrier_frequency = 77.0e9,
                                        .bandwidth = 300.0e6,
                                        .chirp_duration = 20.0e-6,
                                        .sample_rate = 12.8e6,
                                        .chirp_repetition_interval = 50.0e-6});
  const holoscan::radar::FMCWWaveformDescriptor far(
      holoscan::radar::FMCWWaveformSpec{.carrier_frequency = 77.0e9,
                                        .bandwidth = 150.0e6,
                                        .chirp_duration = 40.0e-6,
                                        .sample_rate = 12.8e6,
                                        .chirp_repetition_interval = 90.0e-6});
  const auto near_wire = holoscan::radar::to_wire(near);
  const auto far_wire = holoscan::radar::to_wire(far);

  auto match = viz::detail::selected_profile_matches(near_wire, "near", near);
  ASSERT_TRUE(match);
  EXPECT_TRUE(*match);
  auto mismatch = viz::detail::selected_profile_matches(far_wire, "near", near);
  ASSERT_TRUE(mismatch);
  EXPECT_FALSE(*mismatch);

  const std::optional<holoscan::radar::wire::FMCWWaveformT> missing;
  auto absent = viz::detail::selected_profile_matches(missing, "near", near);
  EXPECT_FALSE(absent);
  EXPECT_EQ(absent.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto unprofiled = viz::detail::selected_profile_matches(missing, "", {});
  ASSERT_TRUE(unprofiled);
  EXPECT_TRUE(*unprofiled);
}

TEST(FmcwRdHeatmapTest, MapsPeakAndDynamicRange) {
  const std::vector<std::uint8_t> rgba =
      viz::detail::make_heatmap_rgba_reference({1.0F, 1000000.0F}, {1, 2}, 60.0F);
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

TEST(FmcwRdHeatmapTest, DenseLeadingAxesUseMaxProjection) {
  const std::vector<std::uint8_t> dense =
      viz::detail::make_heatmap_rgba_reference({1.0F, 10.0F, 100.0F, 1.0F}, {2, 1, 1, 2}, 20.0F);
  const std::vector<std::uint8_t> projected =
      viz::detail::make_heatmap_rgba_reference({100.0F, 10.0F}, {1, 2}, 20.0F);
  EXPECT_EQ(dense, projected);
}

TEST(FmcwRdHeatmapTest, MapsDopplerToXAndRangeUpward) {
  const auto low_range = viz::detail::rd_plot_position(1.0F, 0.0F, 4, 8);
  const auto high_range = viz::detail::rd_plot_position(1.0F, 7.0F, 4, 8);
  const auto later_doppler = viz::detail::rd_plot_position(2.0F, 0.0F, 4, 8);
  EXPECT_LT(low_range[0], later_doppler[0]);
  EXPECT_GT(low_range[1], high_range[1]);
}

TEST(FmcwDashboardTest, UsesUpwardSemicircleForAtMost180DegreeCoverage) {
  constexpr float kHalfPi = 1.57079632679489661923F;
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-0.3F, 0.3F}));
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-kHalfPi, kHalfPi}));
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-2.0F, 0.0F}));
  EXPECT_FALSE(viz::detail::use_azimuth_sector({-2.0F, 2.0F}));
}

TEST(FmcwDashboardTest, ComputesClusterCentroids) {
  const std::vector<viz::DetectionMarker> markers =
      viz::detail::make_cluster_markers({0, 2, 2, 4, 3, 6},
                                        {10.0F, 1.0F, 0.1F, 0.2F, 1000.0F, 14.0F, 3.0F, 0.3F, 0.4F,
                                         -1000.0F, 30.0F, -2.0F, -0.2F, -0.3F, 7.0F},
                                        {0, 0, 1}, {2, 1}, 2, 5);
  ASSERT_EQ(markers.size(), 2U);
  EXPECT_FLOAT_EQ(markers[0].doppler_bin, 1.0F);
  EXPECT_FLOAT_EQ(markers[0].range_bin, 3.0F);
  EXPECT_FLOAT_EQ(markers[0].features[0], 12.0F);
  EXPECT_NEAR(markers[0].features[3], 0.3015F, 0.002F);
  EXPECT_EQ(markers[0].color_bucket, 0);
  EXPECT_EQ(markers[1].color_bucket, 1);
}

TEST(FmcwDashboardTest, ClusterCentroidHandlesAzimuthWrap) {
  const auto markers = viz::detail::make_cluster_markers(
      {0, 1, 0, 1}, {10.0F, 1.0F, 3.13F, 0.1F, 10.0F, 1.0F, -3.13F, 0.1F}, {0, 0}, {2}, 2, 4);
  ASSERT_EQ(markers.size(), 1U);
  EXPECT_GT(std::abs(markers[0].features[2]), 3.0F);
  EXPECT_NEAR(markers[0].features[3], 0.1F, 1e-5F);
}

TEST(FmcwDashboardTest, MakesRawDetectionMarkers) {
  const std::vector<viz::DetectionMarker> markers = viz::detail::make_raw_detection_markers(
      {0, 1, 2, 3, 1, 0, 4, 5},
      {14.0F, 3.0F, 0.3F, 0.4F, -1000.0F, 10.0F, 1.0F, 0.1F, 0.2F, 1000.0F}, 4, 5);
  ASSERT_EQ(markers.size(), 2U);
  EXPECT_FLOAT_EQ(markers[0].doppler_bin, 2.0F);
  EXPECT_FLOAT_EQ(markers[0].range_bin, 3.0F);
  EXPECT_EQ(markers[0].features, (std::array<float, 4>{14.0F, 3.0F, 0.3F, 0.4F}));
  EXPECT_EQ(markers[0].color_bucket, 1);
  EXPECT_EQ(markers[1].color_bucket, 0);
}

TEST(FmcwDashboardTest, ConvertsDirectionCosineMarkersBeforeHistory) {
  holoscan::radar::AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kDirectionCosine;
  const auto markers =
      viz::detail::make_raw_detection_markers({2, 3}, {14.0F, 3.0F, 0.5F, 0.25F}, 2, 4, convention);
  ASSERT_EQ(markers.size(), 1U);
  EXPECT_NEAR(markers[0].features[2], std::atan2(0.5, std::sqrt(0.6875)), 1e-6);
  EXPECT_NEAR(markers[0].features[3], std::asin(0.25), 1e-6);
}

TEST(FmcwDashboardTest, AcceptsEmptyRawDetections) {
  EXPECT_TRUE(viz::detail::make_raw_detection_markers({}, {}, 4, 5).empty());
}

TEST(FmcwDashboardTest, AcceptsTwoColumnAngleFreePoints) {
  const auto raw = viz::detail::make_raw_detection_markers(
      {2, 3}, {14.0F, 3.0F}, 2, 2, holoscan::radar::AngleConvention{}, false);
  ASSERT_EQ(raw.size(), 1U);
  EXPECT_EQ(raw[0].features, (std::array<float, 4>{14.0F, 3.0F, 0.0F, 0.0F}));

  const auto clustered =
      viz::detail::make_cluster_markers({2, 3, 4, 5}, {14.0F, 3.0F, 16.0F, 5.0F}, {0, 0}, {2}, 2, 2,
                                        holoscan::radar::AngleConvention{}, false);
  ASSERT_EQ(clustered.size(), 1U);
  EXPECT_EQ(clustered[0].features, (std::array<float, 4>{15.0F, 4.0F, 0.0F, 0.0F}));
}

TEST(FmcwDashboardTest, RejectsMismatchedRawDetectionRows) {
  EXPECT_THROW(viz::detail::make_raw_detection_markers(
                   {0, 1, 2, 3}, {10.0F, 1.0F, 0.1F, 0.2F, 11.0F, 2.0F, 0.2F, 0.3F}, 4, 4),
               std::invalid_argument);
}

TEST(FmcwDashboardTest, UsesReducedFeaturesForHistory) {
  std::vector<viz::DetectionMarker> markers(2);
  viz::detail::apply_reduced_features(
      &markers, {10.0F, 1.0F, 0.1F, 0.2F, 3.0F, 20.0F, 2.0F, 0.3F, 0.4F, 4.0F}, 5);
  EXPECT_EQ(markers[0].features, (std::array<float, 4>{10.0F, 1.0F, 0.1F, 0.2F}));
  EXPECT_EQ(markers[1].features, (std::array<float, 4>{20.0F, 2.0F, 0.3F, 0.4F}));
}

TEST(FmcwDashboardTest, UsesTwoColumnReducedFeaturesWithoutAngles) {
  std::vector<viz::DetectionMarker> markers(2);
  viz::detail::apply_reduced_features(&markers, {10.0F, 1.0F, 20.0F, 2.0F}, 2,
                                      holoscan::radar::AngleConvention{}, false);
  EXPECT_EQ(markers[0].features, (std::array<float, 4>{10.0F, 1.0F, 0.0F, 0.0F}));
  EXPECT_EQ(markers[1].features, (std::array<float, 4>{20.0F, 2.0F, 0.0F, 0.0F}));
}

TEST(FmcwDashboardTest, BoundsHistoryAndMapsPanelCoordinates) {
  std::deque<viz::HistorySample> history;
  const viz::DetectionMarker marker{.doppler_bin = 7.0F,
                                    .range_bin = 11.0F,
                                    .features = {10.0F, 0.0F, 0.25F, -0.25F},
                                    .color_bucket = 3};
  for (std::int64_t frame = 0; frame < 5; ++frame) {
    viz::detail::append_history(&history, {marker}, frame, 3);
  }
  ASSERT_EQ(history.size(), 3U);
  EXPECT_EQ(history.front().frame, 2);
  EXPECT_FLOAT_EQ(history.front().doppler_bin, 7.0F);
  EXPECT_FLOAT_EQ(history.front().range_bin, 11.0F);
  const auto position = viz::detail::history_plot_position(history.back(), 4, 3, 2, {-0.5F, 0.5F});
  EXPECT_FLOAT_EQ(position[0], 1.0F);
  EXPECT_FLOAT_EQ(position[1], 0.75F);
}

TEST(FmcwDashboardTest, ExpiresDetectionAfterConfiguredFrames) {
  std::deque<viz::HistorySample> history;
  const viz::DetectionMarker marker{
      .doppler_bin = 7.0F,
      .range_bin = 11.0F,
      .features = {10.0F, 0.0F, 0.25F, -0.25F},
  };
  viz::detail::append_history(&history, {marker}, 0, 3);
  viz::detail::append_history(&history, {}, 1, 3);
  viz::detail::append_history(&history, {}, 2, 3);
  ASSERT_EQ(history.size(), 1U);
  viz::detail::append_history(&history, {}, 3, 3);
  EXPECT_TRUE(history.empty());
}

TEST(FmcwDashboardTest, DerivesMovingTargetTruthFeatures) {
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

TEST(FmcwDashboardTest, TruthVisibilityTracksTargetMotion) {
  const viz::TruthTarget target{
      .position = {1.0, 0.0, 0.0},
      .velocity = {-2.0, 0.0, 0.0},
  };
  radar::sim::SceneVisibility visibility;
  EXPECT_TRUE(viz::detail::truth_target_is_visible_at_time(target, 0.0, visibility));
  EXPECT_FALSE(viz::detail::truth_target_is_visible_at_time(target, 1.0, visibility));

  visibility.model = radar::sim::SceneVisibilityModel::kAll;
  EXPECT_TRUE(viz::detail::truth_target_is_visible_at_time(target, 1.0, visibility));
}

}  // namespace
