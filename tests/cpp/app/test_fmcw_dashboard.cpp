// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/app/angle_presentation.hpp"
#include "radar_app/viz/fmcw_dashboard_op.hpp"
#include "radar_app/viz/dashboard_text.hpp"
#include "radar_app/viz/fmcw_holoviz_op.hpp"
#include "radar_app/viz/rgba_image_bridge_op.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace viz = holoscan::radar::app::viz;
namespace radar = holoscan::radar;

class RgbaGraphSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(text).produces_tensor(holoscan::TensorOutputSpec{
        .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 8U, 1U},
                           .rank = 2U,
                           .strides = holoscan::TensorStridePolicy::kContiguousOnly},
        .bounds = holoscan::TensorBounds{.max_rank = 2U, .max_byte_span = 208U},
    });
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
  holoscan::Output<holoscan::Tensor> text{"text"};
};

TEST(FmcwDashboardTest, ConnectsImageAndNativeHostTextToHsdk5Holoviz) {
  holoscan::Graph graph{"dashboard-image-bridge-test"};
  auto source = graph.op<RgbaGraphSourceOp>("source");
  auto bridge = graph.op<viz::RgbaImageBridgeOp>("bridge", 2, 2);
  auto frame = graph.op<viz::FmcwFrameComposerOp>("frame", 2, 2);
  auto sink = graph.op<viz::FmcwHolovizOp>("holoviz", 2, 2, "test");
  graph.partition("application").add(source).add(bridge);
  graph.partition("visualization").add(frame).add(sink);
  graph.add_flow(source->rgba, bridge->rgba);
  graph.add_flow(bridge->image, frame->image_layer_input(0));
  graph.add_flow(frame->output(), sink->image_layer_input(0));
  graph.add_flow(source->text, frame->tensor_layer_input(1));
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "rgba", .device = holoscan::DeviceId{0}});
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "bridge", .output_port = "image", .device = holoscan::DeviceId{0}});
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "frame", .output_port = "output", .device = holoscan::DeviceId{0}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  EXPECT_TRUE(plan.ok()) << plan.json();
}

TEST(FmcwDashboardTest, ComposesOnlyMatchedImageAndTextIntoOneCudaFrame) {
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  constexpr std::int64_t kWidth = 128;
  constexpr std::int64_t kHeight = 64;
  constexpr DLDataType kRgba{kDLUInt, 8U, 1U};
  std::vector<std::uint8_t> pixels(kWidth * kHeight * 4U);
  for (std::size_t i = 0; i < pixels.size(); i += 4U) {
    pixels[i] = 10U;
    pixels[i + 1U] = 20U;
    pixels[i + 2U] = 30U;
    pixels[i + 3U] = 255U;
  }
  holoscan::schema::ImageT image;
  image.width = kWidth;
  image.height = kHeight;
  image.encoding = holoscan::schema::ImageEncoding_RGBA8;
  image.data = radar::testing::to_device_tensor(pixels, {kHeight, kWidth, 4}, kRgba);
  std::vector<std::uint8_t> records;
  viz::detail::append_dashboard_text(&records, kWidth, kHeight, 8.0F, 8.0F, "123", 13.0F);
  const auto text = radar::testing::to_pageable_host_tensor(
      records, {1, static_cast<std::int64_t>(viz::detail::kDashboardTextRowBytes)}, kRgba);
  auto harness = holoscan::testing::create_operator_test<viz::FmcwFrameComposerOp>(
      static_cast<std::int32_t>(kWidth), static_cast<std::int32_t>(kHeight));
  // Unmatched samples must not render. The matched frame deliberately publishes
  // its image first, so rendering cannot rely on sideband-before-driver order.
  harness.push_frame(frame_at(std::chrono::milliseconds{1}, input("image", image)));
  harness.push_frame(frame_at(std::chrono::milliseconds{2}, input("text", *text)));
  harness.push_frame(
      frame_at(std::chrono::milliseconds{3}, input("image", image), input("text", *text)));
  harness.input_source_device("image", holoscan::DeviceId{0});
  harness.output_device("output", holoscan::DeviceId{0});
  harness.capture_snapshot<holoscan::schema::ImageT>("output", 1U, pixels.size());
  auto running = harness.run_async({.expect_stop = true});
  const bool ready = running.wait_for_output("output", 1U, std::chrono::seconds{10});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(ready) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto composed = result.snapshot<holoscan::schema::ImageT>("output");
  const auto output =
      radar::testing::to_host(composed.tensor(holoscan::schema::ImageTensorFields::data));
  ASSERT_EQ(output.shape, (std::vector<std::int64_t>{kHeight, kWidth, 4}));
  const auto* rgba = output.as<std::uint8_t>();
  const std::size_t corner = 4U * (kWidth * kHeight - 1U);
  for (std::size_t channel = 0; channel < 4U; ++channel) {
    EXPECT_EQ(rgba[corner + channel], pixels[corner + channel]);
  }
  bool found_native_text = false;
  for (std::size_t row = 8U; row < 24U; ++row) {
    for (std::size_t col = 8U; col < 40U; ++col) {
      found_native_text = found_native_text || rgba[4U * (row * kWidth + col)] > 128U;
    }
  }
  EXPECT_TRUE(found_native_text) << "SDK text must be present in the emitted GPU image";
}

struct PowerLegendCase {
  float power;
  const char* lower;
  const char* upper;
};

void PrintTo(const PowerLegendCase& value, std::ostream* stream) {
  *stream << value.power;
}

class FmcwDashboardPowerTest : public ::testing::TestWithParam<PowerLegendCase> {};

TEST_P(FmcwDashboardPowerTest, ComposesHeatmapWithMatchingPowerLegend) {
  using Op = viz::FmcwDashboardOp;
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  using radar::testing::to_device_tensor;
  constexpr DLDataType kFloat{kDLFloat, 32U, 1U};
  constexpr DLDataType kInt{kDLInt, 64U, 1U};
  viz::FmcwDashboardConfig config;
  config.show_angles = false;
  config.title = "FMCW test";
  auto harness = holoscan::testing::create_operator_test<Op>(config);
  harness.push_frame(frame_at(
      std::chrono::milliseconds{1},
      input("cfar_power",
            *to_device_tensor(std::vector<float>(4U, GetParam().power), {2, 2}, kFloat)),
      input("detection_idx", *to_device_tensor(std::vector<std::int64_t>{}, {0, 2}, kInt)),
      input("points", *to_device_tensor(std::vector<float>{}, {0, 2}, kFloat)),
      input("range_bins", *to_device_tensor(std::vector<float>{0.0F, 100.0F}, {2}, kFloat)),
      input("velocity_bins", *to_device_tensor(std::vector<float>{-10.0F, 10.0F}, {2}, kFloat))));
  harness.capture_snapshot<holoscan::Tensor>("rgba", 1U, 1280U * 720U * 4U);
  harness.capture_snapshot<holoscan::Tensor>(
      "text", 1U, viz::detail::kDashboardTextRows * viz::detail::kDashboardTextRowBytes);
  for (const auto* port :
       {"cfar_power", "detection_idx", "points", "range_bins", "velocity_bins"}) {
    harness.input_source_device(port, holoscan::DeviceId{0});
  }
  harness.output_device("rgba", holoscan::DeviceId{0});
  auto running = harness.run_async({.expect_stop = true});
  const bool image_ready = running.wait_for_output("rgba", 1U, std::chrono::seconds{5});
  const bool text_ready = running.wait_for_output("text", 1U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(image_ready && text_ready) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto image = radar::testing::to_host(result.snapshot<holoscan::Tensor>("rgba"));
  const auto* pixels = image.as<std::uint8_t>();
  EXPECT_EQ(image.shape, (std::vector<std::int64_t>{720, 1280, 4}));
  // Equal positive power reaches the palette maximum throughout the RD interior.
  const std::size_t center = 4U * (300U * 1280U + 300U);
  EXPECT_EQ(pixels[center], 253U);
  EXPECT_EQ(pixels[center + 1U], 231U);
  EXPECT_EQ(pixels[center + 2U], 37U);
  for (std::size_t i = 3U; i < image.data.size(); i += 4U) {
    ASSERT_EQ(pixels[i], 255U) << "transparent canvas pixel " << i / 4U;
  }
  const auto text = radar::testing::to_host(result.snapshot<holoscan::Tensor>("text"));
  ASSERT_EQ(text.shape.size(), 2U);
  EXPECT_EQ(text.shape[1], viz::detail::kDashboardTextRowBytes);
  EXPECT_GT(text.shape[0], 10);
  EXPECT_STREQ(reinterpret_cast<const char*>(text.data.data() +
                                             holoscan::visualization::kTextRunHeaderBytes),
               "FMCW test");
  std::vector<std::string> labels;
  for (std::size_t offset = 0; offset < text.data.size();
       offset += viz::detail::kDashboardTextRowBytes) {
    labels.emplace_back(reinterpret_cast<const char*>(
        text.data.data() + offset + holoscan::visualization::kTextRunHeaderBytes));
  }
  EXPECT_NE(std::find(labels.begin(), labels.end(), GetParam().lower), labels.end());
  EXPECT_NE(std::find(labels.begin(), labels.end(), GetParam().upper), labels.end());
}

// A 100x power change moves both legend endpoints by 20 dB while retaining
// the same 60 dB span and peak color in the CUDA image.
INSTANTIATE_TEST_SUITE_P(PowerLevels, FmcwDashboardPowerTest,
                         ::testing::Values(PowerLegendCase{1.0F, "-60.0", "0.0 dB"},
                                           PowerLegendCase{100.0F, "-40.0", "20.0 dB"},
                                           PowerLegendCase{0.01F, "-80.0", "-20.0 dB"}));

class FmcwDashboardBackgroundTest : public ::testing::TestWithParam<int> {};

TEST_P(FmcwDashboardBackgroundTest, MasksDetectionCrossesAndProjectsLeadingPlanes) {
  using Op = viz::FmcwDashboardOp;
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  using radar::testing::to_device_tensor;
  constexpr DLDataType kFloat{kDLFloat, 32U, 1U};
  constexpr DLDataType kInt{kDLInt, 64U, 1U};
  viz::FmcwDashboardConfig config;
  config.show_angles = false;
  config.title = "FMCW test";
  config.noise_sample_limit = 256;
  config.noise_doppler_half_width = 1;
  config.noise_range_half_width = 1;
  const int planes = GetParam();
  std::vector<float> powers(planes * 256, 100.0F);
  for (int plane = 0; plane < planes; ++plane) {
    for (int d = 0; d < 16; ++d) {
      for (int r = 0; r < 16; ++r) {
        powers[plane * 256 + d * 16 + r] = d == 4 || r == 5 ? 1.0e12F : 100.0F * (plane + 1);
      }
    }
  }
  std::vector<float> bins(16);
  for (int i = 0; i < 16; ++i)
    bins[i] = static_cast<float>(i);
  auto harness = holoscan::testing::create_operator_test<Op>(config);
  harness.push_frame(frame_at(
      std::chrono::milliseconds{1},
      input("cfar_power", *to_device_tensor(powers, {planes, 16, 16}, kFloat)),
      input("detection_idx", *to_device_tensor(std::vector<std::int64_t>{0, 4, 5}, {1, 3}, kInt)),
      input("points", *to_device_tensor(std::vector<float>{5.0F, 4.0F}, {1, 2}, kFloat)),
      input("range_bins", *to_device_tensor(bins, {16}, kFloat)),
      input("velocity_bins", *to_device_tensor(bins, {16}, kFloat))));
  harness.capture_snapshot<holoscan::Tensor>("rgba", 1U, 1280U * 720U * 4U);
  harness.capture_snapshot<holoscan::Tensor>(
      "text", 1U, viz::detail::kDashboardTextRows * viz::detail::kDashboardTextRowBytes);
  for (const auto* port :
       {"cfar_power", "detection_idx", "points", "range_bins", "velocity_bins"}) {
    harness.input_source_device(port, holoscan::DeviceId{0});
  }
  harness.output_device("rgba", holoscan::DeviceId{0});
  auto running = harness.run_async({.expect_stop = true});
  const bool image_ready = running.wait_for_output("rgba", 1U, std::chrono::seconds{5});
  const bool text_ready = running.wait_for_output("text", 1U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(image_ready && text_ready) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto text = radar::testing::to_host(result.snapshot<holoscan::Tensor>("text"));
  ASSERT_EQ(text.shape.size(), 2U);
  EXPECT_EQ(text.shape[1], viz::detail::kDashboardTextRowBytes);
  EXPECT_GT(text.shape[0], 10);
  EXPECT_STREQ(reinterpret_cast<const char*>(text.data.data() +
                                             holoscan::visualization::kTextRunHeaderBytes),
               "FMCW test");
  std::vector<std::string> labels;
  for (std::size_t offset = 0; offset < text.data.size();
       offset += viz::detail::kDashboardTextRowBytes) {
    labels.emplace_back(reinterpret_cast<const char*>(
        text.data.data() + offset + holoscan::visualization::kTextRunHeaderBytes));
  }
  EXPECT_NE(std::find(labels.begin(), labels.end(), planes == 1 ? "20.0" : "23.0"), labels.end());
  EXPECT_NE(std::find(labels.begin(), labels.end(), planes == 1 ? "80.0 dB" : "83.0 dB"),
            labels.end());
}

INSTANTIATE_TEST_SUITE_P(Projection, FmcwDashboardBackgroundTest, ::testing::Values(1, 2));

TEST(FmcwDashboardBackgroundStatsTest, EstimatesDbMomentsAndRejectsOutliers) {
  std::vector<float> samples(256);
  for (std::size_t i = 0; i < samples.size(); ++i)
    samples[i] = i % 2 ? 19.0F : 21.0F;
  samples.push_back(1000.0F);
  samples.push_back(std::numeric_limits<float>::quiet_NaN());
  const auto stats = viz::detail::background_statistics(samples);
  ASSERT_TRUE(stats);
  EXPECT_NEAR(stats->mean_db, 20.0F, 1e-5F);
  EXPECT_NEAR(stats->stddev_db, 1.0F, 1e-5F);
  EXPECT_EQ(stats->count, 256U);
  EXPECT_FALSE(viz::detail::background_statistics(std::vector<float>(63, 20.0F)));
  EXPECT_FALSE(viz::detail::background_statistics(
      std::vector<float>(2048, std::numeric_limits<float>::quiet_NaN())));
}

TEST(FmcwDashboardBackgroundStatsTest, SmoothsHoldsAndExpiresToPeakFallback) {
  viz::detail::BackgroundScaleState state;
  const viz::detail::BackgroundStatistics background{20.0F, 1.0F, 256};
  EXPECT_EQ(state.limits(std::nullopt, 100.0F, 60.0F, 2.0F, 0),
            (std::array<float, 2>{-40.0F, 20.0F}));
  EXPECT_EQ(state.limits(background, 1e12F, 60.0F, 2.0F, 0), (std::array<float, 2>{22.0F, 82.0F}));
  const auto moved = state.limits(viz::detail::BackgroundStatistics{30.0F, 1.0F, 256}, 1e12F, 60.0F,
                                  2.0F, 250000000);
  EXPECT_NEAR(moved[0], 22.0F + 10.0F * (1.0F - std::exp(-1.0F)), 1e-5F);
  EXPECT_EQ(state.limits(std::nullopt, 1e12F, 60.0F, 2.0F, 750000000), moved);
  EXPECT_EQ(state.limits(std::nullopt, 100.0F, 60.0F, 2.0F, 1250000001),
            (std::array<float, 2>{-40.0F, 20.0F}));
  EXPECT_EQ(state.limits(std::nullopt, 0.0F, 60.0F, 2.0F, 1500000000),
            (std::array<float, 2>{-60.0F, 0.0F}));
}

TEST(FmcwDashboardTextTest, EncodesNativeSdkRowsWithOwnedTextAndPixelPositions) {
  std::vector<std::uint8_t> records;
  std::string label = "Range (m)";
  viz::detail::append_dashboard_text(&records, 1280, 720, 640.0F, 180.0F, label, 13.0F);
  label = "changed";
  viz::detail::append_dashboard_text(&records, 1280, 720, 20.0F, 43.0F, "Frame 2", 13.0F);
  ASSERT_EQ(records.size(), 2U * viz::detail::kDashboardTextRowBytes);
  holoscan::visualization::TextRunHeader header;
  std::memcpy(&header, records.data(), sizeof(header));
  EXPECT_FLOAT_EQ(header.x, 640.0F);
  EXPECT_FLOAT_EQ(header.y, 180.0F);
  EXPECT_FLOAT_EQ(header.size_px, 13.0F);
  EXPECT_EQ(header.color >> 24U, 255U);
  const auto* text = reinterpret_cast<const char*>(records.data() + sizeof(header));
  EXPECT_STREQ(text, "Range (m)");
  for (std::size_t i = sizeof(header) + 9U; i < viz::detail::kDashboardTextRowBytes; ++i) {
    EXPECT_EQ(records[i], 0U);
  }
}

TEST(FmcwDashboardTextTest, FitsLongWorkflowNamesAndEnforcesThePerFrameBudget) {
  std::vector<std::uint8_t> records;
  viz::detail::append_dashboard_text(&records, 640, 360, 20.0F, 15.0F, std::string(400U, 'A'),
                                     20.0F);
  const auto* text =
      reinterpret_cast<const char*>(records.data() + holoscan::visualization::kTextRunHeaderBytes);
  const std::string displayed(text);
  EXPECT_LT(displayed.size(), viz::detail::kDashboardTextBytes);
  EXPECT_TRUE(displayed.ends_with("..."));
  EXPECT_LE(20.0F + displayed.size() * 20.0F * 7.0F / 13.0F, 628.0F);
  for (std::size_t i = 1U; i < viz::detail::kDashboardTextRows; ++i) {
    viz::detail::append_dashboard_text(&records, 640, 360, 20.0F, 15.0F, "x", 13.0F);
  }
  EXPECT_THROW(
      viz::detail::append_dashboard_text(&records, 640, 360, 20.0F, 15.0F, "overflow", 13.0F),
      std::invalid_argument);
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
  EXPECT_EQ(rgba[0], 68U);
  EXPECT_EQ(rgba[1], 1U);
  EXPECT_EQ(rgba[2], 84U);
  EXPECT_EQ(rgba[3], 255U);
  EXPECT_EQ(rgba[4], 253U);
  EXPECT_EQ(rgba[5], 231U);
  EXPECT_EQ(rgba[6], 37U);
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

TEST(FmcwRdHeatmapTest, ReversesMarkersWithDescendingPhysicalVelocityBins) {
  const auto normal = viz::detail::rd_plot_position(1.0F, 2.0F, 8, 4);
  const auto reversed = viz::detail::rd_plot_position(1.0F, 2.0F, 8, 4, true);
  EXPECT_FLOAT_EQ(reversed[0], 1.0F - normal[0]);
  EXPECT_FLOAT_EQ(reversed[1], normal[1]);
}

TEST(FmcwDashboardTest, UsesUpwardSemicircleForAtMost180DegreeCoverage) {
  constexpr float kHalfPi = 1.57079632679489661923F;
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-0.3F, 0.3F}));
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-kHalfPi, kHalfPi}));
  EXPECT_TRUE(viz::detail::use_azimuth_sector({-2.0F, 0.0F}));
  EXPECT_FALSE(viz::detail::use_azimuth_sector({-2.0F, 2.0F}));
}

TEST(FmcwDashboardTest, NarrowSectorFillsPanelWidthAndHeight) {
  const std::array<float, 2> bounds{-0.3F, 0.3F};
  const auto left = viz::detail::sector_plot_position(1.0F, bounds[0], bounds);
  const auto right = viz::detail::sector_plot_position(1.0F, bounds[1], bounds);
  const auto far = viz::detail::sector_plot_position(1.0F, 0.0F, bounds);
  const auto near = viz::detail::sector_plot_position(0.0F, 0.0F, bounds);
  ASSERT_TRUE(left && right && far && near);
  EXPECT_NEAR((*left)[0], 0.0F, 1.0e-6F);
  EXPECT_NEAR((*right)[0], 1.0F, 1.0e-6F);
  EXPECT_NEAR((*left)[1], 1.0F - std::cos(0.3F), 1.0e-6F);
  EXPECT_EQ(*far, (std::array<float, 2>{0.5F, 0.0F}));
  EXPECT_EQ(*near, (std::array<float, 2>{0.5F, 1.0F}));
  const auto halfway = viz::detail::sector_plot_position(0.5F, 0.0F, bounds);
  ASSERT_TRUE(halfway);
  EXPECT_EQ(*halfway, (std::array<float, 2>{0.5F, 0.5F}));
}

TEST(FmcwDashboardTest, FittedSectorHandlesOffCenterAndWrappedAzimuths) {
  const std::array<float, 2> bounds{2.8F, 3.4F};
  constexpr float kTwoPi = 6.28318530717958647692F;
  const auto right = viz::detail::sector_plot_position(1.0F, bounds[1] - kTwoPi, bounds);
  ASSERT_TRUE(right);
  EXPECT_NEAR((*right)[0], 1.0F, 1.0e-6F);
  const auto center = viz::detail::sector_plot_position(0.5F, 3.1F, bounds);
  ASSERT_TRUE(center);
  EXPECT_NEAR((*center)[0], 0.5F, 1.0e-6F);
  EXPECT_NEAR((*center)[1], 0.5F, 1.0e-6F);
}

TEST(FmcwDashboardTest, FittedSectorContainsAllRangesForNarrowAndWideScans) {
  constexpr float kHalfPi = 1.57079632679489661923F;
  for (const auto& bounds :
       {std::array<float, 2>{-1.0e-7F, 1.0e-7F}, std::array<float, 2>{-kHalfPi, kHalfPi},
        std::array<float, 2>{-2.0F, 0.0F}}) {
    for (int step = 0; step <= 100; ++step) {
      const float azimuth = bounds[0] + (bounds[1] - bounds[0]) * step / 100.0F;
      for (float range : {0.0F, 0.25F, 0.5F, 1.0F}) {
        const auto position = viz::detail::sector_plot_position(range, azimuth, bounds);
        ASSERT_TRUE(position);
        for (float coordinate : *position) {
          EXPECT_GE(coordinate, 0.0F);
          EXPECT_LE(coordinate, 1.0F);
        }
      }
    }
  }
}

TEST(FmcwDashboardTest, FittedSectorRejectsOutOfViewAndInvalidSamples) {
  const std::array<float, 2> bounds{-0.3F, 0.3F};
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, 0.4F, bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, -0.4F, bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(-0.1F, 0.0F, bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(1.1F, 0.0F, bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(std::nanf(""), 0.0F, bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, std::nanf(""), bounds));
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, 0.0F, {0.0F, 0.0F}));
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, 0.0F, {0.3F, -0.3F}));
  EXPECT_FALSE(viz::detail::sector_plot_position(0.5F, 0.0F, {-3.14F, 3.14F}));
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
