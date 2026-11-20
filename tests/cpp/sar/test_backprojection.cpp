// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::BackprojectionOp;
using holoscan::radar::BackprojectionParams;
using holoscan::radar::CoordinateFrame;
using holoscan::radar::CoordinateFrameSpec;
using holoscan::radar::CoordinateFrameType;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

TEST(BackprojectionOp, PointTargetFormsAtSceneCentreAndResets) {
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto hud = std::make_shared<holoscan::radar::FormationHud>();
  hud->enable_progress(1, 1);
  const auto progress = hud->progress();
  const std::array first_time{100.};
  const std::array second_time{102.4};
  progress->begin_packet(1, 0, 0, 1, 2, 2.4, 0.2, true, first_time);
  progress->admit(1, 1, true, true);
  progress->begin_packet(2, 0, 1, 1, 2, 2.4, 0.2, false, second_time);
  progress->admit(2, 1, true, true);
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<float>>(extents, hud);
  BackprojectionParams params;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -2.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  fixture.set_params(params);
  std::vector<std::complex<float>> impulse(64U, {0.F, 0.F});
  impulse[32] = {1.F, 0.F};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<float>());
  auto positions =
      to_device_tensor(std::vector<float>{0.F, 0.F, 100.F}, {1, 3}, real_dtype<float>());
  auto reference = to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
  auto control = to_device_tensor(std::vector<uint8_t>{1U}, {1}, DLDataType{kDLUInt, 8U, 1U});
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{1}, holoscan::testing::input("control", *control),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{2}, holoscan::testing::input("control", *control),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  for (const char* port : {"range_profiles", "positions", "reference_position", "control"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.capture_snapshot<holoscan::Tensor>("image", 2U, 5U * 5U * sizeof(std::complex<float>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 2U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.snapshots<holoscan::Tensor>("image");
  ASSERT_EQ(captures.size(), 2U);
  const auto diagnostic = hud->snapshot().bp;
  EXPECT_EQ(diagnostic.capture_time_ns, 2);
  ASSERT_TRUE(diagnostic.budget_s);
  EXPECT_NEAR(*diagnostic.budget_s, 2.4, 1e-12);
  EXPECT_EQ(diagnostic.budget_playback_rate, std::optional{0.2});
  const auto image = to_host(captures[0]);
  const auto reset_image = to_host(captures[1]);
  ASSERT_EQ(image.shape, (std::vector<int64_t>{5, 5}));
  const auto* data = image.as<std::complex<float>>();
  EXPECT_NEAR(std::abs(data[12]), 1.F, 1e-3F);
  for (int64_t i = 0; i < 25; ++i) {
    EXPECT_LE(std::abs(data[i]), std::abs(data[12]) + 1e-3F);
    EXPECT_NEAR(std::abs(reset_image.as<std::complex<float>>()[i] - data[i]), 0.F, 1e-3F)
        << "reset retained accumulation at pixel " << i;
  }
}

// Form the point target above under an explicit `<T, TGeom>` pair, returning
// the image magnitudes. Every other test in this file names the one-argument
// form, which defaults `TGeom` to `T`; the pair applications actually run is
// `<float, double>` -- samples in single, geometry in double, because ECEF
// coordinates do not survive float32 -- and nothing here constructed it, so
// the split-precision path was exercised only by the app smoke tests.
template <typename T, typename TGeom>
std::vector<double> form_point_target(holoscan::radar::BackprojectionBackend backend =
                                          holoscan::radar::kDefaultBackprojectionBackend) {
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<T, TGeom>>(extents);
  BackprojectionParams params;
  params.backend = backend;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -2.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  fixture.set_params(params);
  std::vector<std::complex<T>> impulse(64U, {T{0}, T{0}});
  impulse[32] = {T{1}, T{0}};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<T>());
  auto positions = to_device_tensor(std::vector<TGeom>{TGeom{0}, TGeom{0}, TGeom{100}}, {1, 3},
                                    real_dtype<TGeom>());
  auto reference = to_device_tensor(std::vector<TGeom>{TGeom{0}, TGeom{0}, TGeom{0}}, {1, 3},
                                    real_dtype<TGeom>());
  auto control = to_device_tensor(std::vector<uint8_t>{1U}, {1}, DLDataType{kDLUInt, 8U, 1U});
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{1}, holoscan::testing::input("control", *control),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  for (const char* port : {"range_profiles", "positions", "reference_position", "control"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.template capture_snapshot<holoscan::Tensor>("image", 1U,
                                                      5U * 5U * sizeof(std::complex<T>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  EXPECT_TRUE(observed) << result.describe_failure();
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.template snapshots<holoscan::Tensor>("image");
  EXPECT_EQ(captures.size(), 1U);
  if (captures.size() != 1U) {
    return {};
  }
  const auto image = to_host(captures[0]);
  EXPECT_EQ(image.shape, (std::vector<int64_t>{5, 5}));
  const auto* data = image.template as<std::complex<T>>();
  std::vector<double> magnitudes(25U);
  for (int64_t i = 0; i < 25; ++i) {
    magnitudes[static_cast<size_t>(i)] = static_cast<double>(std::abs(data[i]));
  }
  return magnitudes;
}

// The application pair, `<float, double>`, forms the same image as the uniform
// one on a scene-local lattice. It has to: scene-local coordinates are small
// enough that double geometry buys nothing here, so any disagreement would be
// the split-precision plumbing -- a geometry port read at the wrong width, or
// a position narrowed to T on its way into the kernel -- and not arithmetic.
// Agreement on this scene is therefore the strong claim, and the reason this
// test does not use ECEF: on ECEF the two *should* differ, which tests the
// arithmetic rather than the plumbing this operator owns.
TEST(BackprojectionOp, MixedGeometryPrecisionMatchesUniform) {
  const auto uniform = form_point_target<float, float>();
  const auto mixed = form_point_target<float, double>();
  ASSERT_EQ(uniform.size(), 25U);
  ASSERT_EQ(mixed.size(), 25U);
  EXPECT_NEAR(mixed[12], 1.0, 1e-3) << "the mixed pair did not form the point target";
  for (size_t i = 0; i < 25U; ++i) {
    EXPECT_NEAR(mixed[i], uniform[i], 1e-3) << "pixel " << i;
  }
}

TEST(BackprojectionOp, PrecisionConversionBackendsPreservePointTarget) {
  using Backend = holoscan::radar::BackprojectionBackend;
  for (const auto backend : {Backend::kNvradarFp64Reference, Backend::kNvradarCudaFp32,
                             Backend::kNvradarBlockAnchoredRefinedThroughput}) {
    const auto single = form_point_target<float, double>(backend);
    const auto dual = form_point_target<double, double>(backend);
    ASSERT_EQ(single.size(), 25U);
    ASSERT_EQ(dual.size(), 25U);
    EXPECT_NEAR(single[12], 1.0, 1e-3);
    EXPECT_NEAR(dual[12], 1.0, 1e-3);
    for (std::size_t i = 0; i < single.size(); ++i) {
      EXPECT_LE(single[i], single[12] + 1e-3);
      EXPECT_LE(dual[i], dual[12] + 1e-3);
      EXPECT_NEAR(single[i], dual[i], 1e-3) << "pixel " << i;
    }
  }
}

TEST(BackprojectionOp, ScrollRetainsRowsAndRepeatedGeometryIsNoOp) {
  using holoscan::radar::GeometryChange;
  using holoscan::radar::to_wire;
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<float>>(extents);
  BackprojectionParams params;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -2.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  params.geometry_change = GeometryChange::kScroll;
  fixture.set_params(params);
  const auto moved = SampleGeometry(SampleGeometrySpec{.origin = {-2.0, -1.0, 0.0},
                                                       .axis_u = {1.0, 0.0, 0.0},
                                                       .axis_v = {0.0, 1.0, 0.0},
                                                       .spacing = {1.0, 1.0},
                                                       .counts = {5, 5}});
  std::vector<std::complex<float>> impulse(64U, {0.F, 0.F});
  impulse[32] = {1.F, 0.F};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<float>());
  auto positions =
      to_device_tensor(std::vector<float>{0.F, 0.F, 100.F}, {1, 3}, real_dtype<float>());
  auto reference = to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
  auto control = to_device_tensor(std::vector<uint8_t>{0U}, {1}, DLDataType{kDLUInt, 8U, 1U});
  for (int tick = 0; tick < 3; ++tick) {
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{tick + 1},
        holoscan::testing::input("sample_geometry",
                                 to_wire(tick == 2 ? moved : params.sample_geometry)),
        holoscan::testing::input("control", *control),
        holoscan::testing::input("range_profiles", *profiles),
        holoscan::testing::input("positions", *positions),
        holoscan::testing::input("reference_position", *reference)));
  }
  for (const char* port : {"range_profiles", "positions", "reference_position", "control"})
    fixture.input_source_device(port, holoscan::DeviceId{0});
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.capture_snapshot<holoscan::Tensor>("image", 3U, 5U * 5U * sizeof(std::complex<float>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.snapshots<holoscan::Tensor>("image");
  ASSERT_EQ(captures.size(), 3U);
  const auto first = to_host(captures[0]);
  const auto second = to_host(captures[1]);
  const auto third = to_host(captures[2]);
  const auto* a = first.as<std::complex<float>>();
  const auto* b = second.as<std::complex<float>>();
  const auto* c = third.as<std::complex<float>>();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 5; ++col) {
      const int old_index = (row + 1) * 5 + col;
      EXPECT_NEAR(std::abs(b[old_index] - 2.F * a[old_index]), 0.F, 2e-3F);
      EXPECT_NEAR(std::abs(c[row * 5 + col] - 3.F * a[old_index]), 0.F, 3e-3F);
    }
  }
}
// `sample_points` had no coverage at all, and the Figure 2 polar aux is the
// first caller: a lattice of explicit points is the only way to express rows of
// constant slant range, which `SampleGeometry`'s two axes and uniform spacing
// cannot. This pins that the port is honoured at all, independent of any app
// wiring, and that it overrides the Params geometry for the tick.
TEST(BackprojectionOp, SamplePointsLatticeOverridesGeometry) {
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_range_bins = 64;
  extents.max_image_v = 5;
  extents.max_image_u = 5;
  auto fixture = holoscan::testing::create_operator_test<BackprojectionOp<float>>(extents);
  BackprojectionParams params;
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kSceneLocal});
  // Deliberately a different lattice from the points below, so a formed image
  // at the expected cell can only come from the points.
  params.sample_geometry = SampleGeometry(SampleGeometrySpec{.origin = {100.0, 100.0, 0.0},
                                                             .axis_u = {1.0, 0.0, 0.0},
                                                             .axis_v = {0.0, 1.0, 0.0},
                                                             .spacing = {1.0, 1.0},
                                                             .counts = {5, 5}});
  params.range_bin_spacing = 0.25;
  params.range_bin_origin = -8.0;
  params.reference_frequency = 9.6e9;
  fixture.set_params(params);
  std::vector<std::complex<float>> impulse(64U, {0.F, 0.F});
  impulse[32] = {1.F, 0.F};
  auto profiles = to_device_tensor(impulse, {1, 64}, complex_dtype<float>());
  auto positions =
      to_device_tensor(std::vector<float>{0.F, 0.F, 100.F}, {1, 3}, real_dtype<float>());
  auto reference = to_device_tensor(std::vector<float>{0.F, 0.F, 0.F}, {1, 3}, real_dtype<float>());
  auto control = to_device_tensor(std::vector<uint8_t>{1U}, {1}, DLDataType{kDLUInt, 8U, 1U});

  // The same 5x5 unit grid about the scene centre that the other tests build
  // from SampleGeometry, but handed over as explicit points.
  std::vector<float> points;
  points.reserve(5U * 5U * 3U);
  for (int64_t v = 0; v < 5; ++v) {
    for (int64_t u = 0; u < 5; ++u) {
      points.push_back(static_cast<float>(u) - 2.F);
      points.push_back(static_cast<float>(v) - 2.F);
      points.push_back(0.F);
    }
  }
  auto lattice = to_device_tensor(points, {5, 5, 3}, real_dtype<float>());

  // Order matters: the harness emits a frame's inputs in argument order and the
  // operator activates as soon as the OnAligned set is complete, so a `[live]`
  // override listed after `reference_position` misses the tick it was meant for.

  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{1}, holoscan::testing::input("control", *control),
      holoscan::testing::input("sample_points", *lattice),
      holoscan::testing::input("range_profiles", *profiles),
      holoscan::testing::input("positions", *positions),
      holoscan::testing::input("reference_position", *reference)));
  for (const char* port :
       {"range_profiles", "positions", "reference_position", "control", "sample_points"}) {
    fixture.input_source_device(port, holoscan::DeviceId{0});
  }
  fixture.output_device("image", holoscan::DeviceId{0});
  fixture.capture_snapshot<holoscan::Tensor>("image", 1U, 5U * 5U * sizeof(std::complex<float>));
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "image", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto captures = result.snapshots<holoscan::Tensor>("image");
  ASSERT_EQ(captures.size(), 1U);
  const auto image = to_host(captures[0]);
  ASSERT_EQ(image.shape, (std::vector<int64_t>{5, 5}));
  const auto* data = image.as<std::complex<float>>();
  double best = 0.0;
  int64_t best_i = 0;
  for (int64_t i = 0; i < 25; ++i) {
    if (std::abs(data[i]) > best) {
      best = std::abs(data[i]);
      best_i = i;
    }
  }
  EXPECT_GT(best, 0.0) << "sample_points lattice produced no image";
  EXPECT_EQ(best_i, 12) << "peak should land on the centre of the supplied lattice";
}

}  // namespace
