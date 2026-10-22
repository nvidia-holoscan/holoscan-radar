// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace radar_testing = holoscan::radar::testing;

using holoscan::radar::DopplerNotchOp;
using holoscan::radar::DopplerNotchParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 5;
constexpr int64_t kRange = 2;

MaxTensorExtents extents_for_rank(std::size_t rank, int64_t doppler = kDoppler,
                                  int64_t range = kRange) {
  MaxTensorExtents extents;
  extents.max_doppler_bins = doppler;
  extents.max_range_bins = range;
  if (rank >= 3U)
    extents.max_rx = 2;
  if (rank == 4U)
    extents.max_tx = 2;
  return extents;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_axis(const std::vector<T>& values = std::vector<T>{
                                                T{-2}, T{-1}, T{0}, T{1}, T{2}}) {
  return to_device_tensor<T>(values, {static_cast<int64_t>(values.size())}, real_dtype<T>());
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_maps(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (const int64_t extent : shape)
    count *= extent;
  std::vector<std::complex<T>> values(static_cast<std::size_t>(count),
                                      std::complex<T>{T{2}, T{-1}});
  return to_device_tensor<std::complex<T>>(values, shape, complex_dtype<T>());
}

template <typename T>
holoscan::testing::TestResult run_notch(const std::vector<int64_t>& shape,
                                        DopplerNotchParams params,
                                        std::shared_ptr<holoscan::Tensor> axis = make_axis<T>(),
                                        std::optional<double> center = std::nullopt) {
  using Op = DopplerNotchOp<T>;
  auto maps = make_maps<T>(shape);
  auto fixture = holoscan::testing::create_operator_test<Op>(
      extents_for_rank(shape.size(), shape[shape.size() - 2U], shape.back()));
  fixture.set_params(std::move(params));
  auto frame =
      center.has_value()
          ? holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                        holoscan::testing::input("velocity_bins", *axis),
                                        holoscan::testing::input("rd_maps", *maps),
                                        holoscan::testing::input("center_velocity", *center))
          : holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                        holoscan::testing::input("velocity_bins", *axis),
                                        holoscan::testing::input("rd_maps", *maps));
  fixture.push_frame(frame);
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 1U, maps->nbytes());
  fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "rd_maps_out", 1U);
  (void)observed;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

std::vector<HostTensor> host_frames(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>("rd_maps_out")) {
    frames.push_back(radar_testing::to_host(snapshot));
  }
  return frames;
}

template <typename T>
void expect_gain(const HostTensor& output, int64_t doppler, T expected) {
  const auto* values = output.as<std::complex<T>>();
  for (int64_t index = doppler * kRange; index < (doppler + 1) * kRange; ++index) {
    EXPECT_NEAR(values[index].real(), T{2} * expected, T{1e-5});
    EXPECT_NEAR(values[index].imag(), T{-1} * expected, T{1e-5});
  }
}

TEST(DopplerNotchOp, HardNullsNearestCenterBin) {
  const auto result = run_notch<float>({kDoppler, kRange}, DopplerNotchParams{});
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 1.0F);
  expect_gain(frames[0], 1, 1.0F);
  expect_gain(frames[0], 2, 0.0F);
  expect_gain(frames[0], 3, 1.0F);
  expect_gain(frames[0], 4, 1.0F);
}

TEST(DopplerNotchOp, RaisedCosineMatchesPhysicalAxis) {
  DopplerNotchParams params;
  params.transition_width = 2.0;
  const auto result = run_notch<double>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 1.0);
  expect_gain(frames[0], 1, 0.5);
  expect_gain(frames[0], 2, 0.0);
  expect_gain(frames[0], 3, 0.5);
  expect_gain(frames[0], 4, 1.0);
}

TEST(DopplerNotchOp, CyclicDistanceWrapsAtAxisEdge) {
  DopplerNotchParams params;
  params.center_velocity = 2.0;
  params.transition_width = 2.0;
  const auto result = run_notch<float>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.ok()) << result.describe_failure();

  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 1U);
  expect_gain(frames[0], 0, 0.5F);
  expect_gain(frames[0], 4, 0.0F);
}

TEST(DopplerNotchOp, NamedRanksPreserveShapeAndChannelPhase) {
  for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{
           {kDoppler, kRange}, {2, kDoppler, kRange}, {2, 2, kDoppler, kRange}}) {
    DopplerNotchParams params;
    params.transition_width = 2.0;
    const auto result = run_notch<double>(shape, params);
    ASSERT_TRUE(result.ok()) << result.describe_failure();

    const auto frames = host_frames(result);
    ASSERT_EQ(frames.size(), 1U);
    EXPECT_EQ(frames[0].shape, shape);
    const auto* values = frames[0].as<std::complex<double>>();
    int64_t count = 1;
    for (const int64_t extent : shape)
      count *= extent;
    for (int64_t index = 0; index < count; ++index) {
      if (values[index].real() != 0.0) {
        EXPECT_NEAR(values[index].imag() / values[index].real(), -0.5, 1e-12);
      }
    }
  }
}

TEST(DopplerNotchOp, LiveCenterOverrideAppliesForOneTick) {
  using Op = DopplerNotchOp<float>;
  auto axis0 = make_axis<float>();
  auto axis1 = make_axis<float>();
  auto maps0 = make_maps<float>({kDoppler, kRange});
  auto maps1 = make_maps<float>({kDoppler, kRange});
  auto fixture = holoscan::testing::create_operator_test<Op>(extents_for_rank(2U));
  fixture.set_params(DopplerNotchParams{});
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{100},
                                                 holoscan::testing::input("center_velocity", 1.0),
                                                 holoscan::testing::input("velocity_bins", *axis0),
                                                 holoscan::testing::input("rd_maps", *maps0)));
  fixture.push_frame(holoscan::testing::frame_at(std::chrono::milliseconds{200},
                                                 holoscan::testing::input("velocity_bins", *axis1),
                                                 holoscan::testing::input("rd_maps", *maps1)));
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 2U, maps0->nbytes());
  fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running =
      fixture.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
  const auto released0 = running.release_frame(completion.remaining());
  const bool observed0 = completion.wait_for_output(running, "rd_maps_out", 1U);
  const auto released1 = running.release_frame(completion.remaining());
  const bool observed1 = completion.wait_for_output(running, "rd_maps_out", 2U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(released0) << released0.error().message;
  ASSERT_TRUE(released1) << released1.error().message;
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(observed0);
  ASSERT_TRUE(observed1);
  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), 2U);
  expect_gain(frames[0], 3, 0.0F);
  expect_gain(frames[1], 2, 0.0F);
}

TEST(DopplerNotchOp, RejectsVelocityLengthMismatch) {
  auto short_axis = to_device_tensor<float>({-1.0F, 0.0F, 1.0F}, {3}, real_dtype<float>());
  const auto result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, std::move(short_axis));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DopplerNotchOp, RejectsPrecisionMismatch) {
  const auto result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, make_axis<double>());
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DopplerNotchOp, RejectsNegativeWidths) {
  DopplerNotchParams params;
  params.stop_half_width = -1.0;
  const auto result = run_notch<float>({kDoppler, kRange}, params);
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("rd_maps_out").empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

template <typename T>
class DopplerNotchAxisOrder : public ::testing::Test {};
using NotchPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DopplerNotchAxisOrder, NotchPrecisions);

TYPED_TEST(DopplerNotchAxisOrder, PhysicalGainsAgreeForBothOrdersAndCyclicBoundaries) {
  using T = TypeParam;
  for (const bool descending : {false, true}) {
    std::vector<T> axis{T{-2}, T{-1}, T{0}, T{1}, T{2}};
    if (descending)
      std::reverse(axis.begin(), axis.end());
    for (const bool cyclic : {false, true}) {
      for (const double center : {0.0, 2.0, 7.0, -7.0}) {
        for (const double transition : {0.0, 2.0}) {
          SCOPED_TRACE(::testing::Message()
                       << "descending=" << descending << " cyclic=" << cyclic
                       << " center=" << center << " transition=" << transition);
          DopplerNotchParams params;
          params.center_velocity = center;
          params.transition_width = transition;
          params.cyclic_velocity_axis = cyclic;
          const auto result = run_notch<T>({kDoppler, kRange}, params, make_axis<T>(axis));
          ASSERT_TRUE(result.ok()) << result.describe_failure();
          const auto frames = host_frames(result);
          ASSERT_EQ(frames.size(), 1U);
          // Independent physical-distance reference: scan actual bin coordinates, not index math.
          std::vector<double> distances;
          for (const T velocity : axis) {
            double distance = std::abs(static_cast<double>(velocity) - center);
            if (cyclic) {
              distance = std::fmod(distance, 5.0);
              distance = std::min(distance, 5.0 - distance);
            }
            distances.push_back(distance);
          }
          const auto nearest = std::min_element(distances.begin(), distances.end());
          for (int64_t bin = 0; bin < kDoppler; ++bin) {
            const double distance = distances[static_cast<std::size_t>(bin)];
            double gain = 1.0;
            if (distance == *nearest) {
              gain = 0.0;
            } else if (transition > 0.0 && distance < transition) {
              gain = 0.5 * (1.0 - std::cos(std::acos(-1.0) * distance / transition));
            }
            expect_gain(frames[0], bin, static_cast<T>(gain));
          }
        }
      }
    }
  }
}

TYPED_TEST(DopplerNotchAxisOrder, NonCyclicNonUniformAxisNullsActualNearestBin) {
  using T = TypeParam;
  for (const bool descending : {false, true}) {
    std::vector<T> axis{T{-8}, T{-7}, T{-6}, T{-5}, T{8}};
    if (descending)
      std::reverse(axis.begin(), axis.end());
    for (const double center : {-20.0, -6.8, -5.2, 0.5, 20.0}) {
      for (const double transition : {0.0, 2.0}) {
        SCOPED_TRACE(::testing::Message() << "descending=" << descending << " center=" << center
                                          << " transition=" << transition);
        DopplerNotchParams params;
        params.center_velocity = center;
        params.transition_width = transition;
        params.cyclic_velocity_axis = false;
        const auto result = run_notch<T>({kDoppler, kRange}, params, make_axis<T>(axis));
        ASSERT_TRUE(result.ok()) << result.describe_failure();
        const auto frames = host_frames(result);
        ASSERT_EQ(frames.size(), 1U);
        // Exhaustive physical-distance reference, independent of the kernel's index search.
        std::vector<double> distances;
        for (const T velocity : axis)
          distances.push_back(std::abs(static_cast<double>(velocity) - center));
        const auto nearest = std::min_element(distances.begin(), distances.end());
        for (int64_t bin = 0; bin < kDoppler; ++bin) {
          const double distance = distances[static_cast<std::size_t>(bin)];
          double gain = 1.0;
          if (distance == *nearest) {
            gain = 0.0;
          } else if (transition > 0.0 && distance < transition) {
            gain = 0.5 * (1.0 - std::cos(std::acos(-1.0) * distance / transition));
          }
          expect_gain(frames[0], bin, static_cast<T>(gain));
        }
      }
    }
  }
}

TYPED_TEST(DopplerNotchAxisOrder, DescendingDdmAxisPreservesPositiveEdgePassband) {
  using T = TypeParam;
  constexpr int64_t kDdmDoppler = 60;
  constexpr int64_t kDdmRange = 257;
  // Linear DDM: 77 GHz, PRI 25 us, 128 coded chirps, two TX, four guard bins.
  constexpr double kSpacing = 299792458.0 / (2.0 * 77e9 * 25e-6 * 128.0);
  std::vector<T> axis;
  for (int64_t bin = 0; bin < kDdmDoppler; ++bin)
    axis.push_back(static_cast<T>((30 - bin) * kSpacing));
  for (const bool cyclic : {false, true}) {
    for (const double stop : {0.0, 0.61}) {
      DopplerNotchParams params;
      params.stop_half_width = stop;
      params.transition_width = 1.22;
      params.cyclic_velocity_axis = cyclic;
      const auto result = run_notch<T>({2, 2, kDdmDoppler, kDdmRange}, params, make_axis<T>(axis));
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto frames = host_frames(result);
      ASSERT_EQ(frames.size(), 1U);
      const auto* output = frames[0].template as<std::complex<T>>();
      for (int64_t channel = 0; channel < 4; ++channel) {
        for (int64_t bin = 0; bin < kDdmDoppler; ++bin) {
          const double distance =
              std::abs(static_cast<double>(axis[static_cast<std::size_t>(bin)]));
          double gain = 1.0;
          if (distance <= stop) {
            gain = 0.0;
          } else if (distance < stop + params.transition_width) {
            gain = 0.5 *
                   (1.0 - std::cos(std::acos(-1.0) * (distance - stop) / params.transition_width));
          }
          for (int64_t range = 0; range < kDdmRange; ++range) {
            const auto value = output[(channel * kDdmDoppler + bin) * kDdmRange + range];
            EXPECT_NEAR(value.real(), 2.0 * gain, 1e-5);
            EXPECT_NEAR(value.imag(), -gain, 1e-5);
          }
        }
      }
    }
  }
}

template <typename T>
class DopplerNotchBoundary : public ::testing::Test {};
using DopplerNotchPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DopplerNotchBoundary, DopplerNotchPrecisions);

TYPED_TEST(DopplerNotchBoundary, EveryCellMatchesIndependentGainAcrossRowBoundaries) {
  using T = TypeParam;
  using Op = DopplerNotchOp<T>;
  // The last shape has65541 rows, crossing the prospective capped65535-row grid. The stride changes
  // Doppler bin modulo7.
  const std::vector<std::vector<int64_t>> shapes{{5, 1}, {2, 3, 7, 257}, {3, 3121, 7, 33}};
  for (const auto& shape : shapes) {
    const int64_t range = shape.back();
    const int64_t doppler = shape[shape.size() - 2U];
    int64_t channels = 1;
    for (size_t index = 0; index + 2U < shape.size(); ++index)
      channels *= shape[index];
    for (const bool hard_noncyclic : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "channels=" << channels << " range=" << range
                                        << " hard_noncyclic=" << hard_noncyclic);
      std::vector<std::complex<T>> markers(static_cast<size_t>(channels * doppler * range));
      for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t d = 0; d < doppler; ++d) {
          for (int64_t r = 0; r < range; ++r) {
            const T value = static_cast<T>((channel % 97) * 4 + 8 * d + r + 1);
            markers[static_cast<size_t>((channel * doppler + d) * range + r)] = {value,
                                                                                 -value / T{8}};
          }
        }
      }
      const auto input = to_device_tensor(markers, shape, complex_dtype<T>());
      std::vector<T> velocities(static_cast<size_t>(doppler));
      for (int64_t d = 0; d < doppler; ++d)
        velocities[static_cast<size_t>(d)] = static_cast<T>(d - doppler / 2);
      const auto axis = to_device_tensor(velocities, {doppler}, real_dtype<T>());
      MaxTensorExtents extents;
      extents.max_doppler_bins = doppler;
      extents.max_range_bins = range;
      if (shape.size() >= 3U)
        extents.max_rx = shape[shape.size() - 3U];
      if (shape.size() == 4U)
        extents.max_tx = shape[0];
      DopplerNotchParams params;
      params.center_velocity = hard_noncyclic ? 1.75 : 0.0;
      params.stop_half_width = 0.0;
      params.transition_width = hard_noncyclic ? 0.0 : 2.0;
      params.cyclic_velocity_axis = !hard_noncyclic;
      auto fixture = holoscan::testing::create_operator_test<Op>(extents);
      fixture.set_params(std::move(params));
      fixture.push_frame(holoscan::testing::frame_at(
          std::chrono::nanoseconds{0}, holoscan::testing::input("velocity_bins", *axis),
          holoscan::testing::input("rd_maps", *input)));
      fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 1U, input->nbytes());
      fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
      fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
      fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
      auto running = fixture.run_async({.expect_stop = true});
      (void)running.wait_for_output("rd_maps_out", 1U, std::chrono::seconds{5});
      running.request_stop();
      const auto result = running.finish();
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      const auto frames = host_frames(result);
      ASSERT_EQ(frames.size(), 1U);
      ASSERT_EQ(frames[0].shape, shape);
      const auto* actual = frames[0].template as<std::complex<T>>();
      // Closed-form physical answers: cosine half-gain at +/-1, hard nearest
      // to1.75 is+2; these tables do not call the production gain helper.
      const std::vector<T> gains =
          doppler == 5 ? (hard_noncyclic ? std::vector<T>{1, 1, 1, 1, 0}
                                         : std::vector<T>{1, T{0.5}, 0, T{0.5}, 1})
                       : (hard_noncyclic ? std::vector<T>{1, 1, 1, 1, 1, 0, 1}
                                         : std::vector<T>{1, 1, T{0.5}, 0, T{0.5}, 1, 1});
      const double tolerance = std::is_same_v<T, float> ? 5e-6 : 1e-12;
      for (size_t index = 0; index < markers.size(); ++index) {
        const auto expected =
            markers[index] *
            gains[(index / static_cast<size_t>(range)) % static_cast<size_t>(doppler)];
        const double scale = std::max(1.0, static_cast<double>(std::abs(expected)));
        ASSERT_NEAR(actual[index].real(), expected.real(), tolerance * scale) << "index=" << index;
        ASSERT_NEAR(actual[index].imag(), expected.imag(), tolerance * scale) << "index=" << index;
      }
    }
  }
}

TYPED_TEST(DopplerNotchBoundary, SameInstanceRefreshesAxisShapeAndLiveGainsThenReturnsToParams) {
  using T = TypeParam;
  using Op = DopplerNotchOp<T>;
  const std::vector<std::vector<T>> axes{{-2, -1, 0, 1, 2}, {-4, -2, 0, 2, 4}, {-1, 0, 1},
                                         {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2},
                                         {-2, -1, 0, 1, 2}};
  MaxTensorExtents extents;
  extents.max_tx = 2;
  extents.max_rx = 2;
  extents.max_doppler_bins = 5;
  extents.max_range_bins = 5;
  DopplerNotchParams params;
  params.center_velocity = 0.0;
  params.stop_half_width = 0.0;
  params.transition_width = 2.0;
  params.cyclic_velocity_axis = false;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  std::vector<std::vector<std::complex<T>>> markers;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::shared_ptr<holoscan::Tensor>> input_owners;
  std::vector<std::shared_ptr<holoscan::Tensor>> axis_owners;
  for (size_t tick = 0; tick < axes.size(); ++tick) {
    const int64_t doppler = static_cast<int64_t>(axes[tick].size());
    const int64_t range = tick % 2U == 0U ? 3 : 5;
    shapes.push_back({2, 2, doppler, range});
    auto& values = markers.emplace_back(static_cast<size_t>(4 * doppler * range));
    for (size_t index = 0; index < values.size(); ++index) {
      const T value = static_cast<T>(tick * 256U + index + 1U);
      values[index] = {value, -value / T{8}};
    }
    input_owners.push_back(to_device_tensor(values, shapes.back(), complex_dtype<T>()));
    axis_owners.push_back(to_device_tensor(axes[tick], {doppler}, real_dtype<T>()));
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((tick + 1U) * 100U)};
    if (tick == 3U) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("center_velocity", 2.0),
          holoscan::testing::input("stop_half_width", 0.5),
          holoscan::testing::input("transition_width", 0.0),
          holoscan::testing::input("velocity_bins", *axis_owners.back()),
          holoscan::testing::input("rd_maps", *input_owners.back())));
    } else if (tick == 5U) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("transition_width", 4.0),
          holoscan::testing::input("velocity_bins", *axis_owners.back()),
          holoscan::testing::input("rd_maps", *input_owners.back())));
    } else {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("velocity_bins", *axis_owners.back()),
          holoscan::testing::input("rd_maps", *input_owners.back())));
    }
  }
  constexpr size_t kMaximumBytes = 64U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", axes.size(), kMaximumBytes);
  fixture.input_source_device("velocity_bins", holoscan::DeviceId{0});
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  (void)running.wait_for_output("rd_maps_out", axes.size(), std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = host_frames(result);
  ASSERT_EQ(frames.size(), axes.size());
  for (size_t tick = 0; tick < axes.size(); ++tick) {
    SCOPED_TRACE(tick);
    ASSERT_EQ(frames[tick].shape, shapes[tick]);
    const int64_t range = shapes[tick].back();
    std::vector<double> gains{1.0, 0.5, 0.0, 0.5, 1.0};
    if (tick == 1U)
      gains = {1.0, 1.0, 0.0, 1.0, 1.0};
    if (tick == 2U)
      gains = {0.5, 0.0, 0.5};
    if (tick == 3U)
      gains = {1.0, 1.0, 1.0, 1.0, 0.0};
    if (tick == 5U) {
      const double quarter_gain = (1.0 - std::sqrt(0.5)) / 2.0;
      gains = {0.5, quarter_gain, 0.0, quarter_gain, 0.5};
    }
    const auto* actual = frames[tick].template as<std::complex<T>>();
    const double tolerance = std::is_same_v<T, float> ? 5e-6 : 1e-12;
    for (size_t index = 0; index < markers[tick].size(); ++index) {
      const auto expected =
          markers[tick][index] *
          static_cast<T>(gains[(index / static_cast<size_t>(range)) % gains.size()]);
      const double scale = std::max(1.0, static_cast<double>(std::abs(expected)));
      ASSERT_NEAR(actual[index].real(), expected.real(), tolerance * scale) << "index=" << index;
      ASSERT_NEAR(actual[index].imag(), expected.imag(), tolerance * scale) << "index=" << index;
    }
  }
}

}  // namespace
