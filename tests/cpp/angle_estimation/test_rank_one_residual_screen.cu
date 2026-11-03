// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_stage.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AngleConvention;

template <typename T>
std::vector<cuda::std::complex<T>> steering(const std::vector<double>& positions, double az) {
  std::vector<cuda::std::complex<T>> result(positions.size() / 3);
  for (size_t channel = 0; channel < result.size(); ++channel) {
    const double phase = holoscan::radar::kTwoPi * positions[channel * 3 + 1] * std::sin(az);
    result[channel] = {static_cast<T>(std::cos(phase)), static_cast<T>(std::sin(phase))};
  }
  return result;
}

template <typename T>
std::vector<T> run_screen(const std::vector<cuda::std::complex<T>>& snapshot,
                          const std::vector<T>& aoa, const std::vector<unsigned char>& valid,
                          const std::vector<double>& positions, int64_t items, int64_t channels,
                          int64_t looks, T threshold, std::vector<unsigned char>* selected_out,
                          AngleConvention convention = {}) {
  cuda::std::complex<T>* snapshot_device = nullptr;
  T* aoa_device = nullptr;
  unsigned char* valid_device = nullptr;
  double* positions_device = nullptr;
  T* scores_device = nullptr;
  unsigned char* selected_device = nullptr;
  EXPECT_EQ(cudaMalloc(&snapshot_device, snapshot.size() * sizeof(snapshot[0])), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&aoa_device, aoa.size() * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&positions_device, positions.size() * sizeof(double)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&scores_device, items * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&selected_device, items), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(snapshot_device, snapshot.data(), snapshot.size() * sizeof(snapshot[0]),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(aoa_device, aoa.data(), aoa.size() * sizeof(T), cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(positions_device, positions.data(), positions.size() * sizeof(double),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  if (!valid.empty()) {
    EXPECT_EQ(cudaMalloc(&valid_device, valid.size()), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(valid_device, valid.data(), valid.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  holoscan::radar::detail::residual_screen_kernel<T>
      <<<1, 64>>>(snapshot_device, aoa_device, valid_device, positions_device, items, channels,
                  looks, 1.0, convention, threshold, scores_device, selected_device);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<T> scores(static_cast<size_t>(items));
  selected_out->resize(static_cast<size_t>(items));
  EXPECT_EQ(
      cudaMemcpy(scores.data(), scores_device, scores.size() * sizeof(T), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cudaMemcpy(selected_out->data(), selected_device, selected_out->size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(selected_device);
  cudaFree(scores_device);
  cudaFree(positions_device);
  cudaFree(valid_device);
  cudaFree(aoa_device);
  cudaFree(snapshot_device);
  return scores;
}

template <typename T>
T cpu_residual(const std::vector<cuda::std::complex<T>>& snapshot,
               const std::vector<cuda::std::complex<T>>& steering_vector, int64_t channels,
               int64_t looks, const std::vector<unsigned char>& valid = {}) {
  double total = 0.0;
  double projected = 0.0;
  for (int64_t look = 0; look < looks; ++look) {
    if (!valid.empty() && valid[static_cast<size_t>(look)] == 0) {
      continue;
    }
    cuda::std::complex<double> dot{0.0, 0.0};
    for (int64_t channel = 0; channel < channels; ++channel) {
      const auto x = snapshot[static_cast<size_t>(channel * looks + look)];
      const auto a = steering_vector[static_cast<size_t>(channel)];
      dot += cuda::std::conj(cuda::std::complex<double>{a.real(), a.imag()}) *
             cuda::std::complex<double>{x.real(), x.imag()};
      total += static_cast<double>(x.real()) * x.real() + static_cast<double>(x.imag()) * x.imag();
    }
    projected += dot.real() * dot.real() + dot.imag() * dot.imag();
  }
  return static_cast<T>(1.0 - projected / (channels * total));
}

template <typename T>
void check_single_and_two_source_residuals() {
  constexpr int64_t kChannels = 8;
  constexpr int64_t kLooks = 3;
  std::vector<double> positions;
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    positions.insert(positions.end(), {0.0, 0.5 * channel, 0.0});
  }
  const auto primary = steering<T>(positions, 0.0);
  const auto secondary = steering<T>(positions, 0.22);
  std::vector<cuda::std::complex<T>> snapshots(static_cast<size_t>(2 * kChannels * kLooks));
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    for (int64_t look = 0; look < kLooks; ++look) {
      snapshots[channel * kLooks + look] = primary[channel];
      snapshots[(kChannels + channel) * kLooks + look] = primary[channel] + secondary[channel];
    }
  }
  std::vector<unsigned char> selected;
  const auto scores = run_screen<T>(snapshots, {T{0}, T{0}, T{0}, T{0}}, {}, positions, 2,
                                    kChannels, kLooks, static_cast<T>(0.03), &selected);
  ASSERT_EQ(scores.size(), 2U);
  const double tolerance = std::is_same_v<T, float> ? 1e-5 : 1e-12;
  EXPECT_NEAR(scores[0], T{0}, tolerance);
  std::vector<cuda::std::complex<T>> mixture(static_cast<size_t>(kChannels * kLooks));
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    for (int64_t look = 0; look < kLooks; ++look) {
      mixture[static_cast<size_t>(channel * kLooks + look)] = primary[channel] + secondary[channel];
    }
  }
  EXPECT_NEAR(scores[1], cpu_residual(mixture, primary, kChannels, kLooks), tolerance);
  EXPECT_GT(scores[1], static_cast<T>(0.03));
  EXPECT_EQ(selected, (std::vector<unsigned char>{0, 1}));
}

TEST(RankOneResidualScreenStage, SeparatesRankOneAndMixtureFloat) {
  check_single_and_two_source_residuals<float>();
}

TEST(RankOneResidualScreenStage, SeparatesRankOneAndMixtureDouble) {
  check_single_and_two_source_residuals<double>();
}

TEST(RankOneResidualScreenStage, ValidityMaskExcludesContaminatedLook) {
  constexpr int64_t kChannels = 8;
  std::vector<double> positions;
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    positions.insert(positions.end(), {0.0, 0.5 * channel, 0.0});
  }
  const auto primary = steering<double>(positions, 0.0);
  const auto secondary = steering<double>(positions, 0.25);
  std::vector<cuda::std::complex<double>> snapshots(kChannels * 2);
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    snapshots[channel * 2] = primary[channel];
    snapshots[channel * 2 + 1] = primary[channel] + secondary[channel];
  }
  std::vector<unsigned char> selected;
  const auto scores = run_screen<double>(snapshots, {0.0, 0.0}, {1, 0}, positions, 1, kChannels, 2,
                                         0.03, &selected);
  EXPECT_NEAR(scores[0], 0.0, 1e-12);
  EXPECT_EQ(selected[0], 0);
}

TEST(RankOneResidualScreenStage, SupportsDirectionCosineConvention) {
  constexpr int64_t kChannels = 8;
  constexpr double kAzimuth = 0.2;
  std::vector<double> positions;
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    positions.insert(positions.end(), {0.0, 0.5 * channel, 0.0});
  }
  const auto snapshot = steering<double>(positions, kAzimuth);
  AngleConvention convention;
  convention.preset = holoscan::radar::AngleConventionPreset::kDirectionCosine;
  std::vector<unsigned char> selected;
  const auto scores = run_screen<double>(snapshot, {std::sin(kAzimuth), 0.0}, {}, positions, 1,
                                         kChannels, 1, 0.03, &selected, convention);
  EXPECT_NEAR(scores[0], 0.0, 1e-12);
  EXPECT_EQ(selected[0], 0);
}

TEST(RankOneResidualScreenStage, NoValidLooksProducesNanAndNoSelection) {
  constexpr int64_t kChannels = 4;
  std::vector<double> positions;
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    positions.insert(positions.end(), {0.0, 0.5 * channel, 0.0});
  }
  const auto primary = steering<float>(positions, 0.0);
  std::vector<unsigned char> selected;
  const auto scores =
      run_screen<float>(primary, {0.0F, 0.0F}, {0}, positions, 1, kChannels, 1, 0.03F, &selected);
  EXPECT_TRUE(std::isnan(scores[0]));
  EXPECT_EQ(selected[0], 0);
}

holoscan::radar::ArrayGeometry make_screen_geometry(int64_t channels, double spacing) {
  holoscan::radar::ArrayGeometry geometry;
  geometry.wavelength = 1.0;
  geometry.geometry_type = holoscan::radar::ArrayGeometryType::kUla;
  for (int64_t channel = 0; channel < channels; ++channel) {
    geometry.element_positions.insert(geometry.element_positions.end(),
                                      {0.0, spacing * static_cast<double>(channel), 0.0});
  }
  return geometry;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_graph_snapshot(
    int64_t items, const holoscan::radar::ArrayGeometry& geometry, double azimuth) {
  const auto single = steering<T>(geometry.element_positions, azimuth);
  std::vector<cuda::std::complex<T>> values(static_cast<size_t>(items * geometry.num_elements()));
  for (int64_t item = 0; item < items; ++item) {
    std::copy(single.begin(), single.end(), values.begin() + item * geometry.num_elements());
  }
  return holoscan::radar::testing::to_device_tensor(values, {items, geometry.num_elements(), 1},
                                                    holoscan::radar::complex_dtype<T>());
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_graph_baseline(int64_t items, T azimuth) {
  std::vector<T> values(static_cast<size_t>(items * 2), T{0});
  for (int64_t item = 0; item < items; ++item) {
    values[static_cast<size_t>(item * 2)] = azimuth;
  }
  return holoscan::radar::testing::to_device_tensor(values, {items, 2},
                                                    holoscan::radar::real_dtype<T>());
}

template <typename T>
holoscan::testing::TestResult run_screen_graph(
    std::vector<std::shared_ptr<holoscan::Tensor>> snapshots,
    std::vector<std::shared_ptr<holoscan::Tensor>> baselines,
    holoscan::radar::RankOneResidualScreenParams params,
    const holoscan::radar::MaxTensorExtents& extents) {
  using Op = holoscan::radar::RankOneResidualScreenOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  const std::size_t count = snapshots.size();
  for (std::size_t index = 0; index < count; ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto unmatched_valid = holoscan::radar::testing::to_device_tensor(
        std::vector<uint8_t>{1}, {1, 1}, DLDataType{kDLUInt, 8U, 1U});
    fixture.push_at("is_valid", *unmatched_valid, capture_time + std::chrono::nanoseconds{1});
    fixture.push_frame(holoscan::testing::frame_at(
        capture_time, holoscan::testing::input("snapshot", *snapshots[index]),
        holoscan::testing::input("baseline_aoa", *baselines[index])));
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  for (const char* port :
       {"residual_score", "snapshot_out", "selected_idx", "is_valid_out", "selection_mask"}) {
    fixture.template capture_snapshot<holoscan::Tensor>(port, count, kMaximumBytes);
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.input_source_device("baseline_aoa", holoscan::DeviceId{0});
  fixture.input_source_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* port :
       {"residual_score", "snapshot_out", "selected_idx", "is_valid_out", "selection_mask"}) {
    (void)completion.wait_for_output(running, port, count);
  }
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

TEST(RankOneResidualScreenOpTest, LiveGeometryMatchThenParamsFallback) {
  constexpr int64_t kChannels = 8;
  constexpr double kAzimuth = 0.3;
  const auto params_geometry = make_screen_geometry(kChannels, 0.5);
  const auto override_geometry = make_screen_geometry(kChannels, 0.25);
  auto snapshot0 = make_graph_snapshot<float>(1, override_geometry, kAzimuth);
  auto snapshot1 = make_graph_snapshot<float>(1, override_geometry, kAzimuth);
  auto baseline0 = make_graph_baseline<float>(1, kAzimuth);
  auto baseline1 = make_graph_baseline<float>(1, kAzimuth);

  holoscan::radar::RankOneResidualScreenParams params;
  params.array_geometry = params_geometry;
  params.residual_threshold = 0.03;
  holoscan::radar::MaxTensorExtents extents;
  extents.max_items = 1;
  extents.max_channels = kChannels;
  extents.max_snapshots = 1;

  using Op = holoscan::radar::RankOneResidualScreenOp<float>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  auto valid0 = holoscan::radar::testing::to_device_tensor(std::vector<uint8_t>{1}, {1, 1},
                                                           DLDataType{kDLUInt, 8U, 1U});
  auto valid1 = holoscan::radar::testing::to_device_tensor(std::vector<uint8_t>{1}, {1, 1},
                                                           DLDataType{kDLUInt, 8U, 1U});
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{100},
      holoscan::testing::input("array_geometry", holoscan::radar::to_wire(override_geometry)),
      holoscan::testing::input("is_valid", *valid0),
      holoscan::testing::input("snapshot", *snapshot0),
      holoscan::testing::input("baseline_aoa", *baseline0)));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{200}, holoscan::testing::input("is_valid", *valid1),
      holoscan::testing::input("snapshot", *snapshot1),
      holoscan::testing::input("baseline_aoa", *baseline1)));
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  for (const char* port :
       {"residual_score", "snapshot_out", "selected_idx", "is_valid_out", "selection_mask"}) {
    fixture.capture_snapshot<holoscan::Tensor>(port, 2U, kMaximumBytes);
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  fixture.input_source_device("snapshot", holoscan::DeviceId{0});
  fixture.input_source_device("baseline_aoa", holoscan::DeviceId{0});
  fixture.input_source_device("is_valid", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  for (const char* port :
       {"residual_score", "snapshot_out", "selected_idx", "is_valid_out", "selection_mask"}) {
    (void)completion.wait_for_output(running, port, 2U);
  }
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = result.snapshots<holoscan::Tensor>("residual_score");
  ASSERT_EQ(frames.size(), 2U);
  const auto first = holoscan::radar::testing::to_host(frames[0]);
  const auto second = holoscan::radar::testing::to_host(frames[1]);
  EXPECT_NEAR(first.as<float>()[0], 0.0F, 1e-5F);
  EXPECT_GT(second.as<float>()[0], 0.03F);
}

TEST(RankOneResidualScreenOpTest, EmptyInputEmitsEmptyDiagnostics) {
  const auto geometry = make_screen_geometry(4, 0.5);
  holoscan::radar::RankOneResidualScreenParams params;
  params.array_geometry = geometry;
  holoscan::radar::MaxTensorExtents extents;
  extents.max_items = 1;
  extents.max_channels = 4;
  extents.max_snapshots = 1;
  const auto result =
      run_screen_graph<float>({make_graph_snapshot<float>(0, geometry, 0.0)},
                              {make_graph_baseline<float>(0, 0.0F)}, std::move(params), extents);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = result.snapshots<holoscan::Tensor>("residual_score");
  ASSERT_EQ(frames.size(), 1U);
  const auto output = holoscan::radar::testing::to_host(frames[0]);
  EXPECT_EQ(output.shape, (std::vector<int64_t>{0}));
}

TEST(RankOneResidualScreenOpTest, RejectsThresholdOutsideUnitInterval) {
  const auto geometry = make_screen_geometry(4, 0.5);
  holoscan::radar::MaxTensorExtents extents;
  extents.max_items = 1;
  extents.max_channels = 4;
  extents.max_snapshots = 1;
  for (double threshold : {-0.01, 1.01}) {
    holoscan::radar::RankOneResidualScreenParams params;
    params.array_geometry = geometry;
    params.residual_threshold = threshold;
    const auto result =
        run_screen_graph<float>({make_graph_snapshot<float>(1, geometry, 0.0)},
                                {make_graph_baseline<float>(1, 0.0F)}, std::move(params), extents);
    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(RankOneResidualScreenOpTest, RejectsCapturedExtentOverflow) {
  const auto geometry = make_screen_geometry(4, 0.5);
  holoscan::radar::RankOneResidualScreenParams params;
  params.array_geometry = geometry;
  holoscan::radar::MaxTensorExtents extents;
  extents.max_items = 1;
  extents.max_channels = 4;
  extents.max_snapshots = 1;
  const auto result =
      run_screen_graph<double>({make_graph_snapshot<double>(2, geometry, 0.0)},
                               {make_graph_baseline<double>(2, 0.0)}, std::move(params), extents);
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

}  // namespace
