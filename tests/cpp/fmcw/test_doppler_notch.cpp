// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace radar_testing = holoscan::radar::testing;

using holoscan::radar::DopplerNotchOp;
using holoscan::radar::DopplerNotchParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using radar_testing::run_dual_in_single_out_operator_graph_result;
using radar_testing::run_exact_override_dual_in_single_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr int64_t kDoppler = 5;
constexpr int64_t kRange = 2;

auto notch_wire() {
  return [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink) {
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out0, op->velocity_bins, queue);
    graph.add_flow(source->out1, op->rd_maps, queue);
    graph.add_flow(op->rd_maps_out, sink->in, queue);
  };
}

auto center_override_wire() {
  return [](holoscan::Graph& graph, const auto& source, auto& op, const auto& sink) {
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    graph.add_flow(source->out0, op->velocity_bins, queue);
    graph.add_flow(source->out1, op->rd_maps, queue);
    graph.add_flow(source->override_out, op->center_velocity, queue);
    graph.add_flow(op->rd_maps_out, sink->in, queue);
  };
}

MaxTensorExtents extents_for_rank(std::size_t rank) {
  MaxTensorExtents extents;
  extents.max_doppler_bins = kDoppler;
  extents.max_range_bins = kRange;
  if (rank >= 3U)
    extents.max_rx = 2;
  if (rank == 4U)
    extents.max_tx = 2;
  return extents;
}

template <typename T>
std::shared_ptr<holoscan::Tensor> make_axis(const std::vector<T>& values = std::vector<T>{
                                                T{-2}, T{-1}, T{0}, T{1}, T{2}}) {
  return to_device_tensor<T>(values, {kDoppler}, real_dtype<T>());
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
GraphRunResult run_notch(const std::vector<int64_t>& shape, DopplerNotchParams params,
                         std::shared_ptr<holoscan::Tensor> axis = make_axis<T>()) {
  return run_dual_in_single_out_operator_graph_result<DopplerNotchOp<T>>(
      {std::move(axis)}, {make_maps<T>(shape)}, std::move(params), "rd_maps_out", notch_wire(),
      extents_for_rank(shape.size()));
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
  const GraphRunResult result = run_notch<float>({kDoppler, kRange}, DopplerNotchParams{});
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  expect_gain(result.frames[0], 0, 1.0F);
  expect_gain(result.frames[0], 1, 1.0F);
  expect_gain(result.frames[0], 2, 0.0F);
  expect_gain(result.frames[0], 3, 1.0F);
  expect_gain(result.frames[0], 4, 1.0F);
}

TEST(DopplerNotchOp, RaisedCosineMatchesPhysicalAxis) {
  DopplerNotchParams params;
  params.transition_width = 2.0;
  const GraphRunResult result = run_notch<double>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  expect_gain(result.frames[0], 0, 1.0);
  expect_gain(result.frames[0], 1, 0.5);
  expect_gain(result.frames[0], 2, 0.0);
  expect_gain(result.frames[0], 3, 0.5);
  expect_gain(result.frames[0], 4, 1.0);
}

TEST(DopplerNotchOp, CyclicDistanceWrapsAtAxisEdge) {
  DopplerNotchParams params;
  params.center_velocity = 2.0;
  params.transition_width = 2.0;
  const GraphRunResult result = run_notch<float>({kDoppler, kRange}, params);
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  expect_gain(result.frames[0], 0, 0.5F);
  expect_gain(result.frames[0], 4, 0.0F);
}

TEST(DopplerNotchOp, NamedRanksPreserveShapeAndChannelPhase) {
  for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{
           {kDoppler, kRange}, {2, kDoppler, kRange}, {2, 2, kDoppler, kRange}}) {
    DopplerNotchParams params;
    params.transition_width = 2.0;
    const GraphRunResult result = run_notch<double>(shape, params);
    ASSERT_TRUE(result.plan_ok) << result.plan_json;
    ASSERT_TRUE(result.diagnostics.empty());
    ASSERT_EQ(result.frames.size(), 1U);
    EXPECT_EQ(result.frames[0].shape, shape);
    const auto* values = result.frames[0].as<std::complex<double>>();
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
  DopplerNotchParams params;
  const auto result =
      run_exact_override_dual_in_single_out_operator_graph_result<DopplerNotchOp<float>, double>(
          {make_axis<float>(), make_axis<float>()},
          {make_maps<float>({kDoppler, kRange}), make_maps<float>({kDoppler, kRange})},
          {1.0, std::nullopt}, params, "rd_maps_out", center_override_wire(), extents_for_rank(2U));
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 2U);
  expect_gain(result.frames[0], 3, 0.0F);
  expect_gain(result.frames[1], 2, 0.0F);
}

TEST(DopplerNotchOp, RejectsVelocityLengthMismatch) {
  auto short_axis = to_device_tensor<float>({-1.0F, 0.0F, 1.0F}, {3}, real_dtype<float>());
  const GraphRunResult result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, std::move(short_axis));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DopplerNotchOp, RejectsPrecisionMismatch) {
  const GraphRunResult result =
      run_notch<float>({kDoppler, kRange}, DopplerNotchParams{}, make_axis<double>());
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DopplerNotchOp, RejectsNegativeWidths) {
  DopplerNotchParams params;
  params.stop_half_width = -1.0;
  const GraphRunResult result = run_notch<float>({kDoppler, kRange}, params);
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
