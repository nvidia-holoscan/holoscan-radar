// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Graph coverage for a caller-instantiated DetectionReductionOp ScoreFn.

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/detection_reduction_op/detection_reduction_op_impl.cuh"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace detection_reduction_test {

struct Col0MinusCol1 {
  template <typename T>
  __host__ __device__ T operator()(const T* row, int64_t n) const {
    if (n < 2) {
      return T{0};
    }
    return row[0] - row[1];
  }
};

}  // namespace detection_reduction_test

template class holoscan::radar::DetectionReductionOp<
    float, detection_reduction_test::Col0MinusCol1>;
template class holoscan::radar::DetectionReductionOp<
    double, detection_reduction_test::Col0MinusCol1>;

namespace {

using detection_reduction_test::Col0MinusCol1;
using holoscan::radar::DetectionReductionOp;
using holoscan::radar::DetectionReductionParams;
using holoscan::radar::RepresentativePolicy;
using holoscan::radar::ScoreMethod;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::run_triple_in_single_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

TEST(DetectionReductionOpCustomTest, CustomScoreFnSelectsPeak) {
  const std::vector<int64_t> labels = {0, 0};
  const std::vector<int64_t> idx = {10, 1, 20, 2};
  // custom: row0 = 1, row1 = -2; weighted_sum would pick row1.
  const std::vector<double> points = {1.0, 0.0, 3.0, 5.0};

  DetectionReductionParams params;
  params.representative = RepresentativePolicy::kPeak;
  params.score_method = ScoreMethod::kCustom;

  GraphRunResult result =
      run_triple_in_single_out_operator_graph_result<
          DetectionReductionOp<double, Col0MinusCol1>>(
          {to_device_tensor(labels, {2}, kInt64Dtype)},
          {to_device_tensor(idx, {2, 2}, kInt64Dtype)},
          {to_device_tensor(points, {2, 2}, real_dtype<double>())},
          std::move(params), "rep_idx",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->labels, q);
            graph.add_flow(source->out1, op->detection_idx, q);
            graph.add_flow(source->out2, op->points, q);
            graph.add_flow(op->rep_idx, sink->in, q);
          },
          holoscan::radar::testing::default_test_extents(),
          {"reduced_points"});

  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames.size(), 1U);
  ASSERT_EQ(result.frames[0].shape[0], 1);
  EXPECT_EQ(result.frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(result.frames[0].as<int64_t>()[1], 1);
}

}  // namespace
