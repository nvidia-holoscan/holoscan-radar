// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Graph coverage for a caller-instantiated DetectionReductionOp ScoreFn.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "detection/detection_reduction_op/detection_reduction_op_impl.cuh"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
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

template class holoscan::radar::DetectionReductionOp<float,
                                                     detection_reduction_test::Col0MinusCol1>;
template class holoscan::radar::DetectionReductionOp<double,
                                                     detection_reduction_test::Col0MinusCol1>;

namespace {

using detection_reduction_test::Col0MinusCol1;
using holoscan::radar::DetectionReductionOp;
using holoscan::radar::DetectionReductionParams;
using holoscan::radar::RepresentativePolicy;
using holoscan::radar::ScoreMethod;
using holoscan::radar::real_dtype;
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

  using Op = DetectionReductionOp<double, Col0MinusCol1>;
  auto labels_tensor = to_device_tensor(labels, {2}, kInt64Dtype);
  auto idx_tensor = to_device_tensor(idx, {2, 2}, kInt64Dtype);
  auto points_tensor = to_device_tensor(points, {2, 2}, real_dtype<double>());
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{0}, holoscan::testing::input("labels", *labels_tensor),
      holoscan::testing::input("detection_idx", *idx_tensor),
      holoscan::testing::input("points", *points_tensor)));
  fixture.template capture_snapshot<holoscan::Tensor>("rep_idx", 1U, idx_tensor->nbytes());
  fixture.template capture_snapshot<holoscan::Tensor>("reduced_points", 1U,
                                                      points_tensor->nbytes());
  fixture.input_source_device("labels", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.output_device("reduced_points", holoscan::DeviceId{0});
  fixture.output_device("rep_idx", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("rep_idx", 1U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_TRUE(observed);
  const auto output =
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("rep_idx"));
  ASSERT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.as<int64_t>()[0], 10);
  EXPECT_EQ(output.as<int64_t>()[1], 1);
}

}  // namespace
