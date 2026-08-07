// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Custom ScoreFn path for DetectionReductionOp — compiled as CUDA so the
// functor is device-callable and the op specialization is instantiated.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "detection/detection_reduction_op/detection_reduction_op_impl.cuh"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace detection_reduction_test {
// score = col0 - col1 (nonlinear vs weighted_sum of the same columns).
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
using holoscan::radar::DetectionReductionOp;
using holoscan::radar::RepresentativePolicy;
using holoscan::radar::ScoreMethod;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;
using detection_reduction_test::Col0MinusCol1;

TEST(DetectionReductionOpTest, CustomScoreFnSelectsPeak) {
  // Cluster 0: row0 score 1-0=1, row1 score 3-5=-2 → peak is row0.
  // weighted_sum with equal weights would pick row1 (3+5 > 1+0).
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {1.0, 0.0, 3.0, 5.0};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<double>());

  auto harness =
      create_operator_test<DetectionReductionOp<double, Col0MinusCol1>>(
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("score_method") = ScoreMethod::kCustom);
  harness->template add_input_port<holoscan::TensorMap>(
      "labels", {single_map(std::move(lab_t), "labels")});
  harness->template add_input_port<holoscan::TensorMap>(
      "detection_idx", {single_map(std::move(idx_t), "detection_idx")});
  harness->template add_input_port<holoscan::TensorMap>(
      "points", {single_map(std::move(pts), "points")});
  TensorMapCapture cap;
  harness->template add_output_port<holoscan::TensorMap>("rep_idx",
                                                        cap.validators());
  harness->run_test();
  ASSERT_EQ(cap.frames.size(), 1u);
  const auto& rep = cap.frames.front();
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, RejectsCustomWithoutScoreFn) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<float> points = {1.0f};
  auto lab_t = to_device_tensor(labels, {1}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {1, 1}, real_dtype<float>());

  auto harness = create_operator_test<DetectionReductionOp<float>>(
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("score_method") = ScoreMethod::kCustom);
  harness->template add_input_port<holoscan::TensorMap>(
      "labels", {single_map(std::move(lab_t), "labels")});
  harness->template add_input_port<holoscan::TensorMap>(
      "detection_idx", {single_map(std::move(idx_t), "detection_idx")});
  harness->template add_input_port<holoscan::TensorMap>(
      "points", {single_map(std::move(pts), "points")});
  TensorMapCapture cap;
  harness->template add_output_port<holoscan::TensorMap>("rep_idx",
                                                        cap.validators());
  EXPECT_THROW(harness->run_test(), std::invalid_argument);
}

// Sticky live score_method into custom and back out.
TEST(DetectionReductionOpTest, LiveScoreMethodIntoAndOutOfCustom) {
  // sum (1,1): row1 wins (8>1). custom col0-col1: row0 wins (1-0 > 3-5).
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<float> points = {1.0f, 0.0f, 3.0f, 5.0f};
  auto lab = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto det = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<float>());

  class App : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> labels_frames;
    std::vector<holoscan::TensorMap> idx_frames;
    std::vector<holoscan::TensorMap> points_frames;
    std::vector<ScoreMethod> method_overrides;
    TensorMapCapture* capture = nullptr;

    void compose() override {
      auto op =
          make_operator<DetectionReductionOp<float, Col0MinusCol1>>(
              "reduction",
              holoscan::Arg("representative") = RepresentativePolicy::kPeak,
              holoscan::Arg("score_method") = ScoreMethod::kWeightedSum,
              holoscan::Arg("feature_weights") =
                  std::vector<double>{1.0, 1.0});
      add_operator(op);
      wire_data_source(this, op, "labels", labels_frames);
      wire_data_source(this, op, "detection_idx", idx_frames);
      wire_data_source(this, op, "points", points_frames);
      wire_leading_override<ScoreMethod>(this, op, "score_method",
                                         method_overrides);
      wire_capture_sink(this, op, "rep_idx", capture);
    }
  };

  // Tick0: sum → row1. Tick1: custom → row0. Tick2: sum → row1.
  App app;
  for (int i = 0; i < 3; ++i) {
    app.labels_frames.push_back(single_map(lab, "labels"));
    app.idx_frames.push_back(single_map(det, "detection_idx"));
    app.points_frames.push_back(single_map(pts, "points"));
  }
  app.method_overrides = {ScoreMethod::kWeightedSum, ScoreMethod::kCustom,
                          ScoreMethod::kWeightedSum};
  TensorMapCapture cap;
  app.capture = &cap;
  app.config();
  app.run();

  ASSERT_EQ(cap.frames.size(), 3u);
  auto expect_cell = [](const HostTensor& frame, int64_t a, int64_t b) {
    ASSERT_EQ(frame.shape[0], 1);
    ASSERT_EQ(frame.shape[1], 2);
    EXPECT_EQ(frame.as<int64_t>()[0], a);
    EXPECT_EQ(frame.as<int64_t>()[1], b);
  };
  expect_cell(cap.frames[0], 20, 2);
  expect_cell(cap.frames[1], 10, 1);
  expect_cell(cap.frames[2], 20, 2);
}

}  // namespace
