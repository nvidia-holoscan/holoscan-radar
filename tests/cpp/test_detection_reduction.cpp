// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DetectionReductionOp contract tests (group-by-label → rep_idx).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

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

template <typename T, typename... Args>
HostTensor run_reduction(std::shared_ptr<holoscan::Tensor> labels,
                         std::shared_ptr<holoscan::Tensor> detection_idx,
                         std::shared_ptr<holoscan::Tensor> points,
                         Args&&... args) {
  auto harness = create_operator_test<DetectionReductionOp<T>>(
      std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "labels", {single_map(std::move(labels), "labels")});
  harness->template add_input_port<holoscan::TensorMap>(
      "detection_idx",
      {single_map(std::move(detection_idx), "detection_idx")});
  if (points) {
    harness->template add_input_port<holoscan::TensorMap>(
        "points", {single_map(std::move(points), "points")});
  }
  TensorMapCapture cap;
  harness->template add_output_port<holoscan::TensorMap>("rep_idx",
                                                        cap.validators());
  harness->run_test();
  if (cap.frames.size() != 1u) {
    throw std::runtime_error("expected 1 rep_idx frame");
  }
  return cap.frames.front();
}

TEST(DetectionReductionOpTest, FirstSelectsLowestIndexPerCluster) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // rows: 0→cluster0, 1→cluster1, 2→cluster0, 3→noise
  std::vector<int64_t> labels = {0, 1, 0, -1};
  std::vector<int64_t> idx = {10, 1, 20, 2, 11, 3, 99, 9};
  auto lab_t = to_device_tensor(labels, {4}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {4, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  ASSERT_EQ(rep.shape[0], 2);
  ASSERT_EQ(rep.shape[1], 2);
  const int64_t* r = rep.as<int64_t>();
  // cluster 0 → row 0 (10,1); cluster 1 → row 1 (20,2)
  EXPECT_EQ(r[0], 10);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 20);
  EXPECT_EQ(r[3], 2);
}

TEST(DetectionReductionOpTest, PeakSelectsMaxWeightedSum) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0, 0, 1};
  std::vector<int64_t> idx = {0, 0, 1, 1, 2, 2};
  // scores in column 0: 1, 5, 3 → cluster0 picks row1; cluster1 picks row2
  std::vector<double> points = {1.0, 0.0, 5.0, 0.0, 3.0, 0.0};
  auto lab_t = to_device_tensor(labels, {3}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {3, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {3, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("score_method") = ScoreMethod::kWeightedSum,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0, 0.0});
  ASSERT_EQ(rep.shape[0], 2);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 1);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 2);
  EXPECT_EQ(r[3], 2);
}

TEST(DetectionReductionOpTest, PeakSelectsMaxWeightedL2) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // sum: row0=7 > row1=6; L2: row1=6 > row0=5 — L2 must pick row1.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {4.0, 3.0, 6.0, 0.0};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("score_method") = ScoreMethod::kWeightedL2,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0, 1.0});
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 20);
  EXPECT_EQ(rep.as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, PeakSelectsMaxWeightedMax) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // sum: row1=17 > row0=11; max: row0=10 > row1=9 — max must pick row0.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {10.0, 1.0, 9.0, 8.0};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("score_method") = ScoreMethod::kWeightedMax,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0, 1.0});
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, MedoidRespectsFeatureWeights) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // One cluster, three points on a line; weights emphasize dim 0.
  // Centroid of scaled feats ≈ (2,0); nearest is row1 (value 2).
  std::vector<int64_t> labels = {0, 0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 0, 2, 0};
  std::vector<double> points = {0.0, 10.0, 2.0, 10.0, 4.0, 10.0};
  auto lab_t = to_device_tensor(labels, {3}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {3, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {3, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      holoscan::Arg("representative") = RepresentativePolicy::kMedoid,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0, 0.0});
  ASSERT_EQ(rep.shape[0], 1);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 1);
  EXPECT_EQ(r[1], 0);
}

TEST(DetectionReductionOpTest, ExcludesNoiseAndEmitsOnePerCluster) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {-1, 0, -1, 1};
  std::vector<int64_t> idx = {9, 9, 0, 1, 8, 8, 2, 3};
  auto lab_t = to_device_tensor(labels, {4}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {4, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  ASSERT_EQ(rep.shape[0], 2);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 0);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 2);
  EXPECT_EQ(r[3], 3);
}

TEST(DetectionReductionOpTest, RejectsPeakWithoutPoints) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  auto lab_t = to_device_tensor(labels, {1}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(
      run_reduction<float>(
          lab_t, idx_t, nullptr,
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("feature_weights") = std::vector<double>{1.0}),
      std::invalid_argument);
}

TEST(DetectionReductionOpTest, RejectsMismatchedFeatureWeights) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<double> points = {1.0, 2.0};
  auto lab_t = to_device_tensor(labels, {1}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {1, 2}, real_dtype<double>());
  EXPECT_THROW(
      run_reduction<double>(
          lab_t, idx_t, pts,
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("feature_weights") = std::vector<double>{1.0}),
      std::invalid_argument);
}

TEST(DetectionReductionOpTest, FloatDoubleAgreeOnPeak) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 1};
  std::vector<double> pd = {1.0, 4.0};
  std::vector<float> pf = {1.0f, 4.0f};
  auto lab_d = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto lab_f = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_d = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto idx_f = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts_d = to_device_tensor(pd, {2, 1}, real_dtype<double>());
  auto pts_f = to_device_tensor(pf, {2, 1}, real_dtype<float>());
  const auto rd = run_reduction<double>(
      lab_d, idx_d, pts_d,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0});
  const auto rf = run_reduction<float>(
      lab_f, idx_f, pts_f,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0});
  EXPECT_EQ(rd.as<int64_t>()[0], rf.as<int64_t>()[0]);
  EXPECT_EQ(rd.as<int64_t>()[1], rf.as<int64_t>()[1]);
}

TEST(DetectionReductionOpTest, EmptyInputEmitsZeroClusters) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels;
  std::vector<int64_t> idx;
  auto lab_t = to_device_tensor(labels, {0}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  EXPECT_EQ(rep.shape[0], 0);
  ASSERT_EQ(rep.shape.size(), 2u);
  EXPECT_EQ(rep.shape[1], 2);
}

TEST(DetectionReductionOpTest, AllNoiseEmitsZeroClusters) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {-1, -1};
  std::vector<int64_t> idx = {1, 2, 3, 4};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  EXPECT_EQ(rep.shape[0], 0);
  ASSERT_EQ(rep.shape.size(), 2u);
  EXPECT_EQ(rep.shape[1], 2);
}

TEST(DetectionReductionOpTest, SparseLabelGapsSkipped) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // Labels 0 and 2 (gap at 1) → two clusters, compact order by label id.
  std::vector<int64_t> labels = {0, 2};
  std::vector<int64_t> idx = {10, 1, 30, 3};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  ASSERT_EQ(rep.shape[0], 2);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
  EXPECT_EQ(rep.as<int64_t>()[2], 30);
  EXPECT_EQ(rep.as<int64_t>()[3], 3);
}

TEST(DetectionReductionOpTest, WidenedDetectionIdxCols) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0};
  // (doppler, range, batch) — third column preserved in rep_idx.
  std::vector<int64_t> idx = {7, 8, 9};
  auto lab_t = to_device_tensor(labels, {1}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {1, 3}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst);
  ASSERT_EQ(rep.shape[0], 1);
  ASSERT_EQ(rep.shape[1], 3);
  EXPECT_EQ(rep.as<int64_t>()[0], 7);
  EXPECT_EQ(rep.as<int64_t>()[1], 8);
  EXPECT_EQ(rep.as<int64_t>()[2], 9);
}

TEST(DetectionReductionOpTest, PeakEqualScorePicksLowestIndex) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {5.0, 5.0};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 1}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      holoscan::Arg("representative") = RepresentativePolicy::kPeak,
      holoscan::Arg("feature_weights") = std::vector<double>{1.0});
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, CustomScoreMethodIgnoredForFirst) {
  RADAR_TEST_SKIP_IF_NO_GPU();
  // score_method is unused for first — custom must not require ScoreFn.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  auto lab_t = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      holoscan::Arg("representative") = RepresentativePolicy::kFirst,
      holoscan::Arg("score_method") = ScoreMethod::kCustom);
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

// Sticky [live] representative override: peak → first mid-stream, then sticky.
TEST(DetectionReductionOpTest, LiveRepresentativeStickyOverride) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 1};
  std::vector<float> points = {1.0f, 5.0f};
  auto lab = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto det = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 1}, real_dtype<float>());

  class App : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> labels_frames;
    std::vector<holoscan::TensorMap> idx_frames;
    std::vector<holoscan::TensorMap> points_frames;
    std::vector<RepresentativePolicy> policy_overrides;
    TensorMapCapture* capture = nullptr;

    void compose() override {
      auto op = make_operator<DetectionReductionOp<float>>(
          "reduction",
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("feature_weights") = std::vector<double>{1.0});
      add_operator(op);
      wire_data_source(this, op, "labels", labels_frames);
      wire_data_source(this, op, "detection_idx", idx_frames);
      wire_data_source(this, op, "points", points_frames);
      wire_leading_override<RepresentativePolicy>(this, op, "representative",
                                                  policy_overrides);
      wire_capture_sink(this, op, "rep_idx", capture);
    }
  };

  // Three identical data ticks; override only on ticks 0 and 1.
  // Tick 0: peak → row1 (1,1). Tick 1: first → row0 (0,0).
  // Tick 2: no override → sticky first → (0,0).
  App app;
  for (int i = 0; i < 3; ++i) {
    app.labels_frames.push_back(single_map(lab, "labels"));
    app.idx_frames.push_back(single_map(det, "detection_idx"));
    app.points_frames.push_back(single_map(pts, "points"));
  }
  app.policy_overrides = {RepresentativePolicy::kPeak,
                          RepresentativePolicy::kFirst};
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
  expect_cell(cap.frames[0], 1, 1);
  expect_cell(cap.frames[1], 0, 0);
  expect_cell(cap.frames[2], 0, 0);
}

// Sticky [live] score_method: weighted_sum → weighted_l2, then sticky.
TEST(DetectionReductionOpTest, LiveScoreMethodStickyOverride) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  // sum picks row0 (7>6); L2 picks row1 (6>5).
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<float> points = {4.0f, 3.0f, 6.0f, 0.0f};
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
      auto op = make_operator<DetectionReductionOp<float>>(
          "reduction",
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("score_method") = ScoreMethod::kWeightedSum,
          holoscan::Arg("feature_weights") = std::vector<double>{1.0, 1.0});
      add_operator(op);
      wire_data_source(this, op, "labels", labels_frames);
      wire_data_source(this, op, "detection_idx", idx_frames);
      wire_data_source(this, op, "points", points_frames);
      wire_leading_override<ScoreMethod>(this, op, "score_method",
                                         method_overrides);
      wire_capture_sink(this, op, "rep_idx", capture);
    }
  };

  App app;
  for (int i = 0; i < 3; ++i) {
    app.labels_frames.push_back(single_map(lab, "labels"));
    app.idx_frames.push_back(single_map(det, "detection_idx"));
    app.points_frames.push_back(single_map(pts, "points"));
  }
  app.method_overrides = {ScoreMethod::kWeightedSum, ScoreMethod::kWeightedL2};
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
  expect_cell(cap.frames[0], 10, 1);  // sum → row0
  expect_cell(cap.frames[1], 20, 2);  // L2 → row1
  expect_cell(cap.frames[2], 20, 2);  // sticky L2
}

// Sticky [live] feature_weights: emphasize col0 → col1, then sticky.
TEST(DetectionReductionOpTest, LiveFeatureWeightsStickyOverride) {
  RADAR_TEST_SKIP_IF_NO_GPU();

  // One-hot col0 → row0 (5>1); one-hot col1 → row1 (10>0).
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<float> points = {5.0f, 0.0f, 1.0f, 10.0f};
  auto lab = to_device_tensor(labels, {2}, DLDataType{kDLInt, 64, 1});
  auto det = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<float>());

  class App : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> labels_frames;
    std::vector<holoscan::TensorMap> idx_frames;
    std::vector<holoscan::TensorMap> points_frames;
    std::vector<std::vector<double>> weight_overrides;
    TensorMapCapture* capture = nullptr;

    void compose() override {
      auto op = make_operator<DetectionReductionOp<float>>(
          "reduction",
          holoscan::Arg("representative") = RepresentativePolicy::kPeak,
          holoscan::Arg("score_method") = ScoreMethod::kWeightedSum,
          holoscan::Arg("feature_weights") = std::vector<double>{1.0, 0.0});
      add_operator(op);
      wire_data_source(this, op, "labels", labels_frames);
      wire_data_source(this, op, "detection_idx", idx_frames);
      wire_data_source(this, op, "points", points_frames);
      wire_leading_override<std::vector<double>>(this, op, "feature_weights",
                                                 weight_overrides);
      wire_capture_sink(this, op, "rep_idx", capture);
    }
  };

  App app;
  for (int i = 0; i < 3; ++i) {
    app.labels_frames.push_back(single_map(lab, "labels"));
    app.idx_frames.push_back(single_map(det, "detection_idx"));
    app.points_frames.push_back(single_map(pts, "points"));
  }
  app.weight_overrides = {std::vector<double>{1.0, 0.0},
                          std::vector<double>{0.0, 1.0}};
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
  expect_cell(cap.frames[0], 10, 1);  // w=(1,0) → row0
  expect_cell(cap.frames[1], 20, 2);  // w=(0,1) → row1
  expect_cell(cap.frames[2], 20, 2);  // sticky
}

}  // namespace
