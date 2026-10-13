// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DetectionReductionOp contract tests (HSDK 5 Graph harness): group-by-label
// → rep_idx — first/peak/medoid policies, score methods, noise exclusion,
// Params coverage, error paths, float/double agreement, and pageable-host
// feature_weights live overrides.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::DetectionReductionOp;
using holoscan::radar::DetectionReductionParams;
using holoscan::radar::MaxTensorExtents;
using holoscan::radar::RepresentativePolicy;
using holoscan::radar::ScoreMethod;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

constexpr DLDataType kInt64{kDLInt, 64, 1};
constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};

DetectionReductionParams make_params(
    RepresentativePolicy representative = RepresentativePolicy::kPeak,
    ScoreMethod score_method = ScoreMethod::kWeightedSum,
    std::vector<double> feature_weights = {}) {
  DetectionReductionParams params{};
  params.representative = representative;
  params.score_method = score_method;
  params.feature_weights = std::move(feature_weights);
  return params;
}

template <typename T>
holoscan::testing::TestResult run_reduction_result(std::shared_ptr<holoscan::Tensor> labels,
                                                   std::shared_ptr<holoscan::Tensor> detection_idx,
                                                   std::shared_ptr<holoscan::Tensor> points,
                                                   DetectionReductionParams params,
                                                   bool capture_reduced = false) {
  using Op = DetectionReductionOp<T>;
  if (!points && params.representative == RepresentativePolicy::kFirst) {
    const int64_t num_points = labels->shape()[0];
    points = to_device_tensor(std::vector<T>(static_cast<std::size_t>(num_points)), {num_points, 1},
                              real_dtype<T>());
  }
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  if (points) {
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{0}, holoscan::testing::input("points", *points),
        holoscan::testing::input("labels", *labels),
        holoscan::testing::input("detection_idx", *detection_idx)));
  } else {
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{0}, holoscan::testing::input("labels", *labels),
        holoscan::testing::input("detection_idx", *detection_idx)));
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("rep_idx", 1U, kMaximumBytes);
  if (capture_reduced) {
    fixture.template capture_snapshot<holoscan::Tensor>("reduced_points", 1U, kMaximumBytes);
  }
  fixture.input_source_device("labels", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  if (points) {
    fixture.input_source_device("points", holoscan::DeviceId{0});
  }
  fixture.output_device("rep_idx", holoscan::DeviceId{0});
  fixture.output_device("reduced_points", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool rep = running.wait_for_output("rep_idx", 1U, std::chrono::seconds{5});
  bool reduced = true;
  if (capture_reduced) {
    reduced = running.wait_for_output("reduced_points", 1U, std::chrono::seconds{5});
  }
  (void)rep;
  (void)reduced;
  running.request_stop();
  return running.finish();
}

template <typename T>
HostTensor run_reduction(std::shared_ptr<holoscan::Tensor> labels,
                         std::shared_ptr<holoscan::Tensor> detection_idx,
                         std::shared_ptr<holoscan::Tensor> points,
                         DetectionReductionParams params) {
  const auto result = run_reduction_result<T>(std::move(labels), std::move(detection_idx),
                                              std::move(points), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("rep_idx"));
}

template <typename T>
holoscan::testing::TestResult run_reduction_dual_result(
    std::shared_ptr<holoscan::Tensor> labels, std::shared_ptr<holoscan::Tensor> detection_idx,
    std::shared_ptr<holoscan::Tensor> points, DetectionReductionParams params) {
  return run_reduction_result<T>(std::move(labels), std::move(detection_idx), std::move(points),
                                 std::move(params), true);
}

template <typename T>
std::pair<HostTensor, HostTensor> run_reduction_dual(
    std::shared_ptr<holoscan::Tensor> labels, std::shared_ptr<holoscan::Tensor> detection_idx,
    std::shared_ptr<holoscan::Tensor> points, DetectionReductionParams params) {
  const auto result = run_reduction_dual_result<T>(std::move(labels), std::move(detection_idx),
                                                   std::move(points), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("rep_idx")),
          holoscan::radar::testing::to_host(
              result.template snapshot<holoscan::Tensor>("reduced_points"))};
}

// ---------------------------------------------------------------------------

TEST(DetectionReductionOpTest, RequiresMaxFeaturesForReducedPointsPool) {
  MaxTensorExtents extents;
  extents.max_clusters = 4;
  EXPECT_THROW((DetectionReductionOp<float>{extents}), std::invalid_argument);
}

TEST(DetectionReductionOpTest, UnwiredReducedPointsDoesNotSuppressRepIdxAcrossFrames) {
  std::vector<std::shared_ptr<holoscan::Tensor>> label_frames;
  std::vector<std::shared_ptr<holoscan::Tensor>> index_frames;
  std::vector<std::shared_ptr<holoscan::Tensor>> point_frames;
  for (int frame = 0; frame < 4; ++frame) {
    label_frames.push_back(to_device_tensor(std::vector<int64_t>{0, 0}, {2}, kInt64));
    index_frames.push_back(to_device_tensor(std::vector<int64_t>{1, 2, 3, 4}, {2, 2}, kInt64));
    point_frames.push_back(
        to_device_tensor(std::vector<double>{1.0, 2.0}, {2, 1}, real_dtype<double>()));
  }
  using Op = DetectionReductionOp<double>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  for (std::size_t index = 0; index < label_frames.size(); ++index) {
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::milliseconds{static_cast<int64_t>(index + 1U)},
        holoscan::testing::input("points", *point_frames[index]),
        holoscan::testing::input("labels", *label_frames[index]),
        holoscan::testing::input("detection_idx", *index_frames[index])));
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("rep_idx", 4U, kMaximumBytes);
  fixture.input_source_device("labels", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.output_device("rep_idx", holoscan::DeviceId{0});
  fixture.output_device("reduced_points", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("rep_idx", 4U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto frames = result.snapshots<holoscan::Tensor>("rep_idx");
  ASSERT_EQ(frames.size(), 4U);
  for (const auto& snapshot : frames) {
    const HostTensor frame = holoscan::radar::testing::to_host(snapshot);
    ASSERT_EQ(frame.shape, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(frame.as<int64_t>()[0], 3);
    EXPECT_EQ(frame.as<int64_t>()[1], 4);
  }
}

TEST(DetectionReductionOpTest, FirstSelectsLowestIndexPerCluster) {
  // rows: 0→cluster0, 1→cluster1, 2→cluster0, 3→noise
  std::vector<int64_t> labels = {0, 1, 0, -1};
  std::vector<int64_t> idx = {10, 1, 20, 2, 11, 3, 99, 9};
  auto lab_t = to_device_tensor(labels, {4}, kInt64);
  auto idx_t = to_device_tensor(idx, {4, 2}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
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
  std::vector<int64_t> labels = {0, 0, 1};
  std::vector<int64_t> idx = {0, 0, 1, 1, 2, 2};
  // scores in column 0: 1, 5, 3 → cluster0 picks row1; cluster1 picks row2
  std::vector<double> points = {1.0, 0.0, 5.0, 0.0, 3.0, 0.0};
  auto lab_t = to_device_tensor(labels, {3}, kInt64);
  auto idx_t = to_device_tensor(idx, {3, 2}, kInt64);
  auto pts = to_device_tensor(points, {3, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 0.0}));
  ASSERT_EQ(rep.shape[0], 2);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 1);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 2);
  EXPECT_EQ(r[3], 2);
}

TEST(DetectionReductionOpTest, PeakSelectsMaxWeightedL2) {
  // sum: row0=7 > row1=6; L2: row1=6 > row0=5 — L2 must pick row1.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {4.0, 3.0, 6.0, 0.0};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedL2, {1.0, 1.0}));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 20);
  EXPECT_EQ(rep.as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, PeakSelectsMaxWeightedMax) {
  // sum: row1=17 > row0=11; max: row0=10 > row1=9 — max must pick row0.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {10.0, 1.0, 9.0, 8.0};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts = to_device_tensor(points, {2, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedMax, {1.0, 1.0}));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, MedoidRespectsFeatureWeights) {
  // One cluster, three points on a line; weights emphasize dim 0.
  // Centroid of scaled feats ≈ (2,0); nearest is row1 (value 2).
  std::vector<int64_t> labels = {0, 0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 0, 2, 0};
  std::vector<double> points = {0.0, 10.0, 2.0, 10.0, 4.0, 10.0};
  auto lab_t = to_device_tensor(labels, {3}, kInt64);
  auto idx_t = to_device_tensor(idx, {3, 2}, kInt64);
  auto pts = to_device_tensor(points, {3, 2}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kMedoid, ScoreMethod::kWeightedSum, {1.0, 0.0}));
  ASSERT_EQ(rep.shape[0], 1);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 1);
  EXPECT_EQ(r[1], 0);
}

TEST(DetectionReductionOpTest, ExcludesNoiseAndEmitsOnePerCluster) {
  std::vector<int64_t> labels = {-1, 0, -1, 1};
  std::vector<int64_t> idx = {9, 9, 0, 1, 8, 8, 2, 3};
  auto lab_t = to_device_tensor(labels, {4}, kInt64);
  auto idx_t = to_device_tensor(idx, {4, 2}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  ASSERT_EQ(rep.shape[0], 2);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 0);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 2);
  EXPECT_EQ(r[3], 3);
}

TEST(DetectionReductionOpTest, RejectsMissingRequiredPoints) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto result = run_reduction_result<float>(
      lab_t, idx_t, nullptr,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  EXPECT_FALSE(result.ok());
}

TEST(DetectionReductionOpTest, RejectsMismatchedFeatureWeights) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<double> points = {1.0, 2.0};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto pts = to_device_tensor(points, {1, 2}, real_dtype<double>());
  auto result = run_reduction_result<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DetectionReductionOpTest, RejectsCustomWithoutScoreFn) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<float> points = {1.0f};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto pts = to_device_tensor(points, {1, 1}, real_dtype<float>());
  auto result = run_reduction_result<float>(
      lab_t, idx_t, pts, make_params(RepresentativePolicy::kPeak, ScoreMethod::kCustom));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DetectionReductionOpTest, FloatDoubleAgreeOnPeak) {
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 1};
  std::vector<double> pd = {1.0, 4.0};
  std::vector<float> pf = {1.0f, 4.0f};
  auto lab_d = to_device_tensor(labels, {2}, kInt64);
  auto lab_f = to_device_tensor(labels, {2}, kInt64);
  auto idx_d = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx_f = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts_d = to_device_tensor(pd, {2, 1}, real_dtype<double>());
  auto pts_f = to_device_tensor(pf, {2, 1}, real_dtype<float>());
  const auto rd = run_reduction<double>(
      lab_d, idx_d, pts_d,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  const auto rf = run_reduction<float>(
      lab_f, idx_f, pts_f,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  EXPECT_EQ(rd.as<int64_t>()[0], rf.as<int64_t>()[0]);
  EXPECT_EQ(rd.as<int64_t>()[1], rf.as<int64_t>()[1]);
}

TEST(DetectionReductionOpTest, EmptyInputEmitsZeroClusters) {
  std::vector<int64_t> labels;
  std::vector<int64_t> idx;
  auto lab_t = to_device_tensor(labels, {0}, kInt64);
  auto idx_t = to_device_tensor(idx, {0, 2}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  EXPECT_EQ(rep.shape[0], 0);
  ASSERT_EQ(rep.shape.size(), 2u);
  EXPECT_EQ(rep.shape[1], 2);
}

TEST(DetectionReductionOpTest, AllNoiseEmitsZeroClusters) {
  std::vector<int64_t> labels = {-1, -1};
  std::vector<int64_t> idx = {1, 2, 3, 4};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  EXPECT_EQ(rep.shape[0], 0);
  ASSERT_EQ(rep.shape.size(), 2u);
  EXPECT_EQ(rep.shape[1], 2);
}

TEST(DetectionReductionOpTest, SparseLabelGapsSkipped) {
  // Labels 0 and 2 (gap at 1) → two clusters, compact order by label id.
  std::vector<int64_t> labels = {0, 2};
  std::vector<int64_t> idx = {10, 1, 30, 3};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  ASSERT_EQ(rep.shape[0], 2);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
  EXPECT_EQ(rep.as<int64_t>()[2], 30);
  EXPECT_EQ(rep.as<int64_t>()[3], 3);
}

TEST(DetectionReductionOpTest, WidenedDetectionIdxCols) {
  std::vector<int64_t> labels = {0};
  // (doppler, range, batch) — third column preserved in rep_idx.
  std::vector<int64_t> idx = {7, 8, 9};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 3}, kInt64);
  const auto rep =
      run_reduction<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  ASSERT_EQ(rep.shape[0], 1);
  ASSERT_EQ(rep.shape[1], 3);
  EXPECT_EQ(rep.as<int64_t>()[0], 7);
  EXPECT_EQ(rep.as<int64_t>()[1], 8);
  EXPECT_EQ(rep.as<int64_t>()[2], 9);
}

TEST(DetectionReductionOpTest, RejectsTooManyDetectionIdxCols) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx(11, 0);
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 11}, kInt64);
  auto result =
      run_reduction_result<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DetectionReductionOpTest, RejectsOpaqueLeadingBatch) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  auto lab_t = to_device_tensor(labels, {1, 1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto result =
      run_reduction_result<float>(lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(DetectionReductionOpTest, PeakEqualScorePicksLowestIndex) {
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {5.0, 5.0};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts = to_device_tensor(points, {2, 1}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, MedoidEqualDistancePicksLowestIndex) {
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<double> points = {0.0, 2.0};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts = to_device_tensor(points, {2, 1}, real_dtype<double>());
  const auto rep = run_reduction<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kMedoid, ScoreMethod::kWeightedSum, {1.0}));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, WeightedMeanUsesScoreAndNearestRealIndex) {
  std::vector<int64_t> labels = {0, 0, 0};
  std::vector<int64_t> idx = {0, 0, 4, 0, 10, 0};
  // Feature columns are (value, score). Row weights are 1, 2, 1.
  std::vector<float> points = {10.0F, 1.0F, 20.0F, 2.0F, 40.0F, 1.0F};
  auto lab_t = to_device_tensor(labels, {3}, kInt64);
  auto idx_t = to_device_tensor(idx, {3, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {3, 2}, real_dtype<float>());
  const auto [rep, reduced] = run_reduction_dual<float>(
      lab_t, idx_t, pts_t,
      make_params(RepresentativePolicy::kWeightedMean, ScoreMethod::kWeightedSum, {0.0, 1.0}));

  ASSERT_EQ(rep.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(rep.as<int64_t>()[0], 4);
  EXPECT_EQ(rep.as<int64_t>()[1], 0);
  ASSERT_EQ(reduced.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_NEAR(reduced.as<float>()[0], 22.5F, 1.0e-5F);
  EXPECT_NEAR(reduced.as<float>()[1], 1.5F, 1.0e-5F);
}

TEST(DetectionReductionOpTest, WeightedMeanEmptyWeightsMeansUniformMean) {
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 2, 2};
  std::vector<double> points = {2.0, 4.0, 6.0, 8.0};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {2, 2}, real_dtype<double>());
  const auto [rep, reduced] = run_reduction_dual<double>(
      lab_t, idx_t, pts_t, make_params(RepresentativePolicy::kWeightedMean));

  EXPECT_EQ(rep.as<int64_t>()[0], 0);
  EXPECT_EQ(rep.as<int64_t>()[1], 0);
  EXPECT_DOUBLE_EQ(reduced.as<double>()[0], 4.0);
  EXPECT_DOUBLE_EQ(reduced.as<double>()[1], 6.0);
}

TEST(DetectionReductionOpTest, WeightedMeanNearestUsesAllIndexColumnsAndTies) {
  std::vector<int64_t> labels = {0, 0, 0};
  std::vector<int64_t> idx = {0, 100, 2, 0, 4, 100};
  std::vector<float> points = {1.0F, 2.0F, 3.0F};
  auto lab_t = to_device_tensor(labels, {3}, kInt64);
  auto idx_t = to_device_tensor(idx, {3, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {3, 1}, real_dtype<float>());
  const auto [rep, reduced] = run_reduction_dual<float>(
      lab_t, idx_t, pts_t, make_params(RepresentativePolicy::kWeightedMean));

  // Rows 0 and 2 tie nearest the (2, 66.67) index centroid. Lowest input row
  // wins; using only the first coordinate would incorrectly choose row 1.
  EXPECT_EQ(rep.as<int64_t>()[0], 0);
  EXPECT_EQ(rep.as<int64_t>()[1], 100);
  EXPECT_FLOAT_EQ(reduced.as<float>()[0], 2.0F);
}

TEST(DetectionReductionOpTest, MemberPoliciesGatherReducedPoints) {
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {3, 4, 5, 6};
  std::vector<float> points = {1.0F, 2.0F, 7.0F, 8.0F};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {2, 2}, real_dtype<float>());
  const auto [rep, reduced] = run_reduction_dual<float>(
      lab_t, idx_t, pts_t,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 0.0}));

  EXPECT_EQ(rep.as<int64_t>()[0], 5);
  EXPECT_EQ(rep.as<int64_t>()[1], 6);
  EXPECT_FLOAT_EQ(reduced.as<float>()[0], 7.0F);
  EXPECT_FLOAT_EQ(reduced.as<float>()[1], 8.0F);
}

TEST(DetectionReductionOpTest, WeightedMeanRejectsInvalidRowWeights) {
  for (const float score : {-1.0F, 0.0F, std::numeric_limits<float>::infinity()}) {
    std::vector<int64_t> labels = {0};
    std::vector<int64_t> idx = {0, 0};
    std::vector<float> points = {1.0F, score};
    auto lab_t = to_device_tensor(labels, {1}, kInt64);
    auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
    auto pts_t = to_device_tensor(points, {1, 2}, real_dtype<float>());
    const auto result = run_reduction_result<float>(
        lab_t, idx_t, pts_t,
        make_params(RepresentativePolicy::kWeightedMean, ScoreMethod::kWeightedSum, {0.0, 1.0}));
    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(result.diagnostics().empty());
  }
}

TEST(DetectionReductionOpTest, ReducedPointsCompactSparseLabelsInOrder) {
  std::vector<int64_t> labels = {2, 0};
  std::vector<int64_t> idx = {20, 2, 10, 1};
  std::vector<float> points = {20.0F, 2.0F, 10.0F, 1.0F};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {2, 2}, real_dtype<float>());
  const auto [rep, reduced] =
      run_reduction_dual<float>(lab_t, idx_t, pts_t, make_params(RepresentativePolicy::kFirst));

  ASSERT_EQ(rep.shape[0], 2);
  ASSERT_EQ(reduced.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_FLOAT_EQ(reduced.as<float>()[0], 10.0F);
  EXPECT_FLOAT_EQ(reduced.as<float>()[1], 1.0F);
  EXPECT_FLOAT_EQ(reduced.as<float>()[2], 20.0F);
  EXPECT_FLOAT_EQ(reduced.as<float>()[3], 2.0F);
}

TEST(DetectionReductionOpTest, EmptyWeightedMeanEmitsBothZeroRowOutputs) {
  std::vector<int64_t> labels;
  std::vector<int64_t> idx;
  std::vector<float> points;
  auto lab_t = to_device_tensor(labels, {0}, kInt64);
  auto idx_t = to_device_tensor(idx, {0, 2}, kInt64);
  auto pts_t = to_device_tensor(points, {0, 2}, real_dtype<float>());
  const auto [rep, reduced] = run_reduction_dual<float>(
      lab_t, idx_t, pts_t, make_params(RepresentativePolicy::kWeightedMean));

  EXPECT_EQ(rep.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(reduced.shape, (std::vector<int64_t>{0, 2}));
}

TEST(DetectionReductionOpTest, FloatDoubleAgreeOnWeightedMean) {
  const std::vector<int64_t> labels = {0, 0};
  const std::vector<int64_t> idx = {0, 0, 4, 2};
  auto lab_f = to_device_tensor(labels, {2}, kInt64);
  auto lab_d = to_device_tensor(labels, {2}, kInt64);
  auto idx_f = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx_d = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts_f =
      to_device_tensor(std::vector<float>{2.0F, 1.0F, 6.0F, 3.0F}, {2, 2}, real_dtype<float>());
  auto pts_d =
      to_device_tensor(std::vector<double>{2.0, 1.0, 6.0, 3.0}, {2, 2}, real_dtype<double>());
  const auto [rep_f, reduced_f] = run_reduction_dual<float>(
      lab_f, idx_f, pts_f,
      make_params(RepresentativePolicy::kWeightedMean, ScoreMethod::kWeightedSum, {0.0, 1.0}));
  const auto [rep_d, reduced_d] = run_reduction_dual<double>(
      lab_d, idx_d, pts_d,
      make_params(RepresentativePolicy::kWeightedMean, ScoreMethod::kWeightedSum, {0.0, 1.0}));

  EXPECT_EQ(rep_f.as<int64_t>()[0], rep_d.as<int64_t>()[0]);
  EXPECT_EQ(rep_f.as<int64_t>()[1], rep_d.as<int64_t>()[1]);
  EXPECT_NEAR(reduced_f.as<float>()[0], reduced_d.as<double>()[0], 1.0e-6);
  EXPECT_NEAR(reduced_f.as<float>()[1], reduced_d.as<double>()[1], 1.0e-6);
}

TEST(DetectionReductionOpTest, CustomScoreMethodIgnoredForFirst) {
  // score_method is unused for first — custom must not require ScoreFn.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst, ScoreMethod::kCustom));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, RepresentativeConfigurableFromParams) {
  // Authored Params select the policy while the required input join remains
  // fixed.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 1};
  std::vector<float> points = {1.0f, 5.0f};
  auto lab0 = to_device_tensor(labels, {2}, kInt64);
  auto lab1 = to_device_tensor(labels, {2}, kInt64);
  auto idx0 = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx1 = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts0 = to_device_tensor(points, {2, 1}, real_dtype<float>());
  auto pts1 = to_device_tensor(points, {2, 1}, real_dtype<float>());
  const auto first =
      run_reduction<float>(lab0, idx0, pts0, make_params(RepresentativePolicy::kFirst));
  const auto peak = run_reduction<float>(
      lab1, idx1, pts1, make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0}));
  EXPECT_EQ(first.as<int64_t>()[0], 0);
  EXPECT_EQ(first.as<int64_t>()[1], 0);
  EXPECT_EQ(peak.as<int64_t>()[0], 1);
  EXPECT_EQ(peak.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, ScoreMethodConfigurableFromParams) {
  // Authoring Params must select weighted_sum vs weighted_l2.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<float> points = {4.0f, 3.0f, 6.0f, 0.0f};
  auto lab0 = to_device_tensor(labels, {2}, kInt64);
  auto lab1 = to_device_tensor(labels, {2}, kInt64);
  auto idx0 = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx1 = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts0 = to_device_tensor(points, {2, 2}, real_dtype<float>());
  auto pts1 = to_device_tensor(points, {2, 2}, real_dtype<float>());
  const auto sum = run_reduction<float>(
      lab0, idx0, pts0,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 1.0}));
  const auto l2 = run_reduction<float>(
      lab1, idx1, pts1,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedL2, {1.0, 1.0}));
  EXPECT_EQ(sum.as<int64_t>()[0], 10);
  EXPECT_EQ(sum.as<int64_t>()[1], 1);
  EXPECT_EQ(l2.as<int64_t>()[0], 20);
  EXPECT_EQ(l2.as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, FeatureWeightsConfigurableFromParams) {
  // Params remain the seed when no live feature_weights Tensor is expected.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  std::vector<float> points = {5.0f, 0.0f, 1.0f, 10.0f};
  auto lab0 = to_device_tensor(labels, {2}, kInt64);
  auto lab1 = to_device_tensor(labels, {2}, kInt64);
  auto idx0 = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx1 = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts0 = to_device_tensor(points, {2, 2}, real_dtype<float>());
  auto pts1 = to_device_tensor(points, {2, 2}, real_dtype<float>());
  const auto w0 = run_reduction<float>(
      lab0, idx0, pts0,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 0.0}));
  const auto w1 = run_reduction<float>(
      lab1, idx1, pts1,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {0.0, 1.0}));
  EXPECT_EQ(w0.as<int64_t>()[0], 10);
  EXPECT_EQ(w0.as<int64_t>()[1], 1);
  EXPECT_EQ(w1.as<int64_t>()[0], 20);
  EXPECT_EQ(w1.as<int64_t>()[1], 2);
}

// ---- Pageable-host vector [live] override ---------------------------------

struct LiveWeightsTick {
  std::shared_ptr<holoscan::Tensor> labels;
  std::shared_ptr<holoscan::Tensor> detection_idx;
  std::shared_ptr<holoscan::Tensor> points;
  std::optional<std::vector<double>> weights;
  int64_t skew_ns{0};
  std::optional<RepresentativePolicy> representative;
  std::optional<ScoreMethod> score_method;
};

holoscan::testing::TestResult run_weights_override_graph(std::vector<LiveWeightsTick> ticks,
                                                         DetectionReductionParams params,
                                                         int64_t max_features = 2) {
  using Op = DetectionReductionOp<double>;
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_features = max_features;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    const auto base = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    const auto override_time = std::chrono::duration_cast<std::chrono::nanoseconds>(base) +
                               std::chrono::nanoseconds{ticks[index].skew_ns};
    holoscan::testing::TestFrame frame;
    const auto append = [&frame](holoscan::testing::TestFrame part) {
      for (auto& field : part.fields) {
        frame.fields.push_back(std::move(field));
      }
    };
    if (ticks[index].weights.has_value()) {
      auto weights = holoscan::radar::testing::to_pageable_host_tensor(
          *ticks[index].weights, {static_cast<int64_t>(ticks[index].weights->size())},
          kFloat64Dtype);
      append(holoscan::testing::frame_at(override_time,
                                         holoscan::testing::input("feature_weights", *weights)));
    }
    if (ticks[index].representative.has_value()) {
      append(holoscan::testing::frame_at(
          base, holoscan::testing::input("representative", *ticks[index].representative)));
    }
    if (ticks[index].score_method.has_value()) {
      append(holoscan::testing::frame_at(
          base, holoscan::testing::input("score_method", *ticks[index].score_method)));
    }
    append(holoscan::testing::frame_at(
        base, holoscan::testing::input("points", *ticks[index].points),
        holoscan::testing::input("labels", *ticks[index].labels),
        holoscan::testing::input("detection_idx", *ticks[index].detection_idx)));
    fixture.push_frame(frame);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("rep_idx", ticks.size(), kMaximumBytes);
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.input_source_device("labels", holoscan::DeviceId{0});
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.output_device("rep_idx", holoscan::DeviceId{0});
  fixture.output_device("reduced_points", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("rep_idx", ticks.size(), std::chrono::seconds{5});
  (void)observed;
  running.request_stop();
  return running.finish();
}

std::vector<HostTensor> captured_rep_idx(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>("rep_idx")) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

LiveWeightsTick make_weights_tick(std::optional<std::vector<double>> weights, int64_t skew_ns = 0) {
  const std::vector<int64_t> labels = {0, 0};
  const std::vector<int64_t> idx = {10, 1, 20, 2};
  const std::vector<double> points = {5.0, 0.0, 1.0, 10.0};
  return {
      to_device_tensor(labels, {2}, kInt64),
      to_device_tensor(idx, {2, 2}, kInt64),
      to_device_tensor(points, {2, 2}, real_dtype<double>()),
      std::move(weights),
      skew_ns,
  };
}

DetectionReductionParams live_weights_params() {
  return make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 0.0});
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsExactMatchChangesSelection) {
  const auto out = run_weights_override_graph({make_weights_tick(std::vector<double>{0.0, 1.0})},
                                              live_weights_params());

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 1U);
  EXPECT_EQ(frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(frames[0].as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsSkewedCaptureMissesExact) {
  const auto out = run_weights_override_graph({make_weights_tick(std::vector<double>{0.0, 1.0}, 1)},
                                              live_weights_params());

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 1U);
  EXPECT_EQ(frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(frames[0].as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsRevertsToParamsAfterMiss) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{0.0, 1.0}), make_weights_tick(std::nullopt)},
      live_weights_params());

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 2U);
  EXPECT_EQ(frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(frames[1].as<int64_t>()[0], 10);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsRejectsMalformedLength) {
  const auto out = run_weights_override_graph({make_weights_tick(std::vector<double>{1.0})},
                                              live_weights_params());

  EXPECT_FALSE(out.ok());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsRejectsBoundOverflow) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{1.0, 1.0, 1.0})}, live_weights_params(), 2);

  EXPECT_FALSE(out.ok());
  EXPECT_FALSE(out.diagnostics().empty());
}

// ---- Enum [live] overrides -----------------------------------------------

// Points chosen so the two policies disagree: with w = (1, 0) the first row is
// the lower-scoring one, so `first` keeps row 0 (10, 1) and `peak` keeps
// row 1 (20, 2).
LiveWeightsTick make_representative_tick(std::optional<RepresentativePolicy> representative) {
  const std::vector<int64_t> labels = {0, 0};
  const std::vector<int64_t> idx = {10, 1, 20, 2};
  const std::vector<double> points = {1.0, 0.0, 5.0, 10.0};
  LiveWeightsTick tick{
      to_device_tensor(labels, {2}, kInt64),
      to_device_tensor(idx, {2, 2}, kInt64),
      to_device_tensor(points, {2, 2}, real_dtype<double>()),
      std::nullopt,
      0,
  };
  tick.representative = representative;
  return tick;
}

// With w = (1, 1), weighted_sum ranks row 0 above row 1 (7 vs 6) while
// weighted_l2 reverses it (5 vs 6), so the peak row names the score method.
LiveWeightsTick make_score_method_tick(std::optional<ScoreMethod> score_method) {
  const std::vector<int64_t> labels = {0, 0};
  const std::vector<int64_t> idx = {10, 1, 20, 2};
  const std::vector<double> points = {4.0, 3.0, 6.0, 0.0};
  LiveWeightsTick tick{
      to_device_tensor(labels, {2}, kInt64),
      to_device_tensor(idx, {2, 2}, kInt64),
      to_device_tensor(points, {2, 2}, real_dtype<double>()),
      std::nullopt,
      0,
  };
  tick.score_method = score_method;
  return tick;
}

TEST(DetectionReductionOpTest, LiveRepresentativeAppliesForOneTick) {
  auto params = make_params(RepresentativePolicy::kFirst, ScoreMethod::kWeightedSum, {1.0, 0.0});
  const auto out =
      run_weights_override_graph({make_representative_tick(std::nullopt),
                                  make_representative_tick(RepresentativePolicy::kPeak),
                                  make_representative_tick(std::nullopt)},
                                 std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 3U);
  // Params `first` on the silent ticks either side of the `peak` override.
  EXPECT_EQ(frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(frames[1].as<int64_t>()[0], 20);
  EXPECT_EQ(frames[2].as<int64_t>()[0], 10);
}

TEST(DetectionReductionOpTest, LiveWeightedMeanAppliesForOneTick) {
  auto params = make_params(RepresentativePolicy::kFirst, ScoreMethod::kWeightedSum, {1.0, 0.0});
  const auto out =
      run_weights_override_graph({make_representative_tick(std::nullopt),
                                  make_representative_tick(RepresentativePolicy::kWeightedMean),
                                  make_representative_tick(std::nullopt)},
                                 std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 3U);
  EXPECT_EQ(frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(frames[1].as<int64_t>()[0], 20);
  EXPECT_EQ(frames[2].as<int64_t>()[0], 10);
}

TEST(DetectionReductionOpTest, LiveScoreMethodAppliesForOneTick) {
  auto params = make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum, {1.0, 1.0});
  const auto out = run_weights_override_graph(
      {make_score_method_tick(std::nullopt), make_score_method_tick(ScoreMethod::kWeightedL2),
       make_score_method_tick(std::nullopt)},
      std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto frames = captured_rep_idx(out);
  ASSERT_EQ(frames.size(), 3U);
  EXPECT_EQ(frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(frames[1].as<int64_t>()[0], 20);
  EXPECT_EQ(frames[2].as<int64_t>()[0], 10);
}

}  // namespace
