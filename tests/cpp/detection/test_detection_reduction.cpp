// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DetectionReductionOp contract tests (HSDK 5 Graph harness): group-by-label
// → rep_idx — first/peak/medoid policies, score methods, noise exclusion,
// Params coverage, error paths, and float/double agreement. Graph mid-stream
// sticky [live] overrides deferred (HSDK5_PORT.md §1 / §9).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::DetectionReductionOp;
using holoscan::radar::DetectionReductionParams;
using holoscan::radar::RepresentativePolicy;
using holoscan::radar::ScoreMethod;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_triple_in_single_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;

constexpr DLDataType kInt64{kDLInt, 64, 1};

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

template <typename Source, typename OpRef, typename Sink>
void wire_reduction_with_points(holoscan::Graph& graph, const Source& source,
                                OpRef& op, const Sink& sink) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out0, op->labels, q);
  graph.add_flow(source->out1, op->detection_idx, q);
  graph.add_flow(source->out2, op->points, q);
  graph.add_flow(op->rep_idx, sink->in, q);
}

template <typename T>
GraphRunResult run_reduction_result(
    std::shared_ptr<holoscan::Tensor> labels,
    std::shared_ptr<holoscan::Tensor> detection_idx,
    std::shared_ptr<holoscan::Tensor> points,
    DetectionReductionParams params) {
  // [EA HSDK5 ONLY] points is Exact in the temporal contract — always wire
  // (dummy zeros for first-only). See DetectionReduction OPEN_ITEMS.
  if (!points) {
    const int64_t n = labels->shape()[0];
    std::vector<T> zeros(static_cast<size_t>(n), T{0});
    points = to_device_tensor(zeros, {n, 1}, real_dtype<T>());
  }
  return run_triple_in_single_out_operator_graph_result<
      DetectionReductionOp<T>>(
      {std::move(labels)}, {std::move(detection_idx)}, {std::move(points)},
      std::move(params), "rep_idx",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) {
        wire_reduction_with_points(graph, source, op, sink);
      });
}

template <typename T>
HostTensor run_reduction(std::shared_ptr<holoscan::Tensor> labels,
                         std::shared_ptr<holoscan::Tensor> detection_idx,
                         std::shared_ptr<holoscan::Tensor> points,
                         DetectionReductionParams params) {
  GraphRunResult result = run_reduction_result<T>(
      std::move(labels), std::move(detection_idx), std::move(points),
      std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "DetectionReductionOp test: expected one rep_idx frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
}

// ---------------------------------------------------------------------------

TEST(DetectionReductionOpTest, FirstSelectsLowestIndexPerCluster) {
  // rows: 0→cluster0, 1→cluster1, 2→cluster0, 3→noise
  std::vector<int64_t> labels = {0, 1, 0, -1};
  std::vector<int64_t> idx = {10, 1, 20, 2, 11, 3, 99, 9};
  auto lab_t = to_device_tensor(labels, {4}, kInt64);
  auto idx_t = to_device_tensor(idx, {4, 2}, kInt64);
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0, 0.0}));
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedL2,
                  {1.0, 1.0}));
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedMax,
                  {1.0, 1.0}));
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
      make_params(RepresentativePolicy::kMedoid, ScoreMethod::kWeightedSum,
                  {1.0, 0.0}));
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
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  ASSERT_EQ(rep.shape[0], 2);
  const int64_t* r = rep.as<int64_t>();
  EXPECT_EQ(r[0], 0);
  EXPECT_EQ(r[1], 1);
  EXPECT_EQ(r[2], 2);
  EXPECT_EQ(r[3], 3);
}

TEST(DetectionReductionOpTest, RejectsPeakWithoutPoints) {
  // [EA HSDK5 ONLY] points is Exact in the temporal contract on EA, so an
  // unconnected points port fails compile (TEMPORAL_INPUT_UNCONNECTED) rather
  // than reaching the runtime "points required for peak" check. Same family
  // as FeatureAssembly optional gathers (OPEN_ITEMS). Runtime validation is
  // retained in compute() for when the optional-Tensor recipe lands.
  GTEST_SKIP() << "HSDK 5 EA: unconnected optional points deferred "
                  "(DetectionReduction OPEN_ITEMS — points Exact in contract)";
}

TEST(DetectionReductionOpTest, RejectsMismatchedFeatureWeights) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<double> points = {1.0, 2.0};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto pts = to_device_tensor(points, {1, 2}, real_dtype<double>());
  GraphRunResult result = run_reduction_result<double>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(DetectionReductionOpTest, RejectsCustomWithoutScoreFn) {
  std::vector<int64_t> labels = {0};
  std::vector<int64_t> idx = {0, 0};
  std::vector<float> points = {1.0f};
  auto lab_t = to_device_tensor(labels, {1}, kInt64);
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64);
  auto pts = to_device_tensor(points, {1, 1}, real_dtype<float>());
  GraphRunResult result = run_reduction_result<float>(
      lab_t, idx_t, pts,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kCustom));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0}));
  const auto rf = run_reduction<float>(
      lab_f, idx_f, pts_f,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0}));
  EXPECT_EQ(rd.as<int64_t>()[0], rf.as<int64_t>()[0]);
  EXPECT_EQ(rd.as<int64_t>()[1], rf.as<int64_t>()[1]);
}

TEST(DetectionReductionOpTest, EmptyInputEmitsZeroClusters) {
  std::vector<int64_t> labels;
  std::vector<int64_t> idx;
  auto lab_t = to_device_tensor(labels, {0}, kInt64);
  auto idx_t = to_device_tensor(idx, {0, 2}, kInt64);
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  EXPECT_EQ(rep.shape[0], 0);
  ASSERT_EQ(rep.shape.size(), 2u);
  EXPECT_EQ(rep.shape[1], 2);
}

TEST(DetectionReductionOpTest, AllNoiseEmitsZeroClusters) {
  std::vector<int64_t> labels = {-1, -1};
  std::vector<int64_t> idx = {1, 2, 3, 4};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
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
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
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
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr, make_params(RepresentativePolicy::kFirst));
  ASSERT_EQ(rep.shape[0], 1);
  ASSERT_EQ(rep.shape[1], 3);
  EXPECT_EQ(rep.as<int64_t>()[0], 7);
  EXPECT_EQ(rep.as<int64_t>()[1], 8);
  EXPECT_EQ(rep.as<int64_t>()[2], 9);
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0}));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, CustomScoreMethodIgnoredForFirst) {
  // score_method is unused for first — custom must not require ScoreFn.
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {10, 1, 20, 2};
  auto lab_t = to_device_tensor(labels, {2}, kInt64);
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64);
  const auto rep = run_reduction<float>(
      lab_t, idx_t, nullptr,
      make_params(RepresentativePolicy::kFirst, ScoreMethod::kCustom));
  ASSERT_EQ(rep.shape[0], 1);
  EXPECT_EQ(rep.as<int64_t>()[0], 10);
  EXPECT_EQ(rep.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, RepresentativeConfigurableFromParams) {
  // Authoring Params must select first vs peak (live Graph override deferred
  // — HSDK5_PORT §1).
  std::vector<int64_t> labels = {0, 0};
  std::vector<int64_t> idx = {0, 0, 1, 1};
  std::vector<float> points = {1.0f, 5.0f};
  auto lab0 = to_device_tensor(labels, {2}, kInt64);
  auto lab1 = to_device_tensor(labels, {2}, kInt64);
  auto idx0 = to_device_tensor(idx, {2, 2}, kInt64);
  auto idx1 = to_device_tensor(idx, {2, 2}, kInt64);
  auto pts0 = to_device_tensor(points, {2, 1}, real_dtype<float>());
  auto pts1 = to_device_tensor(points, {2, 1}, real_dtype<float>());
  const auto first = run_reduction<float>(
      lab0, idx0, pts0, make_params(RepresentativePolicy::kFirst));
  const auto peak = run_reduction<float>(
      lab1, idx1, pts1,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0}));
  EXPECT_EQ(first.as<int64_t>()[0], 0);
  EXPECT_EQ(first.as<int64_t>()[1], 0);
  EXPECT_EQ(peak.as<int64_t>()[0], 1);
  EXPECT_EQ(peak.as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, ScoreMethodConfigurableFromParams) {
  // Authoring Params must select weighted_sum vs weighted_l2 (live Graph
  // override deferred — HSDK5_PORT §1).
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0, 1.0}));
  const auto l2 = run_reduction<float>(
      lab1, idx1, pts1,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedL2,
                  {1.0, 1.0}));
  EXPECT_EQ(sum.as<int64_t>()[0], 10);
  EXPECT_EQ(sum.as<int64_t>()[1], 1);
  EXPECT_EQ(l2.as<int64_t>()[0], 20);
  EXPECT_EQ(l2.as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, FeatureWeightsConfigurableFromParams) {
  // Authoring Params must select feature_weights (live Graph override
  // deferred — HSDK5_PORT §9).
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
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {1.0, 0.0}));
  const auto w1 = run_reduction<float>(
      lab1, idx1, pts1,
      make_params(RepresentativePolicy::kPeak, ScoreMethod::kWeightedSum,
                  {0.0, 1.0}));
  EXPECT_EQ(w0.as<int64_t>()[0], 10);
  EXPECT_EQ(w0.as<int64_t>()[1], 1);
  EXPECT_EQ(w1.as<int64_t>()[0], 20);
  EXPECT_EQ(w1.as<int64_t>()[1], 2);
}

// ---- Sticky [live] overrides (deferred on EA) -----------------------------

TEST(DetectionReductionOpTest, LiveRepresentativeStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 — Graph mid-stream
  // sticky override harness fails on EA device/partition binding. Operator
  // still declares Input<RepresentativePolicy> + sticky resolve; Params cover
  // selection (RepresentativeConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live representative override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

TEST(DetectionReductionOpTest, LiveScoreMethodStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1.
  GTEST_SKIP() << "HSDK 5 EA: live score_method override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsStickyOverride) {
  // [EA HSDK5 ONLY] feature_weights is Params-only: unbounded length-F vector
  // cannot use Input<std::vector> (HSDK5_PORT.md §9). Params authoring covers
  // selection (FeatureWeightsConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live feature_weights override deferred "
                  "(HSDK5_PORT.md §9 — Params-only unbounded vector)";
}

}  // namespace
