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
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "radar_common/sdk/tensor_io.hpp"
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

struct LiveWeightsBank {
  static void reset(std::vector<LiveWeightsTick> new_ticks) {
    ticks = std::move(new_ticks);
    index = 0U;
  }
  static inline std::vector<LiveWeightsTick> ticks;
  static inline std::size_t index{0U};
};

class LiveWeightsSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    constexpr holoscan::TensorRepresentation kHostVector{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = kFloat64Dtype,
        .rank = 1U,
    };
    const holoscan::TensorBounds device_bounds{
        .max_rank = 2U, .max_byte_span = 4096U};
    spec.output(labels, "out0")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, device_bounds));
    spec.output(detection_idx, "out1")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, device_bounds));
    spec.output(points, "out2")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, device_bounds));
    spec.output(weights, "feature_weights")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostVector, 1U, 128U * sizeof(double)));
    spec.output(representative, "representative").max_emits_per_compute(1U);
    spec.output(score_method, "score_method").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveWeightsBank::index >= LiveWeightsBank::ticks.size()) {
      return {};
    }
    const LiveWeightsTick tick =
        LiveWeightsBank::ticks[LiveWeightsBank::index++];
    const holoscan::TimePoint capture_time = context.activation_time();
    if (tick.weights.has_value()) {
      const std::vector<double>& values = *tick.weights;
      const std::vector<int64_t> shape{
          static_cast<int64_t>(values.size())};
      auto loan = weights.allocate_tensor(holoscan::TensorLoanRequest{
          .shape = shape, .dtype = kFloat64Dtype});
      if (!loan) {
        return holoscan::make_unexpected(std::move(loan).error());
      }
      auto written = loan->write_host_and_commit(
          [&values](const holoscan::TensorOutputWriteGuard& writer) {
            std::copy(values.begin(), values.end(),
                      writer.data_as<double>());
          });
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      holoscan::TimePoint weights_time = capture_time;
      weights_time.timestamp_ns += tick.skew_ns;
      auto emitted = weights.emit(
          std::move(*loan),
          holoscan::EmitOptions{.capture_time = weights_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    if (tick.representative.has_value()) {
      auto emitted = representative.emit(
          *tick.representative,
          holoscan::EmitOptions{.capture_time = capture_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    if (tick.score_method.has_value()) {
      auto emitted = score_method.emit(
          *tick.score_method,
          holoscan::EmitOptions{.capture_time = capture_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    auto emitted = holoscan::radar::testing::detail::emit_bank_frame(
        labels, context, tick.labels, "LiveWeightsSourceOp.labels");
    if (!emitted) {
      return emitted;
    }
    emitted = holoscan::radar::testing::detail::emit_bank_frame(
        detection_idx, context, tick.detection_idx,
        "LiveWeightsSourceOp.detection_idx");
    if (!emitted) {
      return emitted;
    }
    return holoscan::radar::testing::detail::emit_bank_frame(
        points, context, tick.points, "LiveWeightsSourceOp.points");
  }

  holoscan::Output<holoscan::Tensor> labels{"out0"};
  holoscan::Output<holoscan::Tensor> detection_idx{"out1"};
  holoscan::Output<holoscan::Tensor> points{"out2"};
  holoscan::Output<holoscan::Tensor> weights{"feature_weights"};
  holoscan::Output<RepresentativePolicy> representative{"representative"};
  holoscan::Output<ScoreMethod> score_method{"score_method"};
};

GraphRunResult run_weights_override_graph(
    std::vector<LiveWeightsTick> ticks, DetectionReductionParams params,
    int64_t max_features = 2) {
  GraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  const std::size_t expected = ticks.size();
  LiveWeightsBank::reset(std::move(ticks));

  holoscan::Graph graph{"feature-weights-live-override"};
  const auto source = graph.op<LiveWeightsSourceOp>("source");
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_features = max_features;
  auto op = graph.op<DetectionReductionOp<double>>("op", extents);
  op.set_params(std::move(params));
  const auto sink =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("sink");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->labels, op->labels, q);
  graph.add_flow(source->detection_idx, op->detection_idx, q);
  graph.add_flow(source->points, op->points, q);
  // All three override ports stay wired for every case here: the source only
  // emits the ones a tick carries, so a case that never exercises a port is
  // also the check that a silent enabled port cannot stall the OnAligned
  // trigger.
  graph.add_flow(source->weights, op->feature_weights, q);
  graph.add_flow(source->representative, op->representative, q);
  graph.add_flow(source->score_method, op->score_method, q);
  graph.add_flow(op->rep_idx, sink->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  holoscan::radar::testing::detail::bind_triple_in_single_out_device_pools(
      options, "rep_idx");
  try {
    const holoscan::ExecutionPlan plan =
        holoscan::compile(graph, std::move(options));
    result.plan_ok = plan.ok();
    result.plan_json = plan.json();
    if (!plan.ok()) {
      return result;
    }
    holoscan::RunSession session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() +
                          holoscan::radar::testing::detail::kHarnessDeadline;
    while (session.diagnostics().empty() &&
           std::chrono::steady_clock::now() < deadline) {
      {
        const std::lock_guard lock(
            holoscan::radar::testing::detail::FrameBank::mutex);
        if (holoscan::radar::testing::detail::FrameBank::captured.size() >=
            expected) {
          break;
        }
      }
      std::this_thread::sleep_for(
          holoscan::radar::testing::detail::kHarnessPoll);
    }
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
      result.diagnostics.emplace_back(diagnostic.token);
    }
  } catch (const holoscan::RuntimeError& error) {
    result.plan_ok = false;
    result.plan_json = std::string("RuntimeError: ") + error.what();
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBank::mutex);
    result.frames =
        std::move(holoscan::radar::testing::detail::FrameBank::captured);
  }
  return result;
}

LiveWeightsTick make_weights_tick(
    std::optional<std::vector<double>> weights, int64_t skew_ns = 0) {
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

DetectionReductionParams live_weights_params(bool sticky) {
  auto params = make_params(RepresentativePolicy::kPeak,
                            ScoreMethod::kWeightedSum, {1.0, 0.0});
  params.sticky_feature_weights = sticky;
  return params;
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsExactMatchChangesSelection) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{0.0, 1.0})},
      live_weights_params(false));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 1U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[0].as<int64_t>()[1], 2);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsSkewedCaptureMissesExact) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{0.0, 1.0}, 1)},
      live_weights_params(false));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 1U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(out.frames[0].as<int64_t>()[1], 1);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsStickyKeepsLastExactMatch) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{0.0, 1.0}),
       make_weights_tick(std::nullopt)},
      live_weights_params(true));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[1].as<int64_t>()[0], 20);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsEphemeralRevertsToParams) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{0.0, 1.0}),
       make_weights_tick(std::nullopt)},
      live_weights_params(false));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[1].as<int64_t>()[0], 10);
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsRejectsMalformedLength) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{1.0})},
      live_weights_params(true));

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(DetectionReductionOpTest, LiveFeatureWeightsRejectsBoundOverflow) {
  const auto out = run_weights_override_graph(
      {make_weights_tick(std::vector<double>{1.0, 1.0, 1.0})},
      live_weights_params(true), 2);

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_FALSE(out.diagnostics.empty());
}

// ---- Enum [live] overrides -----------------------------------------------

// Points chosen so the two policies disagree: with w = (1, 0) the first row is
// the lower-scoring one, so `first` keeps row 0 (10, 1) and `peak` keeps
// row 1 (20, 2).
LiveWeightsTick make_representative_tick(
    std::optional<RepresentativePolicy> representative) {
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
LiveWeightsTick make_score_method_tick(
    std::optional<ScoreMethod> score_method) {
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

TEST(DetectionReductionOpTest, LiveRepresentativeStickyOverride) {
  auto params = make_params(RepresentativePolicy::kFirst,
                            ScoreMethod::kWeightedSum, {1.0, 0.0});
  params.sticky_representative = true;
  const auto out = run_weights_override_graph(
      {make_representative_tick(std::nullopt),
       make_representative_tick(RepresentativePolicy::kPeak),
       make_representative_tick(std::nullopt)},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 3U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(out.frames[1].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[2].as<int64_t>()[0], 20);
}

TEST(DetectionReductionOpTest, LiveRepresentativeEphemeralRevertsToParams) {
  auto params = make_params(RepresentativePolicy::kFirst,
                            ScoreMethod::kWeightedSum, {1.0, 0.0});
  params.sticky_representative = false;
  const auto out = run_weights_override_graph(
      {make_representative_tick(RepresentativePolicy::kPeak),
       make_representative_tick(std::nullopt)},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 2U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[1].as<int64_t>()[0], 10);
}

TEST(DetectionReductionOpTest, LiveScoreMethodStickyOverride) {
  auto params = make_params(RepresentativePolicy::kPeak,
                            ScoreMethod::kWeightedSum, {1.0, 1.0});
  params.sticky_score_method = true;
  const auto out = run_weights_override_graph(
      {make_score_method_tick(std::nullopt),
       make_score_method_tick(ScoreMethod::kWeightedL2),
       make_score_method_tick(std::nullopt)},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), 3U);
  EXPECT_EQ(out.frames[0].as<int64_t>()[0], 10);
  EXPECT_EQ(out.frames[1].as<int64_t>()[0], 20);
  EXPECT_EQ(out.frames[2].as<int64_t>()[0], 20);
}

}  // namespace
