// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ClusterDBSCANOp contract tests (HSDK 5 Graph harness): density clustering
// over a feature matrix — blobs/noise, anisotropic epsilon, min_points /
// max_clusters Params coverage, descending-size numbering, error paths, and
// float/double agreement and pageable-host epsilon live overrides.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
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

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClusterDBSCANOp;
using holoscan::radar::ClusterDBSCANParams;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;

constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};

struct DbscanResult {
  HostTensor labels;
  HostTensor sizes;
};

template <typename T>
std::shared_ptr<holoscan::Tensor> points_tensor(const std::vector<double>& host,
                                                int64_t rows, int64_t cols) {
  return real_cube_to_device<T>(host, {rows, cols});
}

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_dbscan(holoscan::Graph& graph, const Source& source, OpRef& op,
                 const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->points, q);
  graph.add_flow(op->labels, sink0->in, q);
  graph.add_flow(op->sizes, sink1->in, q);
}

template <typename T>
DualIOGraphRunResult run_dbscan_result(std::shared_ptr<holoscan::Tensor> points,
                                       ClusterDBSCANParams params) {
  return run_single_in_dual_out_operator_graph_result<ClusterDBSCANOp<T>>(
      {std::move(points)}, std::move(params), "labels", "sizes",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_dbscan(graph, source, op, sink0, sink1);
      });
}

template <typename T>
DbscanResult run_dbscan(std::shared_ptr<holoscan::Tensor> points,
                        ClusterDBSCANParams params) {
  DualIOGraphRunResult result =
      run_dbscan_result<T>(std::move(points), std::move(params));
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
  if (result.frames0.size() != 1U || result.frames1.size() != 1U) {
    throw std::runtime_error(
        "ClusterDBSCANOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  return {std::move(result.frames0.front()),
          std::move(result.frames1.front())};
}

ClusterDBSCANParams make_params(std::vector<double> epsilon,
                                int64_t min_points = 0,
                                int64_t max_clusters = 0) {
  ClusterDBSCANParams params{};
  params.epsilon = std::move(epsilon);
  params.min_points = min_points;
  params.max_clusters = max_clusters;
  return params;
}

// ---------------------------------------------------------------------------

// sizes is a latched optional output: an application may consume the labels
// alone. It used to be emitted under HOLOSCAN_TRY, so declining it killed the
// run on the first tick. Now the first failed emit warns and disables the
// port, which also skips the per-cluster histogram that runs solely to fill
// it — the earlier histogram feeding label renumbering must survive, so this
// checks the labels stay correct on every later tick.
// See OPERATOR_API.md, "Optional output convention".
TEST(ClusterDBSCANOpTest, UnwiredSizesDoesNotFailTheRunOrDisturbLabels) {
  constexpr std::size_t kFrames = 8;
  const std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,   // blob A (4)
      5.0, 5.0,                                    // outlier
      10.0, 10.0, 10.1, 10.0, 10.0, 10.1,          // blob B (3)
  };
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  frames.reserve(kFrames);
  for (std::size_t i = 0; i < kFrames; ++i) {
    frames.push_back(points_tensor<double>(pts, 8, 2));
  }

  const holoscan::radar::testing::GraphRunResult out =
      holoscan::radar::testing::run_single_io_operator_graph_result<
          ClusterDBSCANOp<double>>(
          std::move(frames), make_params({0.5}, 3), "labels",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out, op->points, q);
            graph.add_flow(op->labels, sink->in, q);
          },
          {"sizes"});

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames.size(), kFrames);
  for (const HostTensor& frame : out.frames) {
    ASSERT_EQ(frame.shape, (std::vector<int64_t>{8}));
    const int64_t* lab = frame.as<int64_t>();
    EXPECT_EQ(lab[0], 0);
    EXPECT_EQ(lab[4], -1);  // noise
    EXPECT_EQ(lab[5], 1);
  }
}

TEST(ClusterDBSCANOpTest, TwoBlobsAndNoise) {
  // Rows 0-3: blob A near (0,0). Row 4: isolated outlier. Rows 5-7: blob B
  // near (10,10). epsilon=0.5 isotropic, min_points=3.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,   // blob A (4)
      5.0, 5.0,                                    // outlier
      10.0, 10.0, 10.1, 10.0, 10.0, 10.1,          // blob B (3)
  };
  auto p = points_tensor<double>(pts, 8, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 3));

  ASSERT_EQ(out.labels.shape.size(), 1u);
  ASSERT_EQ(out.labels.shape[0], 8);
  const int64_t* lab = out.labels.as<int64_t>();
  // Blob A is the larger cluster (4 > 3) → dense id 0; blob B → id 1.
  EXPECT_EQ(lab[0], 0);
  EXPECT_EQ(lab[1], 0);
  EXPECT_EQ(lab[2], 0);
  EXPECT_EQ(lab[3], 0);
  EXPECT_EQ(lab[4], -1);  // noise
  EXPECT_EQ(lab[5], 1);
  EXPECT_EQ(lab[6], 1);
  EXPECT_EQ(lab[7], 1);

  ASSERT_EQ(out.sizes.shape.size(), 1u);
  ASSERT_EQ(out.sizes.shape[0], 2);
  const int64_t* sz = out.sizes.as<int64_t>();
  EXPECT_EQ(sz[0], 4);
  EXPECT_EQ(sz[1], 3);
}

TEST(ClusterDBSCANOpTest, AnisotropicPerAxisEpsilon) {
  // A chain wide in x, plus a point far in y. Per-axis epsilon (x=3, y=1)
  // links the chain but rejects the y-separated point; an isotropic radius of
  // 1 would not link the chain at all.
  std::vector<double> pts = {
      0.0, 0.0,  2.0, 0.0,  4.0, 0.0,  // x-chain
      0.0, 5.0,                        // far in y → noise
  };
  auto p = points_tensor<double>(pts, 4, 2);
  const auto out = run_dbscan<double>(p, make_params({3.0, 1.0}, 2));

  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], 0);
  EXPECT_EQ(lab[1], 0);
  EXPECT_EQ(lab[2], 0);
  EXPECT_EQ(lab[3], -1);
  ASSERT_EQ(out.sizes.shape[0], 1);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
}

TEST(ClusterDBSCANOpTest, MinPointsSeparatesCoreFromNoise) {
  // Group 1: a triple; group 2: a pair. epsilon=0.5.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  0.2, 0.0,   // triple
      10.0, 0.0, 10.1, 0.0,             // pair
  };

  // min_points=2 → both groups are clusters (no noise).
  {
    auto p = points_tensor<double>(pts, 5, 2);
    const auto out = run_dbscan<double>(p, make_params({0.5}, 2));
    const int64_t* lab = out.labels.as<int64_t>();
    EXPECT_EQ(lab[0], 0);
    EXPECT_EQ(lab[1], 0);
    EXPECT_EQ(lab[2], 0);
    EXPECT_EQ(lab[3], 1);
    EXPECT_EQ(lab[4], 1);
    ASSERT_EQ(out.sizes.shape[0], 2);
    EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
    EXPECT_EQ(out.sizes.as<int64_t>()[1], 2);
  }

  // min_points=3 → the pair cannot reach 3 neighbors → both noise.
  {
    auto p = points_tensor<double>(pts, 5, 2);
    const auto out = run_dbscan<double>(p, make_params({0.5}, 3));
    const int64_t* lab = out.labels.as<int64_t>();
    EXPECT_EQ(lab[0], 0);
    EXPECT_EQ(lab[1], 0);
    EXPECT_EQ(lab[2], 0);
    EXPECT_EQ(lab[3], -1);
    EXPECT_EQ(lab[4], -1);
    ASSERT_EQ(out.sizes.shape[0], 1);
    EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
  }
}

TEST(ClusterDBSCANOpTest, DefaultMinPointsIsTwiceNumFeatures) {
  // num_features=2 → default min_points=4. A triple cannot form a cluster.
  std::vector<double> pts = {0.0, 0.0, 0.1, 0.0, 0.2, 0.0};
  auto p = points_tensor<double>(pts, 3, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], -1);
  EXPECT_EQ(lab[1], -1);
  EXPECT_EQ(lab[2], -1);
  EXPECT_EQ(out.sizes.shape[0], 0);
}

TEST(ClusterDBSCANOpTest, MaxClustersCapsAndDropsSurplus) {
  // Three separated pairs → three clusters; cap to 2 drops the third to noise.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,    // cluster 0
      10.0, 0.0, 10.1, 0.0,   // cluster 1
      20.0, 0.0, 20.1, 0.0,   // cluster 2 (dropped)
  };
  auto p = points_tensor<double>(pts, 6, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2, 2));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], 0);
  EXPECT_EQ(lab[1], 0);
  EXPECT_EQ(lab[2], 1);
  EXPECT_EQ(lab[3], 1);
  EXPECT_EQ(lab[4], -1);  // surplus cluster relabeled to noise
  EXPECT_EQ(lab[5], -1);
  ASSERT_EQ(out.sizes.shape[0], 2);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 2);
  EXPECT_EQ(out.sizes.as<int64_t>()[1], 2);
}

TEST(ClusterDBSCANOpTest, NumbersClustersByDescendingSize) {
  // A smaller pair at early rows, a larger triple at later rows. Ascending-root
  // numbering would label the pair 0 (it owns the lowest row); descending-size
  // numbering labels the larger triple 0. `sizes` is therefore descending.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,               // pair (rows 0-1) — smaller, root 0
      10.0, 0.0, 10.1, 0.0, 10.2, 0.0,   // triple (rows 2-4) — larger, root 2
  };
  auto p = points_tensor<double>(pts, 5, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], 1);  // pair → id 1 (smaller)
  EXPECT_EQ(lab[1], 1);
  EXPECT_EQ(lab[2], 0);  // triple → id 0 (larger)
  EXPECT_EQ(lab[3], 0);
  EXPECT_EQ(lab[4], 0);
  ASSERT_EQ(out.sizes.shape[0], 2);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);  // largest first
  EXPECT_EQ(out.sizes.as<int64_t>()[1], 2);
}

TEST(ClusterDBSCANOpTest, MaxClustersKeepsLargest) {
  // Same smaller-pair-then-larger-triple layout; cap to 1 must keep the larger
  // cluster (the triple) and drop the pair to noise — a spatially-arbitrary
  // ascending-root cap would instead have kept the pair.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,               // pair (rows 0-1)
      10.0, 0.0, 10.1, 0.0, 10.2, 0.0,   // triple (rows 2-4)
  };
  auto p = points_tensor<double>(pts, 5, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2, 1));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], -1);  // pair dropped to noise
  EXPECT_EQ(lab[1], -1);
  EXPECT_EQ(lab[2], 0);   // triple kept
  EXPECT_EQ(lab[3], 0);
  EXPECT_EQ(lab[4], 0);
  ASSERT_EQ(out.sizes.shape[0], 1);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
}

TEST(ClusterDBSCANOpTest, EmptyPointsEmitsEmptyOutputs) {
  std::vector<double> pts;
  auto p = points_tensor<double>(pts, 0, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2));
  ASSERT_EQ(out.labels.shape.size(), 1u);
  EXPECT_EQ(out.labels.shape[0], 0);
  ASSERT_EQ(out.sizes.shape.size(), 1u);
  EXPECT_EQ(out.sizes.shape[0], 0);
}

TEST(ClusterDBSCANOpTest, FloatAndDoubleAgree) {
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
      10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
      5.0, 5.0,
  };
  auto pf = points_tensor<float>(pts, 7, 2);
  auto pd = points_tensor<double>(pts, 7, 2);
  const auto of = run_dbscan<float>(pf, make_params({0.5}, 3));
  const auto od = run_dbscan<double>(pd, make_params({0.5}, 3));
  ASSERT_EQ(of.labels.shape[0], od.labels.shape[0]);
  for (int64_t i = 0; i < of.labels.shape[0]; ++i) {
    EXPECT_EQ(of.labels.as<int64_t>()[i], od.labels.as<int64_t>()[i]) << "row "
                                                                      << i;
  }
  ASSERT_EQ(of.sizes.shape[0], od.sizes.shape[0]);
}

TEST(ClusterDBSCANOpTest, EpsilonConfigurableFromParams) {
  // Params remain the seed when no live epsilon Tensor is expected.
  std::vector<double> pts = {
      0.0, 0.0,  2.0, 0.0,  4.0, 0.0,  0.0, 5.0,
  };
  auto p_iso = points_tensor<double>(pts, 4, 2);
  auto p_ani = points_tensor<double>(pts, 4, 2);
  const auto iso = run_dbscan<double>(p_iso, make_params({1.0}, 2));
  const auto ani = run_dbscan<double>(p_ani, make_params({3.0, 1.0}, 2));
  // Isotropic radius 1: x-chain does not link → all noise (or tiny clusters).
  EXPECT_EQ(iso.sizes.shape[0], 0);
  // Anisotropic (3,1): x-chain links → one cluster of 3.
  ASSERT_EQ(ani.sizes.shape[0], 1);
  EXPECT_EQ(ani.sizes.as<int64_t>()[0], 3);
}

TEST(ClusterDBSCANOpTest, MinPointsConfigurableFromParams) {
  // Authoring Params must select min_points (live Graph override deferred —
  // HSDK5_PORT §1).
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  0.2, 0.0,  10.0, 0.0, 10.1, 0.0,
  };
  auto p2 = points_tensor<double>(pts, 5, 2);
  auto p3 = points_tensor<double>(pts, 5, 2);
  const auto with2 = run_dbscan<double>(p2, make_params({0.5}, 2));
  const auto with3 = run_dbscan<double>(p3, make_params({0.5}, 3));
  EXPECT_EQ(with2.sizes.shape[0], 2);
  EXPECT_EQ(with3.sizes.shape[0], 1);
}

TEST(ClusterDBSCANOpTest, MaxClustersConfigurableFromParams) {
  // Authoring Params must select max_clusters (live Graph override deferred —
  // HSDK5_PORT §1).
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,  10.0, 0.0, 10.1, 0.0,  20.0, 0.0, 20.1, 0.0,
  };
  auto p = points_tensor<double>(pts, 6, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2, 2));
  ASSERT_EQ(out.sizes.shape[0], 2);
}

// ---- Error paths ----------------------------------------------------------

TEST(ClusterDBSCANOpTest, RejectsEmptyEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  DualIOGraphRunResult result =
      run_dbscan_result<double>(p, make_params({}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ClusterDBSCANOpTest, RejectsEpsilonLengthMismatch) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  DualIOGraphRunResult result =
      run_dbscan_result<double>(p, make_params({1.0, 2.0, 3.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ClusterDBSCANOpTest, RejectsNonPositiveEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  DualIOGraphRunResult result =
      run_dbscan_result<double>(p, make_params({0.0}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(ClusterDBSCANOpTest, RejectsBadPointsRank) {
  std::vector<double> pts = {0.0, 1.0, 2.0};
  auto p = real_cube_to_device<double>(pts, {3});  // rank 1
  DualIOGraphRunResult result =
      run_dbscan_result<double>(p, make_params({0.5}));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_FALSE(result.diagnostics.empty());
}

// ---- [live] overrides (pageable-host epsilon vector + scalars) ------------

// Carries all three of DBSCAN's [live] overrides; skew_ns applies to whichever
// of them this tick publishes. New slots go after skew_ns so the positional
// brace initializers below keep binding what they were written for.
struct LiveEpsilonTick {
  std::shared_ptr<holoscan::Tensor> points;
  std::optional<std::vector<double>> epsilon;
  int64_t skew_ns{0};
  std::optional<int64_t> min_points;
  std::optional<int64_t> max_clusters;
};

struct LiveEpsilonBank {
  static void reset(std::vector<LiveEpsilonTick> new_ticks) {
    ticks = std::move(new_ticks);
    index = 0U;
  }
  static inline std::vector<LiveEpsilonTick> ticks;
  static inline std::size_t index{0U};
};

class LiveEpsilonSourceOp final : public holoscan::Operator<> {
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
    spec.output(points, "out")
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, 4096U));
    spec.output(epsilon, "epsilon")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kHostVector, 1U, 128U * sizeof(double)));
    spec.output(min_points, "min_points");
    spec.output(max_clusters, "max_clusters");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = holoscan::radar::testing::detail::kHarnessTick});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (LiveEpsilonBank::index >= LiveEpsilonBank::ticks.size()) {
      return {};
    }
    const LiveEpsilonTick tick =
        LiveEpsilonBank::ticks[LiveEpsilonBank::index++];
    const holoscan::TimePoint points_time = context.activation_time();
    if (tick.epsilon.has_value()) {
      const std::vector<double>& values = *tick.epsilon;
      const std::vector<int64_t> shape{
          static_cast<int64_t>(values.size())};
      auto loan = epsilon.allocate_tensor(holoscan::TensorLoanRequest{
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
      holoscan::TimePoint epsilon_time = points_time;
      epsilon_time.timestamp_ns += tick.skew_ns;
      auto emitted = epsilon.emit(
          std::move(*loan),
          holoscan::EmitOptions{.capture_time = epsilon_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    holoscan::TimePoint scalar_time = points_time;
    scalar_time.timestamp_ns += tick.skew_ns;
    if (tick.min_points.has_value()) {
      auto emitted = min_points.emit(
          *tick.min_points,
          holoscan::EmitOptions{.capture_time = scalar_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    if (tick.max_clusters.has_value()) {
      auto emitted = max_clusters.emit(
          *tick.max_clusters,
          holoscan::EmitOptions{.capture_time = scalar_time});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    return holoscan::radar::testing::detail::emit_bank_frame(
        points, context, tick.points, "LiveEpsilonSourceOp");
  }

  holoscan::Output<holoscan::Tensor> points{"out"};
  holoscan::Output<holoscan::Tensor> epsilon{"epsilon"};
  holoscan::Output<int64_t> min_points{"min_points"};
  holoscan::Output<int64_t> max_clusters{"max_clusters"};
};

DualIOGraphRunResult run_epsilon_override_graph(
    std::vector<LiveEpsilonTick> ticks, ClusterDBSCANParams params,
    int64_t max_features = 2) {
  DualIOGraphRunResult result;
  holoscan::radar::testing::detail::FrameBank::reset();
  holoscan::radar::testing::detail::FrameBankB::reset();
  const std::size_t expected = ticks.size();
  LiveEpsilonBank::reset(std::move(ticks));

  holoscan::Graph graph{"epsilon-live-override"};
  const auto source = graph.op<LiveEpsilonSourceOp>("source");
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_features = max_features;
  auto op = graph.op<ClusterDBSCANOp<double>>("op", extents);
  op.set_params(std::move(params));
  const auto labels =
      graph.op<holoscan::radar::testing::detail::TensorCaptureOp>("labels");
  const auto sizes =
      graph.op<holoscan::radar::testing::detail::TensorCaptureBOp>("sizes");
  const holoscan::ConnectionOptions q{.queue_depth = 8U};
  graph.add_flow(source->points, op->points, q);
  graph.add_flow(source->epsilon, op->epsilon, q);
  graph.add_flow(source->min_points, op->min_points, q);
  graph.add_flow(source->max_clusters, op->max_clusters, q);
  graph.add_flow(op->labels, labels->in, q);
  graph.add_flow(op->sizes, sizes->in, q);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  holoscan::radar::testing::detail::bind_single_in_dual_out_device_pools(
      options, "labels", "sizes");
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
            holoscan::radar::testing::detail::FrameBankB::mutex);
        if (holoscan::radar::testing::detail::FrameBankB::captured.size() >=
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
    result.frames0 = std::move(
        holoscan::radar::testing::detail::FrameBank::captured);
  }
  {
    const std::lock_guard lock(
        holoscan::radar::testing::detail::FrameBankB::mutex);
    result.frames1 = std::move(
        holoscan::radar::testing::detail::FrameBankB::captured);
  }
  return result;
}

std::shared_ptr<holoscan::Tensor> live_epsilon_points() {
  return points_tensor<double>(
      {0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 5.0}, 4, 2);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonExactMatchChangesClustering) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{3.0, 1.0}}},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 1U);
  EXPECT_EQ(out.frames1[0].shape[0], 1);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonSkewedCaptureMissesExact) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{3.0, 1.0}, 1}},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 1U);
  EXPECT_EQ(out.frames1[0].shape[0], 0);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRevertsToParamsAfterMiss) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{3.0, 1.0}},
       {live_epsilon_points(), std::nullopt}},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 2U);
  EXPECT_EQ(out.frames1[0].shape[0], 1);
  EXPECT_EQ(out.frames1[1].shape[0], 0);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRejectsMalformedVector) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{}}},
      std::move(params));

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_FALSE(out.diagnostics.empty());
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRejectsConfiguredBoundOverflow) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{1.0, 1.0, 1.0}}},
      std::move(params), 2);

  EXPECT_TRUE(out.plan_ok) << out.plan_json;
  EXPECT_FALSE(out.diagnostics.empty());
}

// ---- Scalar [live] overrides ----------------------------------------------

// A triple and a pair, 0.5 apart within each group: min_points 2 admits both
// groups, min_points 3 leaves the pair short of a core point. The cluster
// count is therefore the shape of `sizes` — no float payload is compared.
std::shared_ptr<holoscan::Tensor> live_min_points_points() {
  return points_tensor<double>(
      {0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 10.0, 0.0, 10.1, 0.0}, 5, 2);
}

// Three well-separated pairs → three clusters when uncapped.
std::shared_ptr<holoscan::Tensor> live_max_clusters_points() {
  return points_tensor<double>(
      {0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0, 20.0, 0.0, 20.1, 0.0}, 6,
      2);
}

TEST(ClusterDBSCANOpTest, LiveMinPointsRevertsToParamsAfterMiss) {
  const auto out = run_epsilon_override_graph(
      {{live_min_points_points(), std::nullopt, 0, int64_t{3}},
       {live_min_points_points(), std::nullopt, 0, std::nullopt}},
      make_params({0.5}, 2));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 2U);
  // Tick 0 matches Exact and the pair drops to noise; tick 1 publishes
  // nothing, so min_points is the Params value again and the pair returns.
  EXPECT_EQ(out.frames1[0].shape[0], 1);
  EXPECT_EQ(out.frames1[1].shape[0], 2);
}

TEST(ClusterDBSCANOpTest, LiveMinPointsSkewedCaptureMissesExact) {
  ClusterDBSCANParams params = make_params({0.5}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_min_points_points(), std::nullopt, 1, int64_t{3}}},
      std::move(params));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 1U);
  EXPECT_EQ(out.frames1[0].shape[0], 2);
}

TEST(ClusterDBSCANOpTest, LiveMaxClustersRevertsToParamsAfterMiss) {
  const auto out = run_epsilon_override_graph(
      {{live_max_clusters_points(), std::nullopt, 0, std::nullopt,
        int64_t{2}},
       {live_max_clusters_points(), std::nullopt, 0, std::nullopt,
        std::nullopt}},
      make_params({0.5}, 2));

  ASSERT_TRUE(out.plan_ok) << out.plan_json;
  ASSERT_TRUE(out.diagnostics.empty());
  ASSERT_EQ(out.frames1.size(), 2U);
  // Params leave the cap open (3 clusters); the override caps tick 0 to 2,
  // and tick 1 publishes nothing so the open cap is back.
  EXPECT_EQ(out.frames1[0].shape[0], 2);
  EXPECT_EQ(out.frames1[1].shape[0], 3);
}

}  // namespace
