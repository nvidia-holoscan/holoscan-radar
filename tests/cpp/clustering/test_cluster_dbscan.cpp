// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ClusterDBSCANOp contract tests (HSDK 5 Graph harness): density clustering
// over a feature matrix — blobs/noise, anisotropic epsilon, min_points /
// max_clusters Params coverage, descending-size numbering, error paths, and
// float/double agreement. Graph mid-stream sticky [live] overrides deferred
// (HSDK5_PORT.md §1 / §9).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
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
  // Authoring Params must select isotropic vs per-axis epsilon (live Graph
  // override deferred — HSDK5_PORT §9).
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

// ---- Sticky [live] overrides (deferred on EA) -----------------------------

TEST(ClusterDBSCANOpTest, LiveEpsilonStickyOverride) {
  // [EA HSDK5 ONLY] epsilon is Params-only: unbounded length-1-or-F vector
  // cannot use Input<std::vector> (HSDK5_PORT.md §9). Params authoring covers
  // selection (EpsilonConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live epsilon override deferred "
                  "(HSDK5_PORT.md §9 — Params-only unbounded vector)";
}

TEST(ClusterDBSCANOpTest, LiveMinPointsStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 — Graph mid-stream
  // sticky override harness fails on EA device/partition binding. Operator
  // still declares Input<int64_t> min_points + sticky resolve; Params cover
  // selection (MinPointsConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live min_points override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

TEST(ClusterDBSCANOpTest, LiveMaxClustersStickyOverride) {
  // [EA HSDK5 ONLY] See docs/planning/HSDK5_PORT.md §1 — Graph mid-stream
  // sticky override harness fails on EA device/partition binding. Operator
  // still declares Input<int64_t> max_clusters + sticky resolve; Params cover
  // selection (MaxClustersConfigurableFromParams).
  GTEST_SKIP() << "HSDK 5 EA: live max_clusters override harness deferred "
                  "(HSDK5_PORT.md §1)";
}

}  // namespace
