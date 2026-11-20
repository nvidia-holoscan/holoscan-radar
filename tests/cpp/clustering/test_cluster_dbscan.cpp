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
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ClusterDBSCANOp;
using holoscan::radar::ClusterDBSCANParams;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;

constexpr DLDataType kFloat64Dtype{kDLFloat, 64, 1};

struct DbscanResult {
  HostTensor labels;
  HostTensor sizes;
};

template <typename T>
std::shared_ptr<holoscan::Tensor> points_tensor(const std::vector<double>& host, int64_t rows,
                                                int64_t cols) {
  return real_cube_to_device<T>(host, {rows, cols});
}

template <typename T>
holoscan::testing::TestResult run_dbscan_result(
    std::shared_ptr<holoscan::Tensor> points, ClusterDBSCANParams params,
    holoscan::radar::MaxTensorExtents extents = holoscan::radar::testing::default_test_extents()) {
  using Op = ClusterDBSCANOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(std::move(params));
  fixture.push(&Op::points, *points);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("labels", 1U, kMaximumBytes);
  fixture.template capture_snapshot<holoscan::Tensor>("sizes", 1U, kMaximumBytes);
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.output_device("labels", holoscan::DeviceId{0});
  fixture.output_device("sizes", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool labels = completion.wait_for_output(running, "labels", 1U);
  const bool sizes = completion.wait_for_output(running, "sizes", 1U);
  (void)labels;
  (void)sizes;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
DbscanResult run_dbscan(std::shared_ptr<holoscan::Tensor> points, ClusterDBSCANParams params) {
  const auto result = run_dbscan_result<T>(std::move(points), std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return {holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("labels")),
          holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("sizes"))};
}

ClusterDBSCANParams make_params(std::vector<double> epsilon, int64_t min_points = 0,
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
// See the operator README, "Optional output convention".
TEST(ClusterDBSCANOpTest, UnwiredSizesDoesNotFailTheRunOrDisturbLabels) {
  constexpr std::size_t kFrames = 8;
  const std::vector<double> pts = {
      0.0,  0.0,  0.1,  0.0,  0.0,  0.1,  0.1, 0.1,  // blob A (4)
      5.0,  5.0,                                     // outlier
      10.0, 10.0, 10.1, 10.0, 10.0, 10.1,            // blob B (3)
  };
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  frames.reserve(kFrames);
  for (std::size_t i = 0; i < kFrames; ++i) {
    frames.push_back(points_tensor<double>(pts, 8, 2));
  }

  using Op = ClusterDBSCANOp<double>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(make_params({0.5}, 3));
  for (const auto& frame : frames) {
    fixture.push(&Op::points, *frame);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("labels", kFrames, kMaximumBytes);
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.output_device("labels", holoscan::DeviceId{0});
  fixture.output_device("sizes", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "labels", kFrames);
  running.request_stop();
  const auto out = running.finish();
  completion.check(out);
  ASSERT_TRUE(observed) << out.describe_failure();
  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto captured = out.snapshots<holoscan::Tensor>("labels");
  ASSERT_EQ(captured.size(), kFrames);
  for (const auto& snapshot : captured) {
    const HostTensor frame = holoscan::radar::testing::to_host(snapshot);
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
      0.0,  0.0,  0.1,  0.0,  0.0,  0.1,  0.1, 0.1,  // blob A (4)
      5.0,  5.0,                                     // outlier
      10.0, 10.0, 10.1, 10.0, 10.0, 10.1,            // blob B (3)
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
      0.0, 0.0, 2.0, 0.0, 4.0, 0.0,  // x-chain
      0.0, 5.0,                      // far in y → noise
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

TEST(ClusterDBSCANOpTest, IgnoredFeatureDoesNotAffectLabelsOrReadValues) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> pts = {
      0.0,  0.0, 0.0, 0.0, nan,  0.1,  0.0, 0.0, 0.0, inf,
      10.0, 0.0, 0.0, 0.0, -inf, 10.1, 0.0, 0.0, 0.0, nan,
  };
  auto p = points_tensor<double>(pts, 4, 5);
  const auto out = run_dbscan<double>(p, make_params({0.5, 0.5, 0.5, 0.5, 0.0}, 2));

  EXPECT_EQ(std::vector<int64_t>(out.labels.as<int64_t>(),
                                 out.labels.as<int64_t>() + out.labels.shape[0]),
            (std::vector<int64_t>{0, 0, 1, 1}));
  EXPECT_EQ(
      std::vector<int64_t>(out.sizes.as<int64_t>(), out.sizes.as<int64_t>() + out.sizes.shape[0]),
      (std::vector<int64_t>{2, 2}));
}

TEST(ClusterDBSCANOpTest, MinPointsSeparatesCoreFromNoise) {
  // Group 1: a triple; group 2: a pair. epsilon=0.5.
  std::vector<double> pts = {
      0.0,  0.0, 0.1,  0.0, 0.2, 0.0,  // triple
      10.0, 0.0, 10.1, 0.0,            // pair
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

template <typename T>
class DbscanCoreThresholdBoundary : public ::testing::Test {};
using DbscanCoreThresholdPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DbscanCoreThresholdBoundary, DbscanCoreThresholdPrecisions);

TYPED_TEST(DbscanCoreThresholdBoundary, DenseExactThresholdAndNoCoreKeepExpectedLabels) {
  using T = TypeParam;
  // Five early neighbors, three insufficient neighbors, then four late neighbors.
  // At min_points=4, the final group reaches its threshold only at the last input row.
  const std::vector<double> points{0.0,  0.0, 0.0625,  0.0, 0.125,   0.0, 0.1875,  0.0,
                                   0.25, 0.0, 16.0,    0.0, 16.0625, 0.0, 16.125,  0.0,
                                   32.0, 0.0, 32.0625, 0.0, 32.125,  0.0, 32.1875, 0.0};
  for (const int64_t min_points : {int64_t{1}, int64_t{4}, int64_t{0}, int64_t{13}}) {
    SCOPED_TRACE(::testing::Message() << "min_points=" << min_points);
    const auto output =
        run_dbscan<T>(points_tensor<T>(points, 12, 2), make_params({1.0}, min_points));
    ASSERT_EQ(output.labels.shape, (std::vector<int64_t>{12}));
    const bool no_core = min_points > 12;
    const bool all_groups = min_points == 1;
    const std::vector<int64_t> expected_sizes =
        no_core ? std::vector<int64_t>{}
                : (all_groups ? std::vector<int64_t>{5, 4, 3} : std::vector<int64_t>{5, 4});
    ASSERT_EQ(output.sizes.shape,
              (std::vector<int64_t>{static_cast<int64_t>(expected_sizes.size())}));
    for (size_t cluster = 0; cluster < expected_sizes.size(); ++cluster) {
      EXPECT_EQ(output.sizes.template as<int64_t>()[cluster], expected_sizes[cluster]);
    }
    for (int64_t row = 0; row < 12; ++row) {
      const int64_t expected =
          no_core ? -1 : (row < 5 ? 0 : (row >= 8 ? 1 : (all_groups ? 2 : -1)));
      EXPECT_EQ(output.labels.template as<int64_t>()[row], expected) << "row=" << row;
    }
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

TYPED_TEST(DbscanCoreThresholdBoundary, ShuffledEqualChainsPreserveRootTiesAndNoise) {
  using T = TypeParam;
  // Binary-exact spacing creates two multihop chains, interleaved in input-row order.
  // Their lowest member rows are 0 and 1, not their lowest coordinate positions.
  const std::vector<double> positions{2.25, 0.0, 4.5, 0.75, 3.0, 5.25, 1.5, 3.75};
  std::vector<double> points;
  points.reserve(17U);
  for (const double position : positions) {
    points.push_back(position);
    points.push_back(position + 16.0);
  }
  points.push_back(48.0);
  for (const int64_t min_points : {int64_t{1}, int64_t{2}}) {
    SCOPED_TRACE(::testing::Message() << "min_points=" << min_points);
    const auto output =
        run_dbscan<T>(points_tensor<T>(points, 17, 1), make_params({1.0}, min_points));
    ASSERT_EQ(output.labels.shape, (std::vector<int64_t>{17}));
    const std::vector<int64_t> expected_sizes =
        min_points == 1 ? std::vector<int64_t>{8, 8, 1} : std::vector<int64_t>{8, 8};
    ASSERT_EQ(output.sizes.shape,
              (std::vector<int64_t>{static_cast<int64_t>(expected_sizes.size())}));
    for (size_t cluster = 0; cluster < expected_sizes.size(); ++cluster) {
      EXPECT_EQ(output.sizes.template as<int64_t>()[cluster], expected_sizes[cluster]);
    }
    for (int64_t row = 0; row < 16; ++row) {
      EXPECT_EQ(output.labels.template as<int64_t>()[row], row % 2) << "row=" << row;
    }
    EXPECT_EQ(output.labels.template as<int64_t>()[16], min_points == 1 ? 2 : -1);
  }
}

TEST(ClusterDBSCANOpTest, DefaultMinPointsCountsOnlyActiveFeatures) {
  std::vector<double> pts = {
      0.0,   100.0, 200.0, 300.0, 400.0, 0.1,   101.0, 201.0,
      301.0, 401.0, 0.2,   102.0, 202.0, 302.0, 402.0,
  };
  auto p = points_tensor<double>(pts, 3, 5);
  const auto out = run_dbscan<double>(p, make_params({0.5, 0.0, 0.0, 0.0, 0.0}));

  ASSERT_EQ(out.sizes.shape[0], 1);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
}

TEST(ClusterDBSCANOpTest, MaxClustersCapsAndDropsSurplus) {
  // Three separated pairs → three clusters; cap to 2 drops the third to noise.
  std::vector<double> pts = {
      0.0,  0.0, 0.1,  0.0,  // cluster 0
      10.0, 0.0, 10.1, 0.0,  // cluster 1
      20.0, 0.0, 20.1, 0.0,  // cluster 2 (dropped)
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
      0.0,  0.0, 0.1,  0.0,             // pair (rows 0-1) — smaller, root 0
      10.0, 0.0, 10.1, 0.0, 10.2, 0.0,  // triple (rows 2-4) — larger, root 2
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
      0.0,  0.0, 0.1,  0.0,             // pair (rows 0-1)
      10.0, 0.0, 10.1, 0.0, 10.2, 0.0,  // triple (rows 2-4)
  };
  auto p = points_tensor<double>(pts, 5, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2, 1));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[0], -1);  // pair dropped to noise
  EXPECT_EQ(lab[1], -1);
  EXPECT_EQ(lab[2], 0);  // triple kept
  EXPECT_EQ(lab[3], 0);
  EXPECT_EQ(lab[4], 0);
  ASSERT_EQ(out.sizes.shape[0], 1);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 3);
}

TEST(ClusterDBSCANOpTest, BorderPointJoinsMinRootCluster) {
  // Two disconnected four-point core components straddle row 8. The border
  // point sees only each component's nearest edge point plus itself, so it is
  // not core at min_points=4. It must join the component with root row 0.
  const std::vector<double> pts = {
      -0.9, -1.4, -1.4, -1.4,  // left core component, root row 0
      0.9,  1.4,  1.4,  1.4,   // right core component, root row 4
      0.0,                     // border point, row 8
  };
  auto p = points_tensor<double>(pts, 9, 1);
  const auto out = run_dbscan<double>(p, make_params({1.0}, 4));
  const int64_t* lab = out.labels.as<int64_t>();
  EXPECT_EQ(lab[8], lab[0]);
  EXPECT_NE(lab[8], lab[4]);
  ASSERT_EQ(out.sizes.shape[0], 2);
  EXPECT_EQ(out.sizes.as<int64_t>()[0], 5);
  EXPECT_EQ(out.sizes.as<int64_t>()[1], 4);
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
      0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 5.0, 5.0,
  };
  auto pf = points_tensor<float>(pts, 7, 2);
  auto pd = points_tensor<double>(pts, 7, 2);
  const auto of = run_dbscan<float>(pf, make_params({0.5}, 3));
  const auto od = run_dbscan<double>(pd, make_params({0.5}, 3));
  ASSERT_EQ(of.labels.shape[0], od.labels.shape[0]);
  for (int64_t i = 0; i < of.labels.shape[0]; ++i) {
    EXPECT_EQ(of.labels.as<int64_t>()[i], od.labels.as<int64_t>()[i]) << "row " << i;
  }
  ASSERT_EQ(of.sizes.shape[0], od.sizes.shape[0]);
}

TEST(ClusterDBSCANOpTest, EpsilonConfigurableFromParams) {
  // Params remain the seed when no live epsilon Tensor is expected.
  std::vector<double> pts = {
      0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 5.0,
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
  // Authoring Params select min_points when no Exact override matches.
  std::vector<double> pts = {
      0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 10.0, 0.0, 10.1, 0.0,
  };
  auto p2 = points_tensor<double>(pts, 5, 2);
  auto p3 = points_tensor<double>(pts, 5, 2);
  const auto with2 = run_dbscan<double>(p2, make_params({0.5}, 2));
  const auto with3 = run_dbscan<double>(p3, make_params({0.5}, 3));
  EXPECT_EQ(with2.sizes.shape[0], 2);
  EXPECT_EQ(with3.sizes.shape[0], 1);
}

TEST(ClusterDBSCANOpTest, MaxClustersConfigurableFromParams) {
  // Authoring Params select max_clusters when no Exact override matches.
  std::vector<double> pts = {
      0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0, 20.0, 0.0, 20.1, 0.0,
  };
  auto p = points_tensor<double>(pts, 6, 2);
  const auto out = run_dbscan<double>(p, make_params({0.5}, 2, 2));
  ASSERT_EQ(out.sizes.shape[0], 2);
}

// ---- Error paths ----------------------------------------------------------

template <typename T>
void check_point_capacity() {
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_points = 2;
  const auto exact = run_dbscan_result<T>(points_tensor<T>({0.0, 0.0, 0.1, 0.1}, 2, 2),
                                          make_params({0.5}, 2), extents);
  ASSERT_TRUE(exact.ok()) << exact.describe_failure();
  const auto host =
      holoscan::radar::testing::to_host(exact.template snapshot<holoscan::Tensor>("labels"));
  ASSERT_EQ(host.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(host.template as<int64_t>()[0], 0);
  EXPECT_EQ(host.template as<int64_t>()[1], 0);
  const auto overflow = run_dbscan_result<T>(points_tensor<T>({0.0, 0.0, 0.1, 0.1, 0.2, 0.2}, 3, 2),
                                             make_params({0.5}, 2), extents);
  EXPECT_FALSE(overflow.ok());
  EXPECT_NE(overflow.describe_failure().find("exceeds MaxTensorExtents.max_points 2"),
            std::string::npos)
      << overflow.describe_failure();
  EXPECT_TRUE(overflow.template snapshots<holoscan::Tensor>("labels").empty());
}

TEST(ClusterDBSCANOpTest, RejectsPointCapacityOverflowAndAcceptsExactBoundFloat) {
  check_point_capacity<float>();
}

TEST(ClusterDBSCANOpTest, RejectsPointCapacityOverflowAndAcceptsExactBoundDouble) {
  check_point_capacity<double>();
}

TEST(ClusterDBSCANOpTest, RejectsEmptyEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  auto result = run_dbscan_result<double>(p, make_params({}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsEpsilonLengthMismatch) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  auto result = run_dbscan_result<double>(p, make_params({1.0, 2.0, 3.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsScalarZeroEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  auto result = run_dbscan_result<double>(p, make_params({0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsAllIgnoredEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  auto result = run_dbscan_result<double>(p, make_params({0.0, 0.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsNegativeEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  auto result = run_dbscan_result<double>(p, make_params({1.0, -1.0}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsBadPointsRank) {
  std::vector<double> pts = {0.0, 1.0, 2.0};
  auto p = real_cube_to_device<double>(pts, {3});  // rank 1
  auto result = run_dbscan_result<double>(p, make_params({0.5}));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, RejectsOpaqueLeadingBatch) {
  const std::vector<double> pts = {0.0, 0.0, 0.1, 0.0};
  auto p = real_cube_to_device<double>(pts, {1, 2, 2});
  auto result = run_dbscan_result<double>(p, make_params({0.5}, 2));
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
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

holoscan::testing::TestResult run_epsilon_override_graph(std::vector<LiveEpsilonTick> ticks,
                                                         ClusterDBSCANParams params,
                                                         int64_t max_features = 2) {
  using Op = ClusterDBSCANOp<double>;
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
    if (ticks[index].epsilon.has_value()) {
      auto epsilon = holoscan::radar::testing::to_pageable_host_tensor(
          *ticks[index].epsilon, {static_cast<int64_t>(ticks[index].epsilon->size())},
          kFloat64Dtype);
      append(holoscan::testing::frame_at(override_time,
                                         holoscan::testing::input("epsilon", *epsilon)));
    }
    if (ticks[index].min_points.has_value()) {
      append(holoscan::testing::frame_at(
          override_time, holoscan::testing::input("min_points", *ticks[index].min_points)));
    }
    if (ticks[index].max_clusters.has_value()) {
      append(holoscan::testing::frame_at(
          override_time, holoscan::testing::input("max_clusters", *ticks[index].max_clusters)));
    }
    append(holoscan::testing::frame_at(base,
                                       holoscan::testing::input("points", *ticks[index].points)));
    fixture.push_frame(frame);
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("labels", ticks.size(), kMaximumBytes);
  fixture.capture_snapshot<holoscan::Tensor>("sizes", ticks.size(), kMaximumBytes);
  fixture.input_source_device("points", holoscan::DeviceId{0});
  fixture.output_device("labels", holoscan::DeviceId{0});
  fixture.output_device("sizes", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool labels = completion.wait_for_output(running, "labels", ticks.size());
  const bool sizes = completion.wait_for_output(running, "sizes", ticks.size());
  (void)labels;
  (void)sizes;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

std::vector<HostTensor> captured_sizes(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.snapshots<holoscan::Tensor>("sizes")) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

std::shared_ptr<holoscan::Tensor> live_epsilon_points() {
  return points_tensor<double>({0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 5.0}, 4, 2);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonExactMatchChangesClustering) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{3.0, 1.0}}}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 1U);
  EXPECT_EQ(sizes[0].shape[0], 1);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonSkewedCaptureMissesExact) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{3.0, 1.0}, 1}}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 1U);
  EXPECT_EQ(sizes[0].shape[0], 0);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRevertsToParamsAfterMiss) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out =
      run_epsilon_override_graph({{live_epsilon_points(), std::vector<double>{3.0, 1.0}},
                                  {live_epsilon_points(), std::nullopt}},
                                 std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 2U);
  EXPECT_EQ(sizes[0].shape[0], 1);
  EXPECT_EQ(sizes[1].shape[0], 0);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonCanToggleIgnoredAxis) {
  const auto make_points = [] {
    return points_tensor<double>({0.0, 0.0, 0.1, 0.0, 0.0, 10.0, 0.1, 10.0}, 4, 2);
  };
  const auto out = run_epsilon_override_graph({{make_points(), std::vector<double>{0.5, 0.0}},
                                               {make_points(), std::vector<double>{0.5, 1.0}}},
                                              make_params({0.5, 1.0}, 2));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 2U);
  EXPECT_EQ(sizes[0].shape[0], 1);
  EXPECT_EQ(sizes[1].shape[0], 2);
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRejectsMalformedVector) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph({{live_epsilon_points(), std::vector<double>{}}},
                                              std::move(params));

  EXPECT_FALSE(out.ok());
  EXPECT_FALSE(out.diagnostics().empty());
}

TEST(ClusterDBSCANOpTest, LiveEpsilonRejectsConfiguredBoundOverflow) {
  ClusterDBSCANParams params = make_params({1.0}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_epsilon_points(), std::vector<double>{1.0, 1.0, 1.0}}}, std::move(params), 2);

  EXPECT_FALSE(out.ok());
  EXPECT_FALSE(out.diagnostics().empty());
}

// ---- Scalar [live] overrides ----------------------------------------------

// A triple and a pair, 0.5 apart within each group: min_points 2 admits both
// groups, min_points 3 leaves the pair short of a core point. The cluster
// count is therefore the shape of `sizes` — no float payload is compared.
std::shared_ptr<holoscan::Tensor> live_min_points_points() {
  return points_tensor<double>({0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 10.0, 0.0, 10.1, 0.0}, 5, 2);
}

// Three well-separated pairs → three clusters when uncapped.
std::shared_ptr<holoscan::Tensor> live_max_clusters_points() {
  return points_tensor<double>({0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0, 20.0, 0.0, 20.1, 0.0}, 6,
                               2);
}

TEST(ClusterDBSCANOpTest, LiveMinPointsRevertsToParamsAfterMiss) {
  const auto out =
      run_epsilon_override_graph({{live_min_points_points(), std::nullopt, 0, int64_t{3}},
                                  {live_min_points_points(), std::nullopt, 0, std::nullopt}},
                                 make_params({0.5}, 2));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 2U);
  // Tick 0 matches Exact and the pair drops to noise; tick 1 publishes
  // nothing, so min_points is the Params value again and the pair returns.
  EXPECT_EQ(sizes[0].shape[0], 1);
  EXPECT_EQ(sizes[1].shape[0], 2);
}

TEST(ClusterDBSCANOpTest, LiveMinPointsSkewedCaptureMissesExact) {
  ClusterDBSCANParams params = make_params({0.5}, 2);
  const auto out = run_epsilon_override_graph(
      {{live_min_points_points(), std::nullopt, 1, int64_t{3}}}, std::move(params));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 1U);
  EXPECT_EQ(sizes[0].shape[0], 2);
}

TEST(ClusterDBSCANOpTest, LiveMaxClustersRevertsToParamsAfterMiss) {
  const auto out = run_epsilon_override_graph(
      {{live_max_clusters_points(), std::nullopt, 0, std::nullopt, int64_t{2}},
       {live_max_clusters_points(), std::nullopt, 0, std::nullopt, std::nullopt}},
      make_params({0.5}, 2));

  ASSERT_TRUE(out.ok()) << out.describe_failure();
  const auto sizes = captured_sizes(out);
  ASSERT_EQ(sizes.size(), 2U);
  // Params leave the cap open (3 clusters); the override caps tick 0 to 2,
  // and tick 1 publishes nothing so the open cap is back.
  EXPECT_EQ(sizes[0].shape[0], 2);
  EXPECT_EQ(sizes[1].shape[0], 3);
}

template <typename T>
class DbscanScratchReuseNative : public ::testing::Test {};
using DbscanScratchPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DbscanScratchReuseNative, DbscanScratchPrecisions);

TYPED_TEST(DbscanScratchReuseNative, ShrinkNoiseExpandAndFreshInstancePreserveAllRows) {
  using T = TypeParam;
  using Op = ClusterDBSCANOp<T>;
  const std::vector<std::vector<double>> points{
      {0, 0, 0.125, 0, 0, 0.125, 5, 5, 10, 10, 10.125, 10, 20, 20, 30, 30},
      {0, 0, 5, 5, 10, 10},
      {20, 20, 20.125, 20},
      {0, 0, 0.125, 0, 0, 0.125, 5, 5, 10, 10, 10.125, 10, 20, 20, 30, 30}};
  const std::vector<std::vector<int64_t>> expected_labels{
      {0, 0, 0, -1, 1, 1, -1, -1}, {-1, -1, -1}, {0, 0}, {0, 0, 0, -1, 1, 1, -1, -1}};
  const std::vector<std::vector<int64_t>> expected_sizes{{3, 2}, {}, {2}, {3, 2}};
  const auto run_sequence = [&](std::size_t count) {
    auto extents = holoscan::radar::testing::default_test_extents();
    extents.max_points = 8;
    extents.max_features = 2;
    extents.max_clusters = 8;
    std::vector<std::shared_ptr<holoscan::Tensor>> owners;
    auto fixture = holoscan::testing::create_operator_test<Op>(extents);
    fixture.set_params(make_params({0.5}, 2));
    for (std::size_t frame = 0; frame < count; ++frame) {
      owners.push_back(points_tensor<T>(points[frame], points[frame].size() / 2, 2));
      fixture.push(&Op::points, *owners.back());
    }
    fixture.template capture_snapshot<holoscan::Tensor>("labels", count, 4096U);
    fixture.template capture_snapshot<holoscan::Tensor>("sizes", count, 4096U);
    fixture.input_source_device("points", holoscan::DeviceId{0});
    fixture.output_device("labels", holoscan::DeviceId{0});
    fixture.output_device("sizes", holoscan::DeviceId{0});
    auto running = fixture.run_async({.expect_stop = true});
    const bool labels_ready = running.wait_for_output("labels", count, std::chrono::seconds{5});
    const bool sizes_ready = running.wait_for_output("sizes", count, std::chrono::seconds{5});
    running.request_stop();
    auto result = running.finish();
    EXPECT_TRUE(labels_ready && sizes_ready) << result.describe_failure();
    return result;
  };
  for (const std::size_t count : {points.size(), std::size_t{1}}) {
    const auto result = run_sequence(count);
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto labels = result.template snapshots<holoscan::Tensor>("labels");
    const auto sizes = result.template snapshots<holoscan::Tensor>("sizes");
    ASSERT_EQ(labels.size(), count);
    ASSERT_EQ(sizes.size(), count);
    for (std::size_t frame = 0; frame < count; ++frame) {
      const auto lab = holoscan::radar::testing::to_host(labels[frame]);
      const auto size = holoscan::radar::testing::to_host(sizes[frame]);
      ASSERT_EQ(lab.shape,
                (std::vector<int64_t>{static_cast<int64_t>(expected_labels[frame].size())}));
      ASSERT_EQ(size.shape,
                (std::vector<int64_t>{static_cast<int64_t>(expected_sizes[frame].size())}));
      for (std::size_t i = 0; i < expected_labels[frame].size(); ++i) {
        EXPECT_EQ(lab.template as<int64_t>()[i], expected_labels[frame][i]);
      }
      for (std::size_t i = 0; i < expected_sizes[frame].size(); ++i) {
        EXPECT_EQ(size.template as<int64_t>()[i], expected_sizes[frame][i]);
      }
    }
    if (count > 1) {
      EXPECT_EQ(holoscan::radar::testing::to_host(labels.front()).data,
                holoscan::radar::testing::to_host(labels.back()).data);
      EXPECT_EQ(holoscan::radar::testing::to_host(sizes.front()).data,
                holoscan::radar::testing::to_host(sizes.back()).data);
    }
  }
}

}  // namespace
