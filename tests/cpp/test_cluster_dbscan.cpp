// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ClusterDBSCANOp contract tests (density clustering over a feature matrix).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::ClusterDBSCANOp;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::real_cube_to_device;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

struct DbscanResult {
  HostTensor labels;
  HostTensor sizes;
};

// Build an (rows, cols) points tensor of precision T from row-major doubles.
template <typename T>
std::shared_ptr<holoscan::Tensor> points_tensor(const std::vector<double>& host,
                                                int64_t rows, int64_t cols) {
  return real_cube_to_device<T>(host, {rows, cols});
}

template <typename T, typename... Args>
DbscanResult run_dbscan(std::shared_ptr<holoscan::Tensor> points,
                        Args&&... args) {
  auto harness =
      create_operator_test<ClusterDBSCANOp<T>>(std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "points", {single_map(std::move(points), "points")});

  TensorMapCapture labels_cap;
  TensorMapCapture sizes_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "labels", labels_cap.validators());
  harness->template add_output_port<holoscan::TensorMap>(
      "sizes", sizes_cap.validators());
  harness->run_test();
  if (labels_cap.frames.size() != 1u || sizes_cap.frames.size() != 1u) {
    throw std::runtime_error("ClusterDBSCANOp test: expected 1 output frame");
  }
  return {labels_cap.frames.front(), sizes_cap.frames.front()};
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
  const auto out =
      run_dbscan<double>(p, holoscan::Arg("epsilon") = std::vector<double>{0.5},
                         holoscan::Arg("min_points") = int64_t{3});

  ASSERT_EQ(out.labels.shape.size(), 1u);
  ASSERT_EQ(out.labels.shape[0], 8);
  const int64_t* lab = out.labels.as<int64_t>();
  // Blob A is one cluster (dense id 0 — lowest member row), blob B the next.
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
  const auto out = run_dbscan<double>(
      p, holoscan::Arg("epsilon") = std::vector<double>{3.0, 1.0},
      holoscan::Arg("min_points") = int64_t{2});

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
    const auto out = run_dbscan<double>(
        p, holoscan::Arg("epsilon") = std::vector<double>{0.5},
        holoscan::Arg("min_points") = int64_t{2});
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
    const auto out = run_dbscan<double>(
        p, holoscan::Arg("epsilon") = std::vector<double>{0.5},
        holoscan::Arg("min_points") = int64_t{3});
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
  const auto out = run_dbscan<double>(
      p, holoscan::Arg("epsilon") = std::vector<double>{0.5});
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
  const auto out = run_dbscan<double>(
      p, holoscan::Arg("epsilon") = std::vector<double>{0.5},
      holoscan::Arg("min_points") = int64_t{2},
      holoscan::Arg("max_clusters") = int64_t{2});
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

TEST(ClusterDBSCANOpTest, EmptyPointsEmitsEmptyOutputs) {
  std::vector<double> pts;
  auto p = points_tensor<double>(pts, 0, 2);
  const auto out = run_dbscan<double>(
      p, holoscan::Arg("epsilon") = std::vector<double>{0.5},
      holoscan::Arg("min_points") = int64_t{2});
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
  const auto of = run_dbscan<float>(
      pf, holoscan::Arg("epsilon") = std::vector<double>{0.5},
      holoscan::Arg("min_points") = int64_t{3});
  const auto od = run_dbscan<double>(
      pd, holoscan::Arg("epsilon") = std::vector<double>{0.5},
      holoscan::Arg("min_points") = int64_t{3});
  ASSERT_EQ(of.labels.shape[0], od.labels.shape[0]);
  for (int64_t i = 0; i < of.labels.shape[0]; ++i) {
    EXPECT_EQ(of.labels.as<int64_t>()[i], od.labels.as<int64_t>()[i]) << "row "
                                                                      << i;
  }
  ASSERT_EQ(of.sizes.shape[0], od.sizes.shape[0]);
}

// ---- Error paths ----------------------------------------------------------

TEST(ClusterDBSCANOpTest, RejectsEmptyEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  EXPECT_THROW(
      run_dbscan<double>(p, holoscan::Arg("epsilon") = std::vector<double>{}),
      std::invalid_argument);
}

TEST(ClusterDBSCANOpTest, RejectsEpsilonLengthMismatch) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  EXPECT_THROW(run_dbscan<double>(
                   p, holoscan::Arg("epsilon") = std::vector<double>{1.0, 2.0,
                                                                     3.0}),
               std::invalid_argument);
}

TEST(ClusterDBSCANOpTest, RejectsNonPositiveEpsilon) {
  std::vector<double> pts = {0.0, 0.0, 1.0, 1.0};
  auto p = points_tensor<double>(pts, 2, 2);
  const std::vector<double> zero_eps{0.0};
  EXPECT_THROW(run_dbscan<double>(p, holoscan::Arg("epsilon") = zero_eps),
               std::invalid_argument);
}

TEST(ClusterDBSCANOpTest, RejectsBadPointsRank) {
  std::vector<double> pts = {0.0, 1.0, 2.0};
  auto p = real_cube_to_device<double>(pts, {3});  // rank 1
  const std::vector<double> eps{0.5};
  EXPECT_THROW(run_dbscan<double>(p, holoscan::Arg("epsilon") = eps),
               std::invalid_argument);
}

// ---- Sticky [live] epsilon override ---------------------------------------

TEST(ClusterDBSCANOpTest, LiveEpsilonStickyOverride) {
  // Two nearby pairs. epsilon=0.9 keeps them separate (2 clusters); epsilon=3.0
  // links them (1 cluster). Override to 3.0 on tick 0, back to 0.9 on tick 1,
  // then no override on tick 2 → sticky 0.9.
  std::vector<double> pts = {
      0.0, 0.0,  0.1, 0.0,   // pair A
      2.0, 0.0,  2.1, 0.0,   // pair B
  };
  auto p = points_tensor<double>(pts, 4, 2);

  class App : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> points_frames;
    std::vector<std::vector<double>> epsilon_overrides;
    TensorMapCapture* labels_capture = nullptr;
    TensorMapCapture* sizes_capture = nullptr;

    void compose() override {
      auto op = make_operator<ClusterDBSCANOp<double>>(
          "dbscan", holoscan::Arg("epsilon") = std::vector<double>{0.9},
          holoscan::Arg("min_points") = int64_t{2});
      add_operator(op);
      wire_data_source(this, op, "points", points_frames);
      wire_leading_override<std::vector<double>>(this, op, "epsilon",
                                                 epsilon_overrides);
      wire_capture_sink(this, op, "labels", labels_capture);
      wire_capture_sink(this, op, "sizes", sizes_capture);
    }
  };

  App app;
  auto pm = single_map(p, "points");
  for (int i = 0; i < 3; ++i) {
    app.points_frames.push_back(pm);
  }
  app.epsilon_overrides = {std::vector<double>{3.0}, std::vector<double>{0.9}};
  TensorMapCapture labels_cap;
  TensorMapCapture sizes_cap;
  app.labels_capture = &labels_cap;
  app.sizes_capture = &sizes_cap;
  app.config();
  app.run();

  ASSERT_EQ(sizes_cap.frames.size(), 3u);
  EXPECT_EQ(sizes_cap.frames[0].shape[0], 1);  // eps=3.0 → merged
  EXPECT_EQ(sizes_cap.frames[1].shape[0], 2);  // eps=0.9 → split
  EXPECT_EQ(sizes_cap.frames[2].shape[0], 2);  // sticky 0.9 → split
  EXPECT_EQ(sizes_cap.frames[0].as<int64_t>()[0], 4);
}

}  // namespace
