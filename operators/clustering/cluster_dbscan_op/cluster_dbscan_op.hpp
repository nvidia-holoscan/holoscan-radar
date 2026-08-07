// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// Density clustering (DBSCAN) over a caller-assembled feature matrix. This
// header is the contract reference for ports, shapes, dtypes, and parameters;
// module-wide conventions live in operators/common/README.md.
//
// In:  points  — float32/float64, (num_points, num_features). Radar-agnostic
//               feature matrix (e.g. [range, velocity, az, el]); one row per
//               point. Rows set the output row order.
// Out: labels  — int64, (num_points,). Cluster id per point (dense, starting
//               at 0); noise = -1. Row-aligned with points.
// Out: sizes   — int64, (num_clusters,). Point count per cluster; sized to the
//               actual cluster count and reallocated per tick. When
//               max_clusters is exceeded, surplus clusters are dropped (their
//               points relabeled to noise) and a warning is raised.
//
// Parameters (all [live], sticky overrides via same-named non-blocking ports):
//   epsilon      — std::vector<double>. Neighborhood radius per feature axis.
//                  Length 1 → isotropic scalar; length num_features → per-axis
//                  (anisotropic) ellipsoid: q is a neighbor of p when
//                  Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1. All values must be
//                  finite and > 0. Required (no default).
//   min_points   — int64. Minimum neighbors (including the point itself) for a
//                  core point (DBSCAN minPts). <= 0 selects the default
//                  heuristic 2·num_features (resolved per tick from the input).
//   max_clusters — int64. Upper cap on the number of clusters reported; surplus
//                  clusters are dropped (points relabeled to noise) and a
//                  warning raised if exceeded. <= 0 → uncapped (default).
//
// Precision is fixed at compile time (T = float | double). The clustering is a
// brute-force O(num_points²) neighbor sweep with iterative min-label connected
// components — appropriate for the sparse detection counts this stage sees; see
// OPEN_ITEMS.md for the spatial-index / union-find scaling follow-up.
template <typename T>
class ClusterDBSCANOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ClusterDBSCANOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClusterDBSCANOp)

  ClusterDBSCANOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::vector<double>> epsilon_;
  Parameter<int64_t> min_points_;
  Parameter<int64_t> max_clusters_;

  // Sticky [live] effective values, seeded from parameters on the first tick.
  std::vector<double> effective_epsilon_;
  int64_t effective_min_points_{0};
  int64_t effective_max_clusters_{0};
  bool seeded_{false};
  bool warned_max_clusters_{false};

  // Retain the input until the next tick for stream-ordered lifetime.
  std::shared_ptr<holoscan::Tensor> retained_points_;
};

}  // namespace holoscan::radar
