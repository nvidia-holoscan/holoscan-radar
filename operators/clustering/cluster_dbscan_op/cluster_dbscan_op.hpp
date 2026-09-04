// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar {

namespace detail {
struct ClusterDBSCANState;
}  // namespace detail

// Density clustering (DBSCAN) over a caller-assembled feature matrix. This
// header is the contract reference for ports, shapes, dtypes, and parameters;
// module-wide conventions live in operators/common/README.md.
//
// In:  points  — float32/float64, (num_points, num_features). Radar-agnostic
//               feature matrix (e.g. [range, velocity, az, el]); one row per
//               point. Rows set the output row order.
// Out: labels  — int64, (num_points,). Cluster id per point (dense, starting
//               at 0); noise = -1. Row-aligned with points. Clusters numbered
//               by descending point count (cluster 0 = largest); ties broken
//               by ascending component root.
// Out: sizes   — optional/latched int64, (num_clusters,). Point count per
//               cluster; sized to the actual cluster count and reallocated per
//               tick. When max_clusters is exceeded, surplus clusters are
//               dropped (their points relabeled to noise) and a warning is
//               raised.
//
// Parameters (OPERATOR_API.md [live] tags):
//   epsilon      — std::vector<double>. Neighborhood radius per feature axis.
//                  Length 1 → isotropic scalar; length num_features → per-axis
//                  (anisotropic) ellipsoid: q is a neighbor of p when
//                  Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1. All values must be
//                  finite and > 0. Required (no default).
//                  [live] Exact rank-1 pageable-host float64 Tensor Input,
//                  bounded by MaxTensorExtents.max_features.
//   min_points   — int64. Minimum neighbors (including the point itself) for a
//                  core point (DBSCAN minPts). <= 0 selects the default
//                  heuristic 2·num_features (resolved per tick from the input).
//                  [live] Exact Input.
//   max_clusters — int64. Upper cap on the number of clusters reported; surplus
//                  clusters are dropped (points relabeled to noise) and a
//                  warning raised if exceeded. <= 0 → uncapped (default).
//                  [live] Exact Input.
//
// Precision is fixed at compile time (T = float | double). The clustering is a
// brute-force O(num_points²) neighbor sweep with iterative min-label connected
// components — appropriate for the sparse detection counts this stage sees.

struct ClusterDBSCANParams {
  std::vector<double> epsilon{};
  int64_t min_points{0};
  int64_t max_clusters{0};
};

HOLOSCAN_PARAM_SCHEMA(
    ClusterDBSCANParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.ClusterDBSCANParams", 1),
    holoscan::field<&ClusterDBSCANParams::epsilon>("epsilon")
        .description(
            "Neighborhood radius: length 1 (isotropic) or num_features "
            "(per-axis); all values finite and > 0; required"),
    holoscan::field<&ClusterDBSCANParams::min_points>("min_points")
        .description(
            "Minimum neighbors (incl. self) for a core point; <= 0 selects "
            "the 2*num_features default"),
    holoscan::field<&ClusterDBSCANParams::max_clusters>("max_clusters")
        .description(
            "Upper cap on reported clusters (surplus dropped to noise with a "
            "warning); <= 0 is uncapped"));

static_assert(holoscan::param_schema_availability<ClusterDBSCANParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in ClusterDBSCANParams's "
              "namespace");

template <typename T>
class ClusterDBSCANOp : public holoscan::Operator<ClusterDBSCANParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ClusterDBSCANOp supports float and double precision");

 public:
  // Required: labels and sizes output pools. A live epsilon port also requires
  // max_features to bound its vector payload.
  explicit ClusterDBSCANOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> points{"points"};
  Output<Tensor> labels{"labels"};
  Output<Tensor> sizes{"sizes"};
  // [live] Exact overrides (OPERATOR_API.md).
  Input<Tensor> epsilon{"epsilon"};
  Input<int64_t> min_points{"min_points"};
  Input<int64_t> max_clusters{"max_clusters"};

 private:
  // Keep the previous activation's input alive across ticks so async device
  // reads cannot outlive the buffer.
  Tensor retained_points_;
  // sizes is optional/latched (OPERATOR_API.md).
  LatchedOutput sizes_out_;
  std::shared_ptr<detail::ClusterDBSCANState> state_;

  std::int64_t max_features_{0};
  std::size_t max_labels_byte_span_{0U};
  std::size_t max_sizes_byte_span_{0U};
};

extern template class ClusterDBSCANOp<float>;
extern template class ClusterDBSCANOp<double>;

}  // namespace holoscan::radar
