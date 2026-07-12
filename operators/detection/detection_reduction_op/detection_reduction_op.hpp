// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/reduce/reduction_enums.hpp"

namespace holoscan::radar {

// Default ScoreFn placeholder; used only when score_method == custom.
struct NoCustomScore {};

// Reduces each cluster to one representative detection cell (cluster-then-
// angle path). This header is the contract reference for ports, shapes,
// dtypes, and parameters; module-wide conventions live in
// operators/common/README.md.
//
// In:  labels         — int64, (num_points,). Cluster id per detection;
//                       noise = -1 (excluded). Non-noise labels must be in
//                       [0, max_label] (DBSCAN-style); empty ids are skipped
//                       when compacting to num_clusters.
// In:  detection_idx  — int64, (num_points, 2 [+ n_batch]). CFAR cells;
//                       row-aligned with labels.
// In:  points         — float32/float64, (num_points, num_features). Required
//                       unless effective representative == first.
// Out: rep_idx        — int64, (num_clusters, 2 [+ n_batch]). One real
//                       detection_idx row per non-noise cluster; sized to the
//                       cluster count and reallocated per tick.
//
// Precision T = float | double. Optional ScoreFn is a compile-time functor
// used only when score_method == custom (default instantiation rejects
// custom).
template <typename T, typename ScoreFn = NoCustomScore>
class DetectionReductionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "DetectionReductionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectionReductionOp)

  DetectionReductionOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<RepresentativePolicy> representative_;
  Parameter<ScoreMethod> score_method_;
  Parameter<std::vector<double>> feature_weights_;

  RepresentativePolicy effective_representative_{RepresentativePolicy::kPeak};
  ScoreMethod effective_score_method_{ScoreMethod::kWeightedSum};
  std::vector<double> effective_feature_weights_;
  bool seeded_{false};

  std::shared_ptr<holoscan::Tensor> retained_labels_;
  std::shared_ptr<holoscan::Tensor> retained_idx_;
  std::shared_ptr<holoscan::Tensor> retained_points_;
  std::shared_ptr<holoscan::Tensor> retained_weights_;
  ScoreFn score_fn_{};
};

}  // namespace holoscan::radar
