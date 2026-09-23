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
#include "radar_common/reduce/reduction_enums.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar {

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
// In:  points         — float32/float64, (num_points, num_features). Required;
//                       first ignores its values.
// Out: rep_idx        — int64, (num_clusters, 2 [+ n_batch]). One real
//                       detection_idx row per non-noise cluster; sized to the
//                       cluster count and reallocated per tick.
// Out: reduced_points — optional/latched T, (num_clusters, num_features).
//                       Selected input rows for member policies; synthetic
//                       weighted feature means for weighted_mean.
//
// Parameters (OPERATOR_API.md [live] tags):
//   representative  — enum {peak, medoid, first, weighted_mean}. [live] Exact
//                     Input.
//   score_method    — enum {weighted_sum, weighted_l2, weighted_max, custom}.
//                     Used by peak and weighted_mean. [live] Exact Input.
//   feature_weights — std::vector<double>, length num_features. Required for
//                     peak (non-custom) / medoid; empty selects uniform mean
//                     for weighted_mean.
//                     [live] Exact rank-1 pageable-host float64 Tensor Input,
//                     bounded by MaxTensorExtents.max_features.
//
// Precision T = float | double. Optional ScoreFn is a compile-time device
// functor used when score_method=custom: callable as
//   T operator()(const T* row, int64_t num_features) const
// on device. Default ScoreFn = NoCustomScore rejects custom at runtime.
// Instantiations with a non-default ScoreFn must be compiled in a CUDA TU
// (include detection_reduction_op_impl.cuh and explicitly instantiate).

struct DetectionReductionParams {
  RepresentativePolicy representative{RepresentativePolicy::kPeak};
  ScoreMethod score_method{ScoreMethod::kWeightedSum};
  std::vector<double> feature_weights{};
};

HOLOSCAN_PARAM_SCHEMA(
    DetectionReductionParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.DetectionReductionParams", 1),
    holoscan::field<&DetectionReductionParams::representative>("representative")
        .description("Reduction {peak, medoid, first, weighted_mean}; rep_idx always "
                     "selects a real member; default peak"),
    holoscan::field<&DetectionReductionParams::score_method>("score_method")
        .description("Peak score or weighted-mean row-weight collapse {weighted_sum, "
                     "weighted_l2, weighted_max, custom}; default weighted_sum"),
    holoscan::field<&DetectionReductionParams::feature_weights>("feature_weights")
        .description("Per-feature weights for peak/medoid/weighted_mean (length "
                     "num_features); empty requests uniform weighted_mean"));

static_assert(holoscan::param_schema_availability<DetectionReductionParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in DetectionReductionParams's namespace");

template <typename T, typename ScoreFn = NoCustomScore>
class DetectionReductionOp : public holoscan::Operator<DetectionReductionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "DetectionReductionOp supports float and double precision");

 public:
  explicit DetectionReductionOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> labels{"labels"};
  Input<Tensor> detection_idx{"detection_idx"};
  Input<Tensor> points{"points"};
  Output<Tensor> rep_idx{"rep_idx"};
  Output<Tensor> reduced_points{"reduced_points"};
  // [live] Exact overrides; the entries are unconditional and wiring the port
  // enables it (OPERATOR_API.md).
  Input<RepresentativePolicy> representative{"representative"};
  Input<ScoreMethod> score_method{"score_method"};
  Input<Tensor> feature_weights{"feature_weights"};

 private:
  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_labels_;
  Tensor retained_idx_;
  Tensor retained_points_;
  std::shared_ptr<holoscan::Tensor> retained_weights_;

  std::int64_t max_features_{0};
  std::size_t max_rep_idx_byte_span_{0U};
  std::size_t max_reduced_points_byte_span_{0U};
  LatchedOutput reduced_points_output_;
};

extern template class DetectionReductionOp<float>;
extern template class DetectionReductionOp<double>;

}  // namespace holoscan::radar
