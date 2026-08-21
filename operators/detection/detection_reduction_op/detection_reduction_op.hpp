// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
// In:  points         — float32/float64, (num_points, num_features). Optional;
//                       required unless effective representative == first.
// Out: rep_idx        — int64, (num_clusters, 2 [+ n_batch]). One real
//                       detection_idx row per non-noise cluster; sized to the
//                       cluster count and reallocated per tick.
//
// Parameters (OPERATOR_API.md [live] tags):
//   representative  — enum {peak, medoid, first}. Sticky [live] Input when
//                     wired.
//   score_method    — enum {weighted_sum, weighted_l2, weighted_max, custom}.
//                     Used by peak. Sticky [live] Input when wired.
//   feature_weights — std::vector<double>, length num_features. Required for
//                     peak (non-custom) / medoid.
//                     [EA HSDK5 ONLY] Params-only — Input<std::vector> is not
//                     a valid PublishedPayload (HSDK5_PORT.md §9); see
//                     OPEN_ITEMS.
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
  bool expect_representative{false};
  bool expect_score_method{false};
  bool sticky_representative{true};
  bool sticky_score_method{true};
};

HOLOSCAN_PARAM_SCHEMA(
    DetectionReductionParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.DetectionReductionParams", 1),
    holoscan::field<&DetectionReductionParams::representative>(
        "representative")
        .description(
            "Which cluster member to keep {peak, medoid, first}; default "
            "peak"),
    holoscan::field<&DetectionReductionParams::score_method>("score_method")
        .description(
            "Peak score collapse {weighted_sum, weighted_l2, weighted_max, "
            "custom}; used only by peak; default weighted_sum"),
    holoscan::field<&DetectionReductionParams::feature_weights>(
        "feature_weights")
        .description(
            "Per-feature weights for peak/medoid (length num_features); "
            "required for those policies unless peak+custom"),
    holoscan::field<&DetectionReductionParams::expect_representative>(
        "expect_representative")
        .description(
            "Set when the application wires the representative override."),
    holoscan::field<&DetectionReductionParams::expect_score_method>(
        "expect_score_method")
        .description(
            "Set when the application wires the score_method override."),
    holoscan::field<&DetectionReductionParams::sticky_representative>(
        "sticky_representative")
        .description(
            "Keep the last Exact representative match on a miss. Ignored "
            "unless expect_representative."),
    holoscan::field<&DetectionReductionParams::sticky_score_method>(
        "sticky_score_method")
        .description(
            "Keep the last Exact score_method match on a miss. Ignored "
            "unless expect_score_method."));

static_assert(
    holoscan::param_schema_availability<DetectionReductionParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in DetectionReductionParams's namespace");

template <typename T, typename ScoreFn = NoCustomScore>
class DetectionReductionOp
    : public holoscan::Operator<DetectionReductionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "DetectionReductionOp supports float and double precision");

 public:
  // Required: sizes the rep_idx Tensor pool at setup().
  explicit DetectionReductionOp(
      const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> labels{"labels"};
  Input<Tensor> detection_idx{"detection_idx"};
  // Optional per OPERATOR_API (required unless representative == first).
  // [EA HSDK5 ONLY] Exact in contract() — Graph tests always wire it; see
  // OPEN_ITEMS.
  Input<Tensor> points{"points"};
  Output<Tensor> rep_idx{"rep_idx"};
  // [live] sticky overrides (OPERATOR_API.md). feature_weights stays
  // Params-only (unbounded vector — HSDK5_PORT.md §9 / OPEN_ITEMS).
  Input<RepresentativePolicy> representative{"representative"};
  Input<ScoreMethod> score_method{"score_method"};

 private:
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};

  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_labels_;
  Tensor retained_idx_;
  Tensor retained_points_;
  std::shared_ptr<holoscan::Tensor> retained_weights_;
  // Sticky [live] POD enums: last override wins until the next.
  std::optional<RepresentativePolicy> sticky_representative_;
  std::optional<ScoreMethod> sticky_score_method_;
};

extern template class DetectionReductionOp<float>;
extern template class DetectionReductionOp<double>;

}  // namespace holoscan::radar
