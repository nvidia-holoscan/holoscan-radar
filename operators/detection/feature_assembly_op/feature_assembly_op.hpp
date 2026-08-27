// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
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

namespace holoscan::radar {

// Column tokens for FeatureAssemblyOp.features (static; defines output layout).
enum class AssemblyFeature {
  kRange,       // range_bins[range_bin], or raw range_bin if bins absent
  kVelocity,    // velocity_bins[doppler_bin], or raw doppler_bin if absent
  kScore,       // gather detection_score at the cell (input required)
  kAz,          // aoa_estimates[row, 0] (sparse path; input required)
  kEl,          // aoa_estimates[row, 1] (sparse path; input required)
  kAzBin,       // az_angles[az_bin] (dense path; az_angles input required)
  kElBin,       // el_angles[el_bin] (dense path; el_angles input required)
  kDopplerBin,  // raw doppler_bin escape hatch
  kRangeBin,    // raw range_bin escape hatch
};

constexpr std::string_view to_string(AssemblyFeature feature) {
  switch (feature) {
    case AssemblyFeature::kRange:
      return "range";
    case AssemblyFeature::kVelocity:
      return "velocity";
    case AssemblyFeature::kScore:
      return "score";
    case AssemblyFeature::kAz:
      return "az";
    case AssemblyFeature::kEl:
      return "el";
    case AssemblyFeature::kAzBin:
      return "az_bin";
    case AssemblyFeature::kElBin:
      return "el_bin";
    case AssemblyFeature::kDopplerBin:
      return "doppler_bin";
    case AssemblyFeature::kRangeBin:
      return "range_bin";
  }
  return "unknown";
}

// Assembles the `points` feature matrix for clustering / DetectionReductionOp.
// This header is the contract reference for ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  detection_idx   — int64, (num_points, 2 [+ n_batch]). CFAR cells;
//                       columns are (…preserved_batch, doppler_bin,
//                       range_bin). Required.
// In:  range_bins      — optional float32/float64, (num_range,). Physical
//                       range (m); absent → raw range_bin for `range`.
// In:  velocity_bins   — optional float32/float64, (num_doppler,). Physical
//                       velocity (m/s); absent → raw doppler_bin for
//                       `velocity`.
// In:  detection_score — optional float32/float64, (..., num_doppler,
//                       num_range). Required when `score` is in features.
// In:  aoa_estimates   — optional float32/float64, (num_points, 2). Required
//                       when `az` or `el` is in features (sparse path).
// In:  az_angles       — optional float32/float64, (num_az,). Azimuth scan
//                       grid; required when `az_bin` is in features (dense).
// In:  el_angles       — optional float32/float64, (num_el,). Elevation scan
//                       grid; required when `el_bin` is in features (dense).
// Out: points          — float32/float64, (num_points, num_features). Columns
//                       in `features` order; rows 1:1 with detection_idx.
//
// Precision is fixed at compile time (T = float | double). `features` is
// static (defines the output layout); instantiate a separate operator to
// vary the column set. No [live] parameters.

struct FeatureAssemblyParams {
  std::vector<AssemblyFeature> features{};
};

HOLOSCAN_PARAM_SCHEMA(
    FeatureAssemblyParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.FeatureAssemblyParams",
                           1),
    holoscan::field<&FeatureAssemblyParams::features>("features").description(
        "Ordered AssemblyFeature tokens selecting points columns "
        "(static; defines output layout; required, non-empty)"));

static_assert(holoscan::param_schema_availability<FeatureAssemblyParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in FeatureAssemblyParams's "
              "namespace");

template <typename T>
class FeatureAssemblyOp : public holoscan::Operator<FeatureAssemblyParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FeatureAssemblyOp supports float and double precision");

 public:
  // Required: sizes the points Tensor pool at setup().
  explicit FeatureAssemblyOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> detection_idx{"detection_idx"};
  // Optional gather sources — omitted from contract() so unconnected graphs
  // compile (TEMPORAL_INPUT_UNCONNECTED otherwise; SafeDefault alone is not
  // enough on EA — HSDK5_PORT.md / Beamscan optional is_valid pattern).
  Input<Tensor> range_bins{"range_bins"};
  Input<Tensor> velocity_bins{"velocity_bins"};
  Input<Tensor> detection_score{"detection_score"};
  Input<Tensor> aoa_estimates{"aoa_estimates"};
  Input<Tensor> az_angles{"az_angles"};
  Input<Tensor> el_angles{"el_angles"};
  Output<Tensor> points{"points"};

 private:
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};

  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_idx_;
  Tensor retained_range_bins_;
  Tensor retained_velocity_bins_;
  Tensor retained_score_;
  Tensor retained_aoa_;
  Tensor retained_az_angles_;
  Tensor retained_el_angles_;
  // Scratch uploads used by the assemble kernel (feature ids, score offsets,
  // score leading dims). Feature ids are static → uploaded once. Leading
  // dims are re-uploaded only when the score-map layout changes.
  std::shared_ptr<holoscan::Tensor> retained_feature_ids_;
  std::shared_ptr<holoscan::Tensor> retained_score_offsets_;
  std::shared_ptr<holoscan::Tensor> retained_score_leading_;
  std::shared_ptr<holoscan::Tensor> retained_error_flag_;
  std::vector<int64_t> cached_score_leading_;
  // features is contract-static; lock the column map after the first upload
  // and reject a later Params permute that would silently reuse stale ids.
  std::optional<std::vector<int>> seeded_feature_ids_{};
};

extern template class FeatureAssemblyOp<float>;
extern template class FeatureAssemblyOp<double>;

}  // namespace holoscan::radar
