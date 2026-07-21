// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// Column tokens for FeatureAssemblyOp.features (static; defines output layout).
enum class AssemblyFeature {
  kRange,        // range_bins[range_bin], or raw range_bin if bins absent
  kVelocity,     // velocity_bins[doppler_bin], or raw doppler_bin if absent
  kScore,        // gather detection_score at the cell (input required)
  kAz,           // aoa_estimates[row, 0] (sparse path; input required)
  kEl,           // aoa_estimates[row, 1] (sparse path; input required)
  kAzBin,        // az_angles[az_bin] (dense path; az_angles input required)
  kElBin,        // el_angles[el_bin] (dense path; el_angles input required)
  kDopplerBin,   // raw doppler_bin escape hatch
  kRangeBin,     // raw range_bin escape hatch
};

constexpr std::string_view to_string(AssemblyFeature feature) {
  switch (feature) {
    case AssemblyFeature::kRange: return "range";
    case AssemblyFeature::kVelocity: return "velocity";
    case AssemblyFeature::kScore: return "score";
    case AssemblyFeature::kAz: return "az";
    case AssemblyFeature::kEl: return "el";
    case AssemblyFeature::kAzBin: return "az_bin";
    case AssemblyFeature::kElBin: return "el_bin";
    case AssemblyFeature::kDopplerBin: return "doppler_bin";
    case AssemblyFeature::kRangeBin: return "range_bin";
  }
  return "unknown";
}

// Assembles the `points` feature matrix for clustering / DetectionReductionOp.
// This header is the contract reference for ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  detection_idx   — int64, (num_points, 2 [+ n_batch]). CFAR cells;
//                       columns are (…preserved_batch, doppler_bin,
//                       range_bin).
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
// vary the column set.
template <typename T>
class FeatureAssemblyOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FeatureAssemblyOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FeatureAssemblyOp)

  FeatureAssemblyOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Static — selects the output column layout.
  Parameter<std::vector<AssemblyFeature>> features_;

  // Retain inputs until the next tick for stream-ordered lifetime.
  std::shared_ptr<holoscan::Tensor> retained_idx_;
  std::shared_ptr<holoscan::Tensor> retained_range_bins_;
  std::shared_ptr<holoscan::Tensor> retained_velocity_bins_;
  std::shared_ptr<holoscan::Tensor> retained_score_;
  std::shared_ptr<holoscan::Tensor> retained_aoa_;
  std::shared_ptr<holoscan::Tensor> retained_az_angles_;
  std::shared_ptr<holoscan::Tensor> retained_el_angles_;
  // Scratch uploads used by the assemble kernel (feature ids, score offsets,
  // score leading dims). Feature ids are static → uploaded once. Leading
  // dims are re-uploaded only when the score-map layout changes.
  std::shared_ptr<holoscan::Tensor> retained_feature_ids_;
  std::shared_ptr<holoscan::Tensor> retained_score_offsets_;
  std::shared_ptr<holoscan::Tensor> retained_score_leading_;
  std::shared_ptr<holoscan::Tensor> retained_error_flag_;
  std::vector<int64_t> cached_score_leading_;
  bool feature_ids_ready_{false};
};

}  // namespace holoscan::radar
