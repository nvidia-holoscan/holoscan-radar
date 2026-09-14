// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

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

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar {

// Column tokens for FeatureAssemblyOp.features (static; defines output layout).
enum class AssemblyFeature {
  kRange,       // range_bins[range_bin] (input required)
  kVelocity,    // velocity_bins[doppler_bin] (input required)
  kScore,       // gather detection_score at the cell (input required)
  kAz,          // az_angles[az_bin] (dense path; input required)
  kEl,          // el_angles[el_bin] (dense path; input required)
  kAzEstimate,  // aoa_estimates[row, 0] (sparse path; input required)
  kElEstimate,  // aoa_estimates[row, 1] (sparse path; input required)
  kDopplerBin,  // raw doppler_bin escape hatch
  kRangeBin,    // raw range_bin escape hatch
  kAzBin,       // raw preserved az_bin (dense path)
  kElBin,       // raw preserved el_bin (dense path)
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
    case AssemblyFeature::kAzEstimate:
      return "az_estimate";
    case AssemblyFeature::kElEstimate:
      return "el_estimate";
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

namespace config_detail {

inline constexpr AssemblyFeature kAssemblyFeatures[] = {
    AssemblyFeature::kRange,
    AssemblyFeature::kVelocity,
    AssemblyFeature::kScore,
    AssemblyFeature::kAz,
    AssemblyFeature::kEl,
    AssemblyFeature::kAzEstimate,
    AssemblyFeature::kElEstimate,
    AssemblyFeature::kDopplerBin,
    AssemblyFeature::kRangeBin,
    AssemblyFeature::kAzBin,
    AssemblyFeature::kElBin,
};
using AssemblyFeatureCodec =
    CanonicalEnumCodec<AssemblyFeature, kAssemblyFeatures>;  // NOLINT
using AssemblyFeatureStorage =
    PortableParamStorageTraits<AssemblyFeature>;  // NOLINT

}  // namespace config_detail
}  // namespace holoscan::radar

namespace holoscan::config {

template <>
struct ConfigCodec<radar::AssemblyFeature>
    : radar::config_detail::AssemblyFeatureCodec {};  // NOLINT

}  // namespace holoscan::config

namespace holoscan {

template <>
struct ParamStorageTraits<radar::AssemblyFeature>
    : radar::config_detail::AssemblyFeatureStorage {};  // NOLINT

}  // namespace holoscan

namespace holoscan::radar {

// Assembles the `points` feature matrix for clustering / DetectionReductionOp.
// This header is the contract reference for ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  detection_idx   — int64, (num_points, 2 [+ n_preserved]). CFAR cells;
//                       columns are (…preserved_batch, doppler_bin,
//                       range_bin). Required. Rank 2; no opaque leading batch.
// In:  range_bins      — optional float32/float64, (num_range,). Physical
//                       range (m); required when `range` is selected.
// In:  velocity_bins   — optional float32/float64, (num_doppler,). Physical
//                       velocity (m/s); required when `velocity` is selected.
// In:  detection_score — optional float32/float64, (..., num_doppler,
//                       num_range). Required when `score` is in features.
// In:  aoa_estimates   — optional float32/float64, (num_points, 2). Required
//                       for `az_estimate` / `el_estimate` (sparse path).
// In:  az_angles       — optional float32/float64, (num_az,). Azimuth scan
//                       grid; required when `az` is in features (dense).
// In:  el_angles       — optional float32/float64, (num_el,). Elevation scan
//                       grid; required when `el` is in features (dense).
// Out: points          — float32/float64, (num_points, num_features). Columns
//                       in `features` order; rows 1:1 with detection_idx.
//
// Precision is fixed at compile time (T = float | double). `features` is
// static (defines the output layout and temporal join); instantiate a separate
// operator or recompile to vary the column set. No [live] parameters. Fat
// output pool sizing and private workspace require MaxTensorExtents at
// graph.op() (ctor).

template <typename T>
class FeatureAssemblyOp : public holoscan::Operator<> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FeatureAssemblyOp supports float and double precision");

 public:
  // Features is constructor-only because it fixes both the points layout and
  // the compiled input join before setup().
  FeatureAssemblyOp(const MaxTensorExtents& extents,
                    std::vector<AssemblyFeature> features);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> detection_idx{"detection_idx"};
  // Gather sources. Static features determines which are required aligned
  // members; unselected ports are Exact + SafeDefault side inputs. Not [live]
  // overrides.
  Input<Tensor> range_bins{"range_bins"};
  Input<Tensor> velocity_bins{"velocity_bins"};
  Input<Tensor> detection_score{"detection_score"};
  Input<Tensor> aoa_estimates{"aoa_estimates"};
  Input<Tensor> az_angles{"az_angles"};
  Input<Tensor> el_angles{"el_angles"};
  Output<Tensor> points{"points"};

 private:
  expected<void, Error> ensure_device_feature_ids(
      const std::vector<int>& feature_ids, cudaStream_t stream);
  expected<void, Error> ensure_score_leading(
      const std::vector<int64_t>& score_leading, cudaStream_t stream);

  // Keep the previous activation's inputs alive across ticks so async
  // device reads cannot outlive the buffers.
  Tensor retained_idx_;
  Tensor retained_range_bins_;
  Tensor retained_velocity_bins_;
  Tensor retained_score_;
  Tensor retained_aoa_;
  Tensor retained_az_angles_;
  Tensor retained_el_angles_;

  // Ctor-sized workspace (not ConfigCache): feature-id table, score-plane
  // offsets, score leading dims, and the device error flag.
  std::shared_ptr<Tensor> feature_ids_;
  std::size_t feature_ids_capacity_{0};
  std::shared_ptr<Tensor> score_offsets_;
  std::size_t score_offsets_capacity_{0};
  std::shared_ptr<Tensor> score_leading_;
  std::size_t score_leading_capacity_{0};
  std::shared_ptr<Tensor> error_flag_;
  std::vector<int64_t> cached_score_leading_;
  // features is contract-static; lock the column map after the first
  // successful non-empty emit and reject a later Params permute.
  std::optional<std::vector<int>> seeded_feature_ids_{};

  std::vector<AssemblyFeature> features_;
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};
};

extern template class FeatureAssemblyOp<float>;
extern template class FeatureAssemblyOp<double>;

}  // namespace holoscan::radar
