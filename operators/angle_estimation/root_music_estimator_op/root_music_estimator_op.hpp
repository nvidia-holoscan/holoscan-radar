// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
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

#include "angle_estimation/angle_boundary_device.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

namespace detail {
struct RootMUSICSolverState;
}  // namespace detail

// Gridless Root-MUSIC sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md → "RootMUSICEstimatorOp".
//
// Covariance / Hermitian eig / fixed-or-EFT source count / noise-subspace
// projector match MUSICEstimatorOp. On a ULA, Root-MUSIC roots the
// conjugate-reciprocal polynomial; on a filled URA, reduced-dimension
// per-axis rooting + 2-D MUSIC-metric pairing yields native (az, el).
//
// In:  snapshot       — complex64/complex128,
//                       (num_items, num_channels, num_snapshots).
// In:  is_valid       — optional bool/uint8, (num_items, num_snapshots).
//                       Omitted from the temporal contract when unconnected
//                       (HSDK5_PORT §1).
// Out: aoa_estimates  — float32/float64, (num_items, 2). Always emitted.
// Out: doa_multi      — float32/float64, (num_items, max_sources, 2).
//                       Always emitted (optional in OPERATOR_API.md).
// Out: source_count   — int32, (num_items,). Always emitted.
// Out: eigenvalues    — float32/float64,
//                       (num_items, post-smoothing num_channels). Always
//                       emitted. No angle_spectrum (gridless).
//
// Exact Input for all [live] parameters. angle_convention and forward_backward stay Params-only
// (STATIC / not [live]).

struct RootMUSICParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  // Negative → auto (per-item EFT); non-negative → fixed order. Default -1.
  int64_t num_sources{-1};
  double source_count_pfa{1e-2};
  int64_t max_sources{4};
  // Empty → off; [P] ULA; [P_az, P_el] URA.
  std::vector<int64_t> spatial_smoothing{};
  bool forward_backward{true};
  bool sticky_array_geometry{true};
  bool sticky_spatial_smoothing{true};
  bool sticky_num_sources{true};
  bool sticky_source_count_pfa{true};
  bool sticky_max_sources{true};
};

HOLOSCAN_PARAM_SCHEMA(
    RootMUSICParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.RootMUSICParams", 1),
    holoscan::field<&RootMUSICParams::array_geometry>("array_geometry")
        .description(
            "ULA or filled URA positions and wavelength; num_elements must "
            "match snapshot channels"),
    holoscan::field<&RootMUSICParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the gridless DOA output "
            "(no live port)"),
    holoscan::field<&RootMUSICParams::num_sources>("num_sources")
        .description(
            "Signal-subspace dimension; negative selects per-item EFT auto"),
    holoscan::field<&RootMUSICParams::source_count_pfa>("source_count_pfa")
        .description(
            "EFT over-estimation probability for num_sources: auto"),
    holoscan::field<&RootMUSICParams::max_sources>("max_sources")
        .description(
            "doa_multi / source_count capacity and auto source-count cap"),
    holoscan::field<&RootMUSICParams::spatial_smoothing>("spatial_smoothing")
        .description(
            "Per-aperture-axis sub-array counts on a filled ULA/URA "
            "(empty = off)"),
    holoscan::field<&RootMUSICParams::forward_backward>("forward_backward")
        .description(
            "Include conjugate-reversed looks when spatial_smoothing is set"),
    holoscan::field<&RootMUSICParams::sticky_array_geometry>(
        "sticky_array_geometry")
        .description("Keep the last Exact array_geometry match on a miss."),
    holoscan::field<&RootMUSICParams::sticky_spatial_smoothing>(
        "sticky_spatial_smoothing")
        .description("Keep the last Exact spatial_smoothing match on a miss."),
    holoscan::field<&RootMUSICParams::sticky_num_sources>("sticky_num_sources")
        .description("Keep the last Exact num_sources match on a miss."),
    holoscan::field<&RootMUSICParams::sticky_source_count_pfa>(
        "sticky_source_count_pfa")
        .description("Keep the last Exact source_count_pfa match on a miss."),
    holoscan::field<&RootMUSICParams::sticky_max_sources>("sticky_max_sources")
        .description("Keep the last Exact max_sources match on a miss."));

static_assert(holoscan::param_schema_availability<RootMUSICParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in RootMUSICParams's namespace");

template <typename T>
class RootMUSICEstimatorOp : public holoscan::Operator<RootMUSICParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "RootMUSICEstimatorOp supports float and double precision");

 public:
  // Required: sizes the fat Tensor output pools at setup().
  explicit RootMUSICEstimatorOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;
  // Drain MatX eig plan caches + global MemTracker once at graph stop, while
  // CUDA is still valid. HSDK 5 still exposes OperatorBase::stop().
  void stop() override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional; unconnected graphs omit it from the temporal contract.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> doa_multi{"doa_multi"};
  Output<Tensor> source_count{"source_count"};
  Output<Tensor> eigenvalues{"eigenvalues"};
  // [live] Exact overrides (OPERATOR_API.md).
  Input<WireArrayGeometry> array_geometry{"array_geometry"};
  Input<SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  Input<int64_t> num_sources{"num_sources"};
  Input<double> source_count_pfa{"source_count_pfa"};
  Input<int64_t> max_sources{"max_sources"};

 private:
  MaxTensorExtents extents_;
  std::size_t aoa_max_byte_span_{0U};
  std::size_t doa_max_byte_span_{0U};
  std::size_t count_max_byte_span_{0U};
  std::size_t eig_max_byte_span_{0U};
  std::optional<int64_t> max_channels_;

  Tensor retained_snapshot_;
  Tensor retained_is_valid_;
  std::optional<ArrayGeometry> sticky_array_geometry_;
  std::optional<std::vector<int64_t>> sticky_spatial_smoothing_;
  std::optional<int64_t> sticky_num_sources_;
  std::optional<double> sticky_source_count_pfa_;
  std::optional<int64_t> sticky_max_sources_;

  bool warned_rank_deficient_{false};
  detail::DeviceAngleConvention<T> device_angle_convention_{};
  // The convention the device relabeling was built from; angle_convention is
  // contract-static, so a later change is rejected rather than ignored.
  std::optional<AngleConvention> seeded_angle_convention_{};

  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::unordered_set<std::string> seen_geometry_keys_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};

  std::shared_ptr<detail::RootMUSICSolverState> solver_state_{};
  // Second companion-root solver for the URA elevation axis.
  std::shared_ptr<detail::RootMUSICSolverState> solver_state_z_{};
};

extern template class RootMUSICEstimatorOp<float>;
extern template class RootMUSICEstimatorOp<double>;

}  // namespace holoscan::radar
