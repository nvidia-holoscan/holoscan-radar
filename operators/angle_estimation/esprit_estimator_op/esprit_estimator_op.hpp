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
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/dsp/shift_invariant_subarray.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"

namespace holoscan::radar {

// Gridless least-squares ESPRIT sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md → "ESPRITEstimatorOp".
//
// Covariance / Hermitian eig / fixed-or-EFT source count match MUSICEstimatorOp.
// Signal subspace is restricted to two corresponding displaced subarrays;
// eigenvalue rotation gives continuous 1-D DOAs (ULA) or dual-invariance 2-D
// DOAs (filled URA, subarray_spec: auto).
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
// [EA HSDK5 ONLY] Sticky Input for POD [live] scalars (num_sources,
// source_count_pfa, max_sources). array_geometry / spatial_smoothing /
// subarray_spec are Params-only (HSDK5_PORT.md §9; subarray_spec is STATIC).
// angle_convention and forward_backward stay Params-only (STATIC / not [live]).

struct ESPRITParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  // Negative → auto (per-item EFT); non-negative → fixed order. Default -1.
  int64_t num_sources{-1};
  double source_count_pfa{1e-2};
  int64_t max_sources{4};
  // Empty → off; [P] ULA; [P_az, P_el] URA. Params-only on EA (§9).
  std::vector<int64_t> spatial_smoothing{};
  bool forward_backward{true};
  // Static structural; empty vectors = auto. Params-only (§9).
  ShiftInvariantSubarrayPair subarray_spec{};
};

HOLOSCAN_PARAM_SCHEMA(
    ESPRITParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.ESPRITParams", 1),
    holoscan::field<&ESPRITParams::array_geometry>("array_geometry")
        .description(
            "Shift-invariant array positions and wavelength; num_elements "
            "must match snapshot channels"),
    holoscan::field<&ESPRITParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the gridless DOA output "
            "(no live port)"),
    holoscan::field<&ESPRITParams::num_sources>("num_sources")
        .description(
            "Signal-subspace dimension; negative selects per-item EFT auto"),
    holoscan::field<&ESPRITParams::source_count_pfa>("source_count_pfa")
        .description(
            "EFT over-estimation probability for num_sources: auto"),
    holoscan::field<&ESPRITParams::max_sources>("max_sources")
        .description(
            "doa_multi / source_count capacity and auto source-count cap"),
    holoscan::field<&ESPRITParams::spatial_smoothing>("spatial_smoothing")
        .description(
            "Per-aperture-axis sub-array counts on a filled ULA/URA "
            "(empty = off)"),
    holoscan::field<&ESPRITParams::forward_backward>("forward_backward")
        .description(
            "Include conjugate-reversed looks when spatial_smoothing is set"),
    holoscan::field<&ESPRITParams::subarray_spec>("subarray_spec")
        .description(
            "Shift-invariant subarray pair; empty derives ULA/URA auto"));

static_assert(holoscan::param_schema_availability<ESPRITParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in ESPRITParams's namespace");

inline constexpr std::uint32_t kESPRITEstimatorMaxRank = 8U;
// 256 MiB
inline constexpr std::size_t kESPRITEstimatorMaxByteSpan = 256ULL << 20;

template <typename T>
class ESPRITEstimatorOp : public holoscan::Operator<ESPRITParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ESPRITEstimatorOp supports float and double precision");

 public:
  ESPRITEstimatorOp() = default;

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
  // [live] sticky POD overrides (OPERATOR_API.md). Vector / descriptor live
  // params stay Params-only on HSDK 5 EA — see OPEN_ITEMS.md / HSDK5_PORT §9.
  Input<int64_t> num_sources{"num_sources"};
  Input<double> source_count_pfa{"source_count_pfa"};
  Input<int64_t> max_sources{"max_sources"};

 private:
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;
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
  std::string subarray_key_{};
  ShiftInvariantSubarrayPlan subarray_plan_{};
  std::unordered_set<std::string> seen_geometry_keys_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};
  std::shared_ptr<holoscan::Tensor> d_reference_indices_{};
  std::shared_ptr<holoscan::Tensor> d_shifted_indices_{};

  // URA (2-D) dual-invariance path.
  std::string ura_key_{};
  int64_t ura_ny_{0};
  int64_t ura_nz_{0};
  int64_t ura_y_pairs_{0};
  int64_t ura_z_pairs_{0};
  std::shared_ptr<holoscan::Tensor> d_y_reference_{};
  std::shared_ptr<holoscan::Tensor> d_y_shifted_{};
  std::shared_ptr<holoscan::Tensor> d_z_reference_{};
  std::shared_ptr<holoscan::Tensor> d_z_shifted_{};
};

extern template class ESPRITEstimatorOp<float>;
extern template class ESPRITEstimatorOp<double>;

}  // namespace holoscan::radar
