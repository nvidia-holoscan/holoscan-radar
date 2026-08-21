// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// Retire MatX eig/inv plan caches while CUDA is still alive. Implemented
// in the .cu (MatX headers stay out of the public .hpp / pybind TU).
// Prefer install_matx_cache_atexit() from Python after importing cupy.
void retire_matx_solver_caches();

// Adaptive (MVDR / Capon) sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md → "MVDREstimatorOp".
//
// For each item, forms the sample covariance R̂ = (1/K) Σ_k x_k·x_kᴴ over its
// valid snapshots, applies diagonal loading (R + δI), inverts it, and scans the
// (az, el) grid for the Capon power P(θ) = 1 / (a(θ)ᴴ R̂⁻¹ a(θ)); the argmax is
// the reported DOA. The steering manifold a(θ) is the untapered array response
// (uniform weighting) — MVDR shapes the beam adaptively from the data, so no
// array taper parameter.
//
// In:  snapshot        — complex64/complex128, (num_items, num_channels) or
//                       (num_items, num_channels, num_snapshots). The 2-D form
//                       is treated as num_snapshots == 1 (rank-1 covariance —
//                       relies on diagonal loading for invertibility).
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots). When
//                       present the covariance sums only over marked slots (so
//                       the effective K is the per-item valid count); absent →
//                       all num_snapshots are used. Omitted from the temporal
//                       contract when unconnected (HSDK5_PORT §1).
// Out: aoa_estimates   — float32/float64, (num_items, 2).
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el). Always
//                       emitted (optional in OPERATOR_API.md; see OPEN_ITEMS).
// Out: doa_multi       — float32/float64, (num_items, max_sources, 2). Always
//                       emitted.
// Out: source_count    — int32, (num_items,). Always emitted.
// Out: eigenvalues     — float32/float64, (num_items, num_channels). Always
//                       emitted (computed every tick).
//
// [EA HSDK5 ONLY] Live override ports: sticky Input for POD scalars
// (diagonal_loading, max_sources). Descriptor / unbounded-vector live params
// (array_geometry, az_angles, el_angles, spatial_smoothing) are Params-only
// (HSDK5_PORT.md §9). angle_convention and forward_backward stay Params-only
// (STATIC / not [live]).

struct MVDRParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  std::vector<double> az_angles{};
  std::vector<double> el_angles{};
  // Negative → auto (δ ~ trace(R̂)/C); non-negative → absolute δ. Default -1.
  double diagonal_loading{-1.0};
  int64_t max_sources{1};
  // Empty → off; [P] ULA; [P_az, P_el] URA. Params-only on EA (§9).
  std::vector<int64_t> spatial_smoothing{};
  bool forward_backward{true};
  bool expect_diagonal_loading{false};
  bool expect_max_sources{false};
  bool sticky_diagonal_loading{true};
  bool sticky_max_sources{true};
};

HOLOSCAN_PARAM_SCHEMA(
    MVDRParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.MVDRParams", 1),
    holoscan::field<&MVDRParams::array_geometry>("array_geometry")
        .description(
            "Element positions and wavelength; num_elements must match "
            "snapshot channels"),
    holoscan::field<&MVDRParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the scan grid and DOA output "
            "(no live port)"),
    holoscan::field<&MVDRParams::az_angles>("az_angles")
        .description(
            "Azimuth scan grid (or u under direction_cosine); required "
            "non-empty"),
    holoscan::field<&MVDRParams::el_angles>("el_angles")
        .description(
            "Elevation scan grid (or v under direction_cosine); required "
            "non-empty"),
    holoscan::field<&MVDRParams::diagonal_loading>("diagonal_loading")
        .description(
            "R + delta*I regularization; negative selects auto"),
    holoscan::field<&MVDRParams::max_sources>("max_sources")
        .description(
            "Capacity of doa_multi/source_count — Capon peaks to report"),
    holoscan::field<&MVDRParams::spatial_smoothing>("spatial_smoothing")
        .description(
            "Per-aperture-axis sub-array counts on a filled ULA/URA "
            "(empty = off)"),
    holoscan::field<&MVDRParams::forward_backward>("forward_backward")
        .description(
            "Include conjugate-reversed looks when spatial_smoothing is set"),
    holoscan::field<&MVDRParams::expect_diagonal_loading>(
        "expect_diagonal_loading")
        .description(
            "Set when the application wires the diagonal_loading override."),
    holoscan::field<&MVDRParams::expect_max_sources>("expect_max_sources")
        .description(
            "Set when the application wires the max_sources override."),
    holoscan::field<&MVDRParams::sticky_diagonal_loading>(
        "sticky_diagonal_loading")
        .description(
            "Keep the last Exact diagonal_loading match on a miss. Ignored "
            "unless expect_diagonal_loading."),
    holoscan::field<&MVDRParams::sticky_max_sources>("sticky_max_sources")
        .description(
            "Keep the last Exact max_sources match on a miss. Ignored "
            "unless expect_max_sources."));

static_assert(holoscan::param_schema_availability<MVDRParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in MVDRParams's namespace");

template <typename T>
class MVDREstimatorOp : public holoscan::Operator<MVDRParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MVDREstimatorOp supports float and double precision");

 public:
  explicit MVDREstimatorOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;
  // Drain MatX solver plan caches + the global MemTracker once at graph stop,
  // while CUDA is still valid (per-tick only clears plan caches; see
  // mvdr_stage.hpp). HSDK 5 still exposes OperatorBase::stop().
  void stop() override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional; unconnected graphs omit it from the temporal contract.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> angle_spectrum{"angle_spectrum"};
  Output<Tensor> doa_multi{"doa_multi"};
  Output<Tensor> source_count{"source_count"};
  Output<Tensor> eigenvalues{"eigenvalues"};
  // [live] sticky POD overrides (OPERATOR_API.md). Vector / descriptor live
  // params stay Params-only on HSDK 5 EA — see OPEN_ITEMS.md / HSDK5_PORT §9.
  Input<double> diagonal_loading{"diagonal_loading"};
  Input<int64_t> max_sources{"max_sources"};

 private:
  MaxTensorExtents extents_;
  std::size_t spectrum_max_byte_span_{0U};
  std::size_t aoa_max_byte_span_{0U};
  std::size_t doa_max_byte_span_{0U};
  std::size_t count_max_byte_span_{0U};
  std::size_t eig_max_byte_span_{0U};

  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;
  // Sticky [live] scalars: last override wins until the next.
  std::optional<double> sticky_diagonal_loading_;
  std::optional<int64_t> sticky_max_sources_;

  // Untapered steering manifold a(θ) (num_beams, num_channels), beam el-major
  // (beam = ie * num_az + ia) to match the Capon reduce kernel's decode. Built
  // against the reduced sub-array geometry when smoothing is on. Cached,
  // warn-once on rebuild (geometry / scan-grid / smoothing change).
  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;

  // Spatial-smoothing plan + uploaded index tables, rebuilt only when the
  // geometry / smoothing config changes (keyed by smoothing_key_).
  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};
};

extern template class MVDREstimatorOp<float>;
extern template class MVDREstimatorOp<double>;

}  // namespace holoscan::radar
