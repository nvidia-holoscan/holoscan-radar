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
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

// Retire MatX eig plan caches while CUDA is still alive. Implemented in the
// .cu (MatX headers stay out of the public .hpp / pybind TU). Prefer
// install_matx_cache_atexit() from Python after importing cupy.
void retire_music_matx_caches();

// Subspace (MUSIC) sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md → "MUSICEstimatorOp".
//
// For each item, forms the sample covariance R_i = (1/K) Σ_k x_k x_kᴴ over
// its valid snapshots, eigendecomposes it (ascending eigenvalues, columns of
// eigenvectors), resolves a per-item source count d_i (fixed or per-item EFT
// on the eigenvalues), and scans the (az, el) grid for the MUSIC
// pseudospectrum P(θ) = 1 / (a(θ)ᴴ P_n a(θ)), where P_n is the projector
// onto the noise subspace (the num_channels - d_i smallest eigenvectors);
// the argmax is the reported DOA. The steering manifold a(θ) is the
// untapered array response (uniform weighting).
//
// In:  snapshot        — complex64/complex128, (num_items, num_channels) or
//                       (num_items, num_channels, num_snapshots). The 2-D
//                       form is treated as num_snapshots == 1.
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots).
//                       When present the covariance sums only over marked
//                       slots (effective K is the per-item valid count);
//                       absent → all num_snapshots are used. Omitted from
//                       the temporal contract when unconnected
//                       (HSDK5_PORT §1).
// Out: aoa_estimates   — float32/float64, (num_items, 2).
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el).
//                       Always emitted (optional in OPERATOR_API.md; see
//                       OPEN_ITEMS).
// Out: doa_multi       — float32/float64, (num_items, max_sources, 2).
//                       Always emitted.
// Out: source_count    — int32, (num_items,). Always emitted.
// Out: eigenvalues     — float32/float64, (num_items, num_channels). Always
//                       emitted (computed every tick).
//
// [live] ports join snapshot with Exact + SafeDefault and are gated by their
// expect_* Params. angle_convention and forward_backward stay static.

struct MUSICParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  std::vector<double> az_angles{};
  std::vector<double> el_angles{};
  // Signal-subspace dimension. Negative → auto (per-item EFT); non-negative
  // → fixed order. Default -1 (auto).
  int64_t num_sources{-1};
  // EFT over-estimation (false-alarm) probability for num_sources: auto.
  double source_count_pfa{1e-2};
  // Capacity of doa_multi / source_count, and the K_max cap on auto's EFT
  // search. Default 4.
  int64_t max_sources{4};
  // Empty → off; [P] ULA; [P_az, P_el] URA.
  std::vector<int64_t> spatial_smoothing{};
  bool forward_backward{true};
  bool expect_array_geometry{false};
  bool expect_az_angles{false};
  bool expect_el_angles{false};
  bool expect_spatial_smoothing{false};
  bool expect_num_sources{false};
  bool expect_source_count_pfa{false};
  bool expect_max_sources{false};
  bool sticky_array_geometry{true};
  bool sticky_az_angles{true};
  bool sticky_el_angles{true};
  bool sticky_spatial_smoothing{true};
  bool sticky_num_sources{true};
  bool sticky_source_count_pfa{true};
  bool sticky_max_sources{true};
};

HOLOSCAN_PARAM_SCHEMA(
    MUSICParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.MUSICParams", 1),
    holoscan::field<&MUSICParams::array_geometry>("array_geometry")
        .description(
            "Element positions and wavelength; num_elements must match "
            "snapshot channels"),
    holoscan::field<&MUSICParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the scan grid and DOA output "
            "(no live port)"),
    holoscan::field<&MUSICParams::az_angles>("az_angles")
        .description(
            "Azimuth scan grid (or u under direction_cosine); required "
            "non-empty"),
    holoscan::field<&MUSICParams::el_angles>("el_angles")
        .description(
            "Elevation scan grid (or v under direction_cosine); required "
            "non-empty"),
    holoscan::field<&MUSICParams::num_sources>("num_sources")
        .description(
            "Signal-subspace dimension; negative selects auto (per-item "
            "EFT on the covariance eigenvalues)"),
    holoscan::field<&MUSICParams::source_count_pfa>("source_count_pfa")
        .description(
            "EFT over-estimation probability for num_sources: auto"),
    holoscan::field<&MUSICParams::max_sources>("max_sources")
        .description(
            "Capacity of doa_multi/source_count, and the K_max cap on "
            "num_sources: auto"),
    holoscan::field<&MUSICParams::spatial_smoothing>("spatial_smoothing")
        .description(
            "Per-aperture-axis sub-array counts on a filled ULA/URA "
            "(empty = off)"),
    holoscan::field<&MUSICParams::forward_backward>("forward_backward")
        .description(
            "Include conjugate-reversed looks when spatial_smoothing is "
            "set"),
    holoscan::field<&MUSICParams::expect_array_geometry>(
        "expect_array_geometry")
        .description(
            "Set when the application wires the array_geometry override."),
    holoscan::field<&MUSICParams::expect_az_angles>("expect_az_angles")
        .description(
            "Set when the application wires the az_angles override."),
    holoscan::field<&MUSICParams::expect_el_angles>("expect_el_angles")
        .description(
            "Set when the application wires the el_angles override."),
    holoscan::field<&MUSICParams::expect_spatial_smoothing>(
        "expect_spatial_smoothing")
        .description(
            "Set when the application wires the spatial_smoothing override."),
    holoscan::field<&MUSICParams::expect_num_sources>("expect_num_sources")
        .description(
            "Set when the application wires the num_sources override."),
    holoscan::field<&MUSICParams::expect_source_count_pfa>(
        "expect_source_count_pfa")
        .description(
            "Set when the application wires the source_count_pfa override."),
    holoscan::field<&MUSICParams::expect_max_sources>("expect_max_sources")
        .description(
            "Set when the application wires the max_sources override."),
    holoscan::field<&MUSICParams::sticky_array_geometry>(
        "sticky_array_geometry")
        .description(
            "Keep the last valid Exact array_geometry match on a miss."),
    holoscan::field<&MUSICParams::sticky_az_angles>("sticky_az_angles")
        .description(
            "Keep the last valid Exact az_angles match on a miss."),
    holoscan::field<&MUSICParams::sticky_el_angles>("sticky_el_angles")
        .description(
            "Keep the last valid Exact el_angles match on a miss."),
    holoscan::field<&MUSICParams::sticky_spatial_smoothing>(
        "sticky_spatial_smoothing")
        .description(
            "Keep the last valid Exact spatial_smoothing match on a miss."),
    holoscan::field<&MUSICParams::sticky_num_sources>("sticky_num_sources")
        .description(
            "Keep the last Exact num_sources match on a miss. Ignored "
            "unless expect_num_sources."),
    holoscan::field<&MUSICParams::sticky_source_count_pfa>(
        "sticky_source_count_pfa")
        .description(
            "Keep the last Exact source_count_pfa match on a miss. Ignored "
            "unless expect_source_count_pfa."),
    holoscan::field<&MUSICParams::sticky_max_sources>("sticky_max_sources")
        .description(
            "Keep the last Exact max_sources match on a miss. Ignored "
            "unless expect_max_sources."));

static_assert(holoscan::param_schema_availability<MUSICParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in MUSICParams's namespace");

template <typename T>
class MUSICEstimatorOp : public holoscan::Operator<MUSICParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MUSICEstimatorOp supports float and double precision");

 public:
  explicit MUSICEstimatorOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;
  // Drain MatX eig plan caches + the global MemTracker once at graph stop,
  // while CUDA is still valid (per-tick only clears plan caches; see
  // music_stage.hpp). HSDK 5 still exposes OperatorBase::stop().
  void stop() override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional; unconnected graphs omit it from the temporal contract.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> angle_spectrum{"angle_spectrum"};
  Output<Tensor> doa_multi{"doa_multi"};
  Output<Tensor> source_count{"source_count"};
  Output<Tensor> eigenvalues{"eigenvalues"};
  // [live] Exact overrides, gated by expect_*.
  Input<WireArrayGeometry> array_geometry{"array_geometry"};
  Input<Tensor> az_angles{"az_angles"};
  Input<Tensor> el_angles{"el_angles"};
  Input<SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  Input<int64_t> num_sources{"num_sources"};
  Input<double> source_count_pfa{"source_count_pfa"};
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
  // Sticky [live] state stores logical values, never wire-owned storage.
  std::optional<ArrayGeometry> sticky_array_geometry_;
  std::optional<std::vector<double>> sticky_az_angles_;
  std::optional<std::vector<double>> sticky_el_angles_;
  std::optional<std::vector<int64_t>> sticky_spatial_smoothing_;
  std::optional<int64_t> sticky_num_sources_;
  std::optional<double> sticky_source_count_pfa_;
  std::optional<int64_t> sticky_max_sources_;
  bool warned_rank_deficient_{false};

  // Untapered steering manifold a(θ) (num_beams, num_channels), beam el-major
  // (beam = ie * num_az + ia). Built against the reduced sub-array geometry
  // when smoothing is on. Cached, warn-once on rebuild (geometry / scan-grid
  // / smoothing change).
  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;

  // Spatial-smoothing plan + uploaded index tables, rebuilt only when the
  // geometry / smoothing config changes (keyed by smoothing_key_).
  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};
};

extern template class MUSICEstimatorOp<float>;
extern template class MUSICEstimatorOp<double>;

}  // namespace holoscan::radar
