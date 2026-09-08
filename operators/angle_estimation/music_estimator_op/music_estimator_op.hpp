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

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

template <typename T>
struct MUSICRuntimeState;

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
// In:  snapshot        — complex64/complex128,
//                       (num_items, num_channels, num_snapshots).
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots).
//                       When present the covariance sums only over marked
//                       slots (effective K is the per-item valid count);
//                       absent → all num_snapshots are used. Exact +
//                       SafeDefault admits an unwired port; publish a mask
//                       before its snapshot.
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
// [live] ports join snapshot with Exact + SafeDefault. angle_convention
// and forward_backward stay static.

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
            "(default radar_boresight; no live port)"),
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
            "set"));

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
  // Optional same-tick mask; Exact + SafeDefault admits an unwired port.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> angle_spectrum{"angle_spectrum"};
  Output<Tensor> doa_multi{"doa_multi"};
  Output<Tensor> source_count{"source_count"};
  Output<Tensor> eigenvalues{"eigenvalues"};
  // [live] Exact overrides.
  Input<WireArrayGeometry> array_geometry{"array_geometry"};
  Input<Tensor> az_angles{"az_angles"};
  Input<Tensor> el_angles{"el_angles"};
  Input<SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  Input<int64_t> num_sources{"num_sources"};
  Input<double> source_count_pfa{"source_count_pfa"};
  Input<int64_t> max_sources{"max_sources"};

 private:
  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;

  // Latched optional outputs (OPERATOR_API.md). Both come from one
  // peak-extraction pass, which stops only once both latch.
  LatchedOutput doa_multi_out_;
  LatchedOutput source_count_out_;

  bool warned_rank_deficient_{false};

  std::shared_ptr<MUSICRuntimeState<T>> state_;

  MaxTensorExtents extents_;
  std::size_t spectrum_max_byte_span_{0U};
  std::size_t aoa_max_byte_span_{0U};
  std::size_t doa_max_byte_span_{0U};
  std::size_t count_max_byte_span_{0U};
  std::size_t eig_max_byte_span_{0U};
};

extern template class MUSICEstimatorOp<float>;
extern template class MUSICEstimatorOp<double>;

}  // namespace holoscan::radar
