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
struct MVDRRuntimeState;

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
// In:  snapshot        — complex64/complex128,
//                       (num_items, num_channels, num_snapshots).
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots). When
//                       present the covariance sums only over marked slots (so
//                       the effective K is the per-item valid count); absent →
//                       all num_snapshots are used. Exact + SafeDefault admits
//                       an unwired port; publish a mask before its snapshot.
// Out: aoa_estimates   — float32/float64, (num_items, 2).
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el). Always
//                       emitted (optional in OPERATOR_API.md).
// Out: doa_multi       — float32/float64, (num_items, max_sources, 2). Always
//                       emitted.
// Out: source_count    — int32, (num_items,). Always emitted.
// Out: eigenvalues     — float32/float64, (num_items, num_channels). Always
//                       emitted (computed every tick).
//
// [live] ports join snapshot with Exact + SafeDefault. angle_convention
// and forward_backward stay static.

struct MVDRParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  std::vector<double> az_coordinates{};
  std::vector<double> el_coordinates{};
  // Negative → auto (δ ~ trace(R̂)/C); non-negative → absolute δ. Default -1.
  double diagonal_loading{-1.0};
  int64_t max_sources{1};
  // Empty → off; [P] ULA; [P_az, P_el] URA.
  std::vector<int64_t> spatial_smoothing{};
  bool forward_backward{true};
};

HOLOSCAN_PARAM_SCHEMA(
    MVDRParams, holoscan::param_schema("com.nvidia.holoscan.radar.MVDRParams", 2),
    holoscan::field<&MVDRParams::array_geometry>("array_geometry")
        .description("Element positions and wavelength; num_elements must match "
                     "snapshot channels"),
    holoscan::field<&MVDRParams::angle_convention>("angle_convention")
        .description("Static AER labeling for the scan grid and DOA output "
                     "(default radar_boresight; no live port)"),
    holoscan::field<&MVDRParams::az_coordinates>("az_coordinates")
        .description("First scan-coordinate grid: azimuth radians for angular "
                     "conventions, or dimensionless u under direction_cosine; "
                     "required non-empty"),
    holoscan::field<&MVDRParams::el_coordinates>("el_coordinates")
        .description("Second scan-coordinate grid: elevation radians for angular "
                     "conventions, or dimensionless v under direction_cosine; "
                     "required non-empty"),
    holoscan::field<&MVDRParams::diagonal_loading>("diagonal_loading")
        .description("R + delta*I regularization; negative selects auto"),
    holoscan::field<&MVDRParams::max_sources>("max_sources")
        .description("Capacity of doa_multi/source_count — Capon peaks to report"),
    holoscan::field<&MVDRParams::spatial_smoothing>("spatial_smoothing")
        .description("Per-aperture-axis sub-array counts on a filled ULA/URA "
                     "(empty = off)"),
    holoscan::field<&MVDRParams::forward_backward>("forward_backward")
        .description("Include conjugate-reversed looks when spatial_smoothing is set"));

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
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  // Drain MatX solver plan caches + the global MemTracker once at graph stop,
  // while CUDA and the stable operator-owned solver buffers are still valid.
  // HSDK 5 still exposes OperatorBase::stop().
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
  Input<Tensor> az_coordinates{"az_coordinates"};
  Input<Tensor> el_coordinates{"el_coordinates"};
  Input<SpatialSmoothingOverride> spatial_smoothing{"spatial_smoothing"};
  Input<double> diagonal_loading{"diagonal_loading"};
  Input<int64_t> max_sources{"max_sources"};

 private:
  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;

  // Latched optional outputs (OPERATOR_API.md). doa_multi and source_count
  // come from one peak-extraction pass, which stops only once both latch.
  LatchedOutput doa_multi_out_;
  LatchedOutput source_count_out_;
  LatchedOutput eigenvalues_out_;

  std::shared_ptr<MVDRRuntimeState<T>> state_;

  MaxTensorExtents extents_;
  std::size_t spectrum_max_byte_span_{0U};
  std::size_t aoa_max_byte_span_{0U};
  std::size_t doa_max_byte_span_{0U};
  std::size_t count_max_byte_span_{0U};
  std::size_t eig_max_byte_span_{0U};
};

extern template class MVDREstimatorOp<float>;
extern template class MVDREstimatorOp<double>;

}  // namespace holoscan::radar
