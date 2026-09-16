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
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {

struct BeamscanRuntimeState;

// Conventional (Bartlett) sparse-path DOA estimator. This header is the
// contract reference for the operator's ports, shapes, dtypes, and
// parameters; module-wide conventions live in operators/common/README.md.
//
// In:  snapshot        — complex64/complex128, (num_items, num_channels) or
//                       (num_items, num_channels, 1).
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots);
//                       validated when present (num_snapshots is 1 for the
//                       2-D snapshot form).
// Out: aoa_estimates   — float32/float64, (num_items, 2).
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el).
//
// Precision is fixed at compile time by the template instantiation.

struct BeamscanParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  std::vector<double> az_coordinates{};
  std::vector<double> el_coordinates{};
  WindowSpec array_taper{};
  PeakInterpolation peak_interpolation{PeakInterpolation::kNone};
};

HOLOSCAN_PARAM_SCHEMA(
    BeamscanParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.BeamscanParams", 2),
    holoscan::field<&BeamscanParams::array_geometry>("array_geometry")
        .description(
            "Element positions and wavelength; num_elements must match "
            "snapshot channels"),
    holoscan::field<&BeamscanParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the scan grid and DOA output "
            "(default radar_boresight; no live port)"),
    holoscan::field<&BeamscanParams::az_coordinates>("az_coordinates")
        .description(
            "First scan-coordinate grid: azimuth radians for angular "
            "conventions, or dimensionless u under direction_cosine; "
            "required non-empty"),
    holoscan::field<&BeamscanParams::el_coordinates>("el_coordinates")
        .description(
            "Second scan-coordinate grid: elevation radians for angular "
            "conventions, or dimensionless v under direction_cosine; "
            "required non-empty"),
    holoscan::field<&BeamscanParams::array_taper>("array_taper")
        .description("Amplitude taper across array elements"),
    holoscan::field<&BeamscanParams::peak_interpolation>("peak_interpolation")
        .description(
            "Sub-grid peak refinement {none, parabolic}"));

static_assert(holoscan::param_schema_availability<BeamscanParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in BeamscanParams's namespace");

template <typename T>
class BeamscanEstimatorOp : public holoscan::Operator<BeamscanParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BeamscanEstimatorOp supports float and double precision");

 public:
  // Required: sizes the output Tensor pools at setup().
  explicit BeamscanEstimatorOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional same-tick mask; Exact + SafeDefault admits an unwired port.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> angle_spectrum{"angle_spectrum"};
  Input<WireArrayGeometry> array_geometry{"array_geometry"};
  Input<Tensor> az_coordinates{"az_coordinates"};
  Input<Tensor> el_coordinates{"el_coordinates"};
  Input<WireWindowSpec> array_taper{"array_taper"};
  Input<PeakInterpolation> peak_interpolation{"peak_interpolation"};

 private:
  // Keep the previous activation's inputs alive across ticks so async
  // device reads cannot outlive the buffers.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;

  // Constructor-created steering cache and bounded GEMM workspace.
  std::shared_ptr<BeamscanRuntimeState> state_;

  std::int64_t max_items_{0};
  std::int64_t max_az_{0};
  std::int64_t max_el_{0};
  std::size_t spectrum_max_byte_span_{0U};
  std::size_t aoa_max_byte_span_{0U};
};

extern template class BeamscanEstimatorOp<float>;
extern template class BeamscanEstimatorOp<double>;

}  // namespace holoscan::radar
