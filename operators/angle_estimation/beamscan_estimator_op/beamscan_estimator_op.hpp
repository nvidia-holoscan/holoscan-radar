// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

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
  std::vector<double> az_angles{};
  std::vector<double> el_angles{};
  WindowSpec array_taper{};
  PeakInterpolation peak_interpolation{PeakInterpolation::kNone};
};

HOLOSCAN_PARAM_SCHEMA(
    BeamscanParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.BeamscanParams", 1),
    holoscan::field<&BeamscanParams::array_geometry>("array_geometry")
        .description(
            "Element positions and wavelength; num_elements must match "
            "snapshot channels"),
    holoscan::field<&BeamscanParams::angle_convention>("angle_convention")
        .description(
            "Static AER labeling for the scan grid and DOA output "
            "(no live port)"),
    holoscan::field<&BeamscanParams::az_angles>("az_angles")
        .description(
            "Azimuth scan grid (or u under direction_cosine); required "
            "non-empty"),
    holoscan::field<&BeamscanParams::el_angles>("el_angles")
        .description(
            "Elevation scan grid (or v under direction_cosine); required "
            "non-empty"),
    holoscan::field<&BeamscanParams::array_taper>("array_taper")
        .description("Amplitude taper across array elements"),
    holoscan::field<&BeamscanParams::peak_interpolation>("peak_interpolation")
        .description(
            "Sub-grid peak refinement {none, parabolic}"));

static_assert(holoscan::param_schema_availability<BeamscanParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in BeamscanParams's namespace");

// Admission bounds for the device output pools.
inline constexpr std::uint32_t kBeamscanEstimatorMaxRank = 8U;
// 256 MiB
inline constexpr std::size_t kBeamscanEstimatorMaxByteSpan = 256ULL << 20;

template <typename T>
class BeamscanEstimatorOp : public holoscan::Operator<BeamscanParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BeamscanEstimatorOp supports float and double precision");

 public:
  BeamscanEstimatorOp() = default;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> snapshot{"snapshot"};
  // Optional; unconnected graphs omit it from the temporal contract.
  Input<Tensor> is_valid{"is_valid"};
  Output<Tensor> aoa_estimates{"aoa_estimates"};
  Output<Tensor> angle_spectrum{"angle_spectrum"};
  // [live] sticky override of params().peak_interpolation (OPERATOR_API.md).
  // Vector / descriptor live params (az_angles, el_angles, array_geometry,
  // array_taper) are Params-only on HSDK 5 EA — see OPEN_ITEMS.md /
  // HSDK5_PORT.md §9. angle_convention is static (Params only).
  Input<PeakInterpolation> peak_interpolation{"peak_interpolation"};

 private:
  // Keep the previous activation's inputs alive across ticks so async device
  // reads cannot outlive the buffers.
  Tensor retained_snapshot_;
  Tensor retained_is_valid_;
  // Sticky [live] peak_interpolation: last override wins until the next.
  std::optional<PeakInterpolation> sticky_peak_interpolation_;
  // Conjugate-transposed steering weights Wᶜ, built on device, laid out
  // (num_channels, num_beams) with beams el-major. Cached; warn-once on
  // rebuild.
  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;
};

extern template class BeamscanEstimatorOp<float>;
extern template class BeamscanEstimatorOp<double>;

}  // namespace holoscan::radar
