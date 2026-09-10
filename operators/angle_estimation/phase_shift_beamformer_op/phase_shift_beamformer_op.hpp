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
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "radar_schemas/window_spec_bridge.hpp"

namespace holoscan::radar {

namespace detail {

struct PhaseShiftBeamformerArtifacts {
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> az_grid;
  std::shared_ptr<Tensor> el_grid;
};

}  // namespace detail

// Conventional (Bartlett) dense-path beamformer. Contract reference: ports,
// shapes, dtypes, parameters.
//
// Forms a beam for every range-Doppler cell, replacing the channel axis with
// the (num_az, num_el) beam grid — the Range-Doppler-Angle (RDA) cube that
// CFARDetectorOp detects on directly (dense angle path; angle-then-cluster
// only). This is the *only* dense estimator: a full MVDR/MUSIC cube over every
// RD cell is prohibitive, so the dense slot is conventional beamforming.
//
// In:  rd_maps   — complex64/complex128, (num_channels, num_doppler,
//                  num_range_bins). Channel axis immediately before the
//                  (doppler, range) plane; must match array_geometry.
// Out: rda_cube  — complex64/complex128, (num_az, num_el, num_doppler,
//                  num_range_bins). Channel axis replaced by the az x el beam
//                  grid (u x v under direction_cosine).
// Out: az_grid   — optional float32/float64, (num_az,). Effective scan axis
//                  for this tick, including a matched live override.
// Out: el_grid   — optional float32/float64, (num_el,). Effective scan axis
//                  for this tick, including a matched live override.
//
// Precision is fixed at compile time by the template instantiation.
//
struct PhaseShiftBeamformerParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  std::vector<double> az_angles{};
  std::vector<double> el_angles{};
  WindowSpec array_taper{};
};

HOLOSCAN_PARAM_SCHEMA(
    PhaseShiftBeamformerParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.PhaseShiftBeamformerParams", 1),
    holoscan::field<&PhaseShiftBeamformerParams::array_geometry>(
        "array_geometry")
        .description(
            "Element positions and wavelength; num_elements must match "
            "rd_maps channel axis"),
    holoscan::field<&PhaseShiftBeamformerParams::angle_convention>(
        "angle_convention")
        .description(
            "Boundary AER labeling for scan grid and beam axes; defaults to "
            "radar_boresight (STATIC)"),
    holoscan::field<&PhaseShiftBeamformerParams::az_angles>("az_angles")
        .description(
            "Azimuth scan grid (u grid under direction_cosine)"),
    holoscan::field<&PhaseShiftBeamformerParams::el_angles>("el_angles")
        .description(
            "Elevation scan grid (v grid under direction_cosine); "
            "size-1 yields a 1-D azimuth scan"),
    holoscan::field<&PhaseShiftBeamformerParams::array_taper>("array_taper")
        .description(
            "Amplitude taper across elements for spatial sidelobe control"));

static_assert(
    holoscan::param_schema_availability<PhaseShiftBeamformerParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in PhaseShiftBeamformerParams's "
    "namespace");

template <typename T>
class PhaseShiftBeamformerOp
    : public holoscan::Operator<PhaseShiftBeamformerParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "PhaseShiftBeamformerOp supports float and double "
                "precision");

 public:
  // Required: sizes the rda_cube Tensor pool at setup().
  explicit PhaseShiftBeamformerOp(
      const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> rda_cube{"rda_cube"};
  Output<Tensor> az_grid{"az_grid"};
  Output<Tensor> el_grid{"el_grid"};
  Input<WireArrayGeometry> array_geometry{"array_geometry"};
  Input<Tensor> az_angles{"az_angles"};
  Input<Tensor> el_angles{"el_angles"};
  Input<WireWindowSpec> array_taper{"array_taper"};

 private:
  // Keep the previous activation's input alive across ticks so async
  // device reads cannot outlive the buffer.
  Tensor retained_input_;

  // Conjugated steering weights and their physical axes, built on device and
  // cached by the complete geometry/convention/grid/taper configuration.
  ConfigCache<detail::PhaseShiftBeamformerArtifacts> artifact_cache_;

  // The physical axes are optional/latched outputs.
  LatchedOutput az_grid_out_;
  LatchedOutput el_grid_out_;

  std::int64_t max_az_{0};
  std::int64_t max_el_{0};
  std::int64_t max_doppler_{0};
  std::int64_t max_range_{0};
  std::size_t max_byte_span_{0U};
  std::size_t max_az_grid_byte_span_{0U};
  std::size_t max_el_grid_byte_span_{0U};
};

extern template class PhaseShiftBeamformerOp<float>;
extern template class PhaseShiftBeamformerOp<double>;

}  // namespace holoscan::radar
