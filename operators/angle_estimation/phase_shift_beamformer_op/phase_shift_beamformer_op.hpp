// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// Conventional (Bartlett) dense-path beamformer. Contract reference: ports,
// shapes, dtypes, parameters.
//
// Forms a beam for every range-Doppler cell, replacing the channel axis with
// the (num_az, num_el) beam grid — the Range-Doppler-Angle (RDA) cube that
// CFARDetectorOp detects on directly (dense angle path; angle-then-cluster
// only). This is the *only* dense estimator: a full MVDR/MUSIC cube over every
// RD cell is prohibitive, so the dense slot is conventional beamforming.
//
// In:  rd_maps   — complex64/complex128, (..., num_channels, num_doppler,
//                  num_range_bins). Channel axis immediately before the
//                  (doppler, range) plane; must match array_geometry.
// Out: rda_cube  — complex64/complex128, (..., num_az, num_el, num_doppler,
//                  num_range_bins). Channel axis replaced by the az x el beam
//                  grid (u x v under direction_cosine).
template <typename T>
class PhaseShiftBeamformerOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "PhaseShiftBeamformerOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PhaseShiftBeamformerOp)

  PhaseShiftBeamformerOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Conjugated steering weights, built on device, laid out
  // (num_beams, num_channels) with the beam axis enumerated az-major
  // (beam = ia * num_el + ie) so the GEMM output reshapes directly to the
  // (..., num_az, num_el, D, R) cube. Cached, warn-once on rebuild.
  Parameter<ArrayGeometry> array_geometry_;
  Parameter<AngleConvention> angle_convention_;
  Parameter<std::vector<double>> az_angles_;
  Parameter<std::vector<double>> el_angles_;
  Parameter<WindowSpec> array_taper_;

  ArrayGeometry effective_geometry_{};
  std::vector<double> effective_az_{};
  std::vector<double> effective_el_{};
  WindowSpec effective_taper_{};
  bool seeded_{false};

  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;
};

}  // namespace holoscan::radar
