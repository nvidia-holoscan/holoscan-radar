// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// Conventional (Bartlett) sparse-path DOA estimator. Contract reference:
// ports, shapes, dtypes, parameters.
//
// In:  snapshot        — complex64/complex128, (num_items, num_channels) or
//                       (num_items, num_channels, 1).
// In:  is_valid        — optional bool, (num_items, num_snapshots); ignored
//                       unless num_snapshots == 1.
// Out: aoa_estimates   — float32/float64, (num_items, 2).
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el).
template <typename T>
class BeamscanEstimatorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BeamscanEstimatorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BeamscanEstimatorOp)

  BeamscanEstimatorOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Conjugate-transposed steering weights Wᶜ, built on device, laid out
  // (num_channels, num_beams) with beams el-major (beam = ie * num_az + ia) to
  // match the reduce kernel's beam decode. Cached, warn-once on rebuild.
  Parameter<ArrayGeometry> array_geometry_;
  Parameter<AngleConvention> angle_convention_;
  Parameter<std::vector<double>> az_angles_;
  Parameter<std::vector<double>> el_angles_;
  Parameter<WindowSpec> array_taper_;
  Parameter<PeakInterpolation> peak_interpolation_;

  ArrayGeometry effective_geometry_{};
  std::vector<double> effective_az_{};
  std::vector<double> effective_el_{};
  WindowSpec effective_taper_{};
  PeakInterpolation effective_peak_interp_{PeakInterpolation::kNone};
  bool seeded_{false};

  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;
};

}  // namespace holoscan::radar
