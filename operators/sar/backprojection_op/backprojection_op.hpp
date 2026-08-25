// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"

namespace holoscan::radar {

namespace detail {
template <typename T>
struct BackprojectionState;
}  // namespace detail

// Time-domain SAR image formation. Accumulating across ticks — emits only
// when a control message says so. This header is the contract reference
// for ports, shapes, dtypes and parameters; the locked contract is
// docs/planning/SAR/OPERATOR_API.md.
//
// In:  range_profiles      complex, (..., num_pulses, num_range_bins)
// In:  positions           real,    (..., num_pulses, 3)
// In:  reference_position  real,    (..., num_pulses, 3)
// In:  control             uint8,   (1,)  required connection, optional
//                                      delivery (non-blocking)
// In:  range_bins          optional real (num_range_bins,)
// In:  sample_points       optional real (num_v, num_u, 3)
// In:  weight              optional real (..., num_pulses)
// In:  initial_image       optional; rejected this slice
// Out: image               complex, (..., num_v, num_u) on emit only
// Out: sample_geometry     optional SampleGeometry, grid mode on emit
//
// Precision is fixed at compile time (T = float | double).
template <typename T>
class BackprojectionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BackprojectionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BackprojectionOp)

  BackprojectionOp() = default;
  ~BackprojectionOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<CoordinateFrame> coordinate_frame_;
  Parameter<SampleGeometry> sample_geometry_;
  Parameter<double> range_bin_spacing_;
  Parameter<double> range_bin_origin_;
  Parameter<double> reference_frequency_;
  Parameter<RangeInterpolation> interpolation_;
  Parameter<BackprojectionBackend> backend_;
  Parameter<GeometryChange> geometry_change_;

  CoordinateFrame effective_frame_{};
  SampleGeometry effective_geometry_{};
  std::optional<double> effective_spacing_{};
  std::optional<double> effective_origin_{};
  double effective_frequency_{0.0};
  RangeInterpolation effective_interpolation_{RangeInterpolation::kLinear};
  BackprojectionBackend backend_value_{kDefaultBackprojectionBackend};
  GeometryChange geometry_change_value_{GeometryChange::kReset};

  bool sample_points_connected_{false};
  bool range_bins_connected_{false};
  bool control_connected_{false};

  int64_t unemitted_pulses_{0};
  bool locality_warned_{false};

  std::shared_ptr<detail::BackprojectionState<T>> state_;
  std::shared_ptr<holoscan::Tensor> retained_profiles_;
  std::shared_ptr<holoscan::Tensor> retained_positions_;
  std::shared_ptr<holoscan::Tensor> retained_reference_;
  std::shared_ptr<holoscan::Tensor> retained_sample_points_;
  std::shared_ptr<holoscan::Tensor> retained_weight_;
};

extern template class BackprojectionOp<float>;
extern template class BackprojectionOp<double>;

}  // namespace holoscan::radar
