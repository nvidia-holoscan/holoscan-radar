// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Static-clutter / zero-Doppler suppression along slow-time; sits between
// the range and Doppler stages. This header is the contract reference for
// the operator's ports, shapes, dtypes, and parameters; module-wide
// conventions live in operators/common/README.md.
//
// In:  range_profiles — complex64/complex128,
//                       (..., num_chirps, num_range_bins).
// Out: range_profiles — complex64/complex128,
//                       (..., num_chirps_out, num_range_bins), where
//                       num_chirps_out = num_chirps (mean_subtraction),
//                       num_chirps - 1 (single_delay), or
//                       num_chirps - 2 (double_delay).
//
// Precision is fixed at compile time by the template instantiation.
template <typename T>
class ClutterFilterOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ClutterFilterOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ClutterFilterOp)

  ClutterFilterOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Overridable; mean_subtraction nulls zero-Doppler with no chirp loss, the
  // delay cancellers are 1st/2nd-order MTI high-pass filters. kNone is
  // rejected (it exists for the fused operator's inline stage only).
  Parameter<ClutterFilterMethod> method_;

  // Stream-ordered lifetime guard for the previous tick's input.
  std::shared_ptr<holoscan::Tensor> retained_input_;
};

// Explicit instantiations live in clutter_filter_op.cu.
extern template class ClutterFilterOp<float>;
extern template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
