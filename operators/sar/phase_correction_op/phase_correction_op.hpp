// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <type_traits>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Per-pulse phase applied to range profiles ahead of product formation
// (D20's second half). Broadcast multiply: subtract is exp(−iφ), add is
// exp(+iφ). This header is the contract reference; the locked text is
// docs/planning/SAR/OPERATOR_API.md → PhaseCorrectionOp.
//
// In:  range_profiles   complex64/complex128,
//                       (..., num_pulses, num_range_bins). Required.
// In:  phase_correction float32/float64, (..., num_pulses), radians.
//                       Required (blocking); co-arrives this tick.
// Out: range_profiles   same shape and dtype as the input.
//
// sense is Static and required (no default). Precision is the template
// instantiation (T = float → complex64/float32, T = double →
// complex128/float64).
template <typename T>
class PhaseCorrectionOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "PhaseCorrectionOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PhaseCorrectionOp)

  PhaseCorrectionOp() = default;
  ~PhaseCorrectionOp() override = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<PhaseCorrectionSense> sense_;
  PhaseCorrectionSense sense_value_{PhaseCorrectionSense::kUnspecified};

  // Stream-ordered lifetime of this tick's inputs.
  std::shared_ptr<holoscan::Tensor> retained_profiles_;
  std::shared_ptr<holoscan::Tensor> retained_phase_;
};

extern template class PhaseCorrectionOp<float>;
extern template class PhaseCorrectionOp<double>;

}  // namespace holoscan::radar
