// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// TODO: rename RadarOp and update port/parameter declarations.
class RadarOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RadarOp)

  RadarOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& /*context*/) override;

 private:
  // TODO: declare parameters, e.g.:
  // Parameter<int> my_param_;
};

}  // namespace holoscan::radar
