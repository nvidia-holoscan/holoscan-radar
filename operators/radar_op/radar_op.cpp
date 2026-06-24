// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_op/radar_op.hpp"

namespace holoscan::radar {

void RadarOp::setup(OperatorSpec& spec) {
  // TODO: declare ports and parameters, e.g.:
  // spec.input<int>("in");
  // spec.output<int>("out");
  // spec.param(my_param_, "my_param", "My Parameter", "Description", 1);
}

void RadarOp::compute(InputContext& op_input, OutputContext& op_output,
                      ExecutionContext& /*context*/) {
  // TODO: implement compute logic, e.g.:
  // auto value = op_input.receive<int>("in").value();
  // op_output.emit(value, "out");
}

}  // namespace holoscan::radar
