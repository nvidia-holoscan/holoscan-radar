// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Terminal and null sinks for the performance harness.
//
//  - PerfTerminalSinkOp drains one output port and records an end timestamp
//    after a full device sync, so the recorded latency includes all GPU work
//    the operator enqueued for that frame.
//  - PerfNullSinkOp drains the remaining dense output ports as raw GXF entities
//    (no Tensor deserialization), so the bool detection_mask port does not
//    trip the SDK's unsupported-DLPack-bool error and no host copies are made.

#pragma once

#include <cuda_runtime_api.h>

#include <holoscan/holoscan.hpp>

#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

class PerfTerminalSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfTerminalSinkOp)
  PerfTerminalSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)op_input.receive<gxf::Entity>("input");
    cudaDeviceSynchronize();
    if (collector_ != nullptr) {
      collector_->ends.push_back(Clock::now());
    }
  }

  void configure(TimingCollector* collector) { collector_ = collector; }

 private:
  TimingCollector* collector_{nullptr};
};

class PerfNullSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfNullSinkOp)
  PerfNullSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)op_input.receive<gxf::Entity>("input");
  }
};

}  // namespace holoscan::radar::perf
