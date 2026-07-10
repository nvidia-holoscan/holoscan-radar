// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Timestamping source operators for the performance harness.
//
//  - PerfDataSourceOp<T> emits one preallocated power-map frame per tick and
//    records a start timestamp into the shared TimingCollector.
//  - PerfOverrideSourceOp<V> emits a single [live] override value on one
//    chosen tick, then stays silent (sticky-override semantics), so a
//    scenario can measure the reconfiguration cost on a known frame.

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

// Emits the same preallocated TensorMap every tick (no per-frame H2D), so the
// measured latency is dominated by the operator under test, not the source.
class PerfDataSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfDataSourceOp)
  PerfDataSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("rd_maps");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    const auto now = Clock::now();
    if (collector_ != nullptr) {
      if (!collector_->first_tick_set) {
        collector_->t_first_tick = now;
        collector_->first_tick_set = true;
      }
      collector_->starts.push_back(now);
    }
    op_output.emit(frame_, "rd_maps");
  }

  void configure(TimingCollector* collector, holoscan::TensorMap frame) {
    collector_ = collector;
    frame_ = std::move(frame);
  }

 private:
  TimingCollector* collector_{nullptr};
  holoscan::TensorMap frame_;
};

// Emits `value` exactly once on tick == target_tick, silent otherwise. The
// override input ports on the operator under test are non-blocking, so the
// silent ticks keep the previous (sticky) value.
template <typename V>
class PerfOverrideSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfOverrideSourceOp)
  PerfOverrideSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<V>("output");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    if (tick_ == target_tick_) {
      op_output.emit(value_, "output");
    }
    ++tick_;
  }

  void configure(int64_t target_tick, V value) {
    target_tick_ = target_tick;
    value_ = std::move(value);
  }

 private:
  int64_t tick_{0};
  int64_t target_tick_{-1};
  V value_{};
};

}  // namespace holoscan::radar::perf
