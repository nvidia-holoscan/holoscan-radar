// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Terminal and null sinks for the performance harness.
//
//  - PerfTerminalSinkOp drains one output port and records an end timestamp
//    after a full device sync, so the recorded latency includes all GPU work
//    the operator enqueued for that frame.
//  - PerfNullSinkOp drains the remaining dense output ports as raw GXF entities
//    (no Tensor deserialization), so the bool detection_mask port does not
//    trip the SDK's unsupported-DLPack-bool error and no host copies are made.
//  - PerfChecksumSinkOp deserializes one output port's TensorMap on the first
//    frame, copies it host-side and prints an FNV-1a hash of the raw bytes.
//    Because the harness input is deterministic and identical every tick, the
//    hash is a byte-equivalence fingerprint of the operator's output on that
//    input — the gate the perf-optimize skill uses to prove a change preserved
//    the output. Do not wire it on a bool-tensor port (DLPack-bool transport
//    gap); it is meant for numeric outputs (range_profiles, rd_maps, ...).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "utils/perf_app.hpp"

namespace holoscan::radar::perf {

class PerfTerminalSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfTerminalSinkOp)
  PerfTerminalSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<gxf::Entity>("input");
  }

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

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<gxf::Entity>("input");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)op_input.receive<gxf::Entity>("input");
  }
};

// Prints one `output_checksum <bytes> <hex>` line (to stdout) on the first
// frame it receives — an FNV-1a hash of the port's TensorMap bytes, copied
// host-side after a full device sync. Deterministic input => stable hash, so
// comparing the hash before/after a code change is a byte-equivalence gate.
class PerfChecksumSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerfChecksumSinkOp)
  PerfChecksumSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("input");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto maybe = op_input.receive<holoscan::TensorMap>("input");
    if (printed_ || !maybe) {
      return;
    }
    cudaDeviceSynchronize();  // the operator's kernels are async on its stream
    uint64_t h = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
    uint64_t total_bytes = 0;
    for (const auto& [key, tensor] : maybe.value()) {
      (void)key;
      const size_t n = tensor->nbytes();
      std::vector<unsigned char> host(n);
      if (cudaMemcpy(host.data(), tensor->data(), n,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "output_checksum: D2H copy failed\n";
        return;
      }
      for (size_t i = 0; i < n; ++i) {
        h ^= host[i];
        h *= 1099511628211ULL;  // FNV-1a 64-bit prime
      }
      total_bytes += n;
    }
    std::cout << "output_checksum " << total_bytes << " " << std::hex << h
              << std::dec << "\n"
              << std::flush;
    printed_ = true;
  }

 private:
  bool printed_{false};
};

}  // namespace holoscan::radar::perf
