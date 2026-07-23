// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/tx_schedule.hpp"

namespace holoscan::radar {

// TDM front-end marshaling: split interleaved slow-time into an explicit TX
// axis. Contract reference: ports, shapes, dtypes, parameters.
//
// In:  adc_cube — complex64/complex128 or float32/float64 (real IF),
//               (..., N_rx, M·L, samples_per_chirp).
// Out: adc_cube — same dtype, (..., M_tx, N_rx, L, samples_per_chirp).
//
// Scope (v1): uniform round-robin TDM only — each TX fires once per length-M
// cycle and the cycle repeats identically across all L repetitions (equal
// per-TX counts, fixed PRI/M inter-TX spacing). Non-uniform / staggered
// schedules (varying TX dwell or count) are rejected by TXSchedule with an
// error. Supporting them is a deferred future enhancement (needs a
// per-TX-timing schedule descriptor plus a matching MIMOCoherentCompensationOp
// model) — see STATUS.md → Roadmap (settled as L12).
template <typename T>
class MIMODeinterleaveOp : public holoscan::Operator {
 public:
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "MIMODeinterleaveOp supports float and double only");

  HOLOSCAN_OPERATOR_FORWARD_ARGS(MIMODeinterleaveOp)

  MIMODeinterleaveOp() = default;
  ~MIMODeinterleaveOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  void ensure_device_gather_map(int64_t slow_time, const TXSchedule& schedule,
                                cudaStream_t stream);

  Parameter<TXSchedule> tx_schedule_;

  std::shared_ptr<holoscan::Tensor> retained_input_;

  int64_t cached_slow_time_{-1};
  TXSchedule cached_schedule_{};
  std::vector<int64_t> cached_gather_host_;
  int64_t* d_gather_map_{nullptr};
  size_t d_gather_capacity_{0};
};

}  // namespace holoscan::radar
