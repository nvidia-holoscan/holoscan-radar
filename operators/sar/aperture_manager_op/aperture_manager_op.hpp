// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sar/aperture_buffer.hpp"

namespace holoscan::radar {

// Aperture membership, emit cadence, and geometry staging for SAR
// formation. No kernel: it passes tensor handles, concatenates when
// a tick has both admitted and departing pulses, and drives
// BackprojectionOp.control. Contract:
// docs/planning/SAR/OPERATOR_API.md → ApertureManagerOp.
//
// In:  range_profiles      complex, (..., num_pulses, num_range_bins)
// In:  positions           real,    (..., num_pulses, 3)
// In:  reference_position  real,    (..., num_pulses, 3)
// In:  sample_geometry     optional SampleGeometry [live], held until
//                          the aperture ends
// In:  sample_points       optional real (num_v, num_u, 3), same hold
// In:  control             optional uint8 (1,) request
// In:  evict               optional uint32 (1,) oldest-pulse count
// Out: range_profiles      admitted then departing, one batch
// Out: positions, reference_position  matching order
// Out: weight              +1 / −1 / taper
// Out: sample_geometry, sample_points  forwarded at an aperture end
// Out: control             uint8 (1,) on emit ticks only
//
// max_aperture_pulses = 0 is a pass-through (mode 1 / 5 spotlight).
// Precision is fixed at compile time (T = float | double).
template <typename T>
class ApertureManagerOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ApertureManagerOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ApertureManagerOp)

  ApertureManagerOp() = default;
  ~ApertureManagerOp() override = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  // Test hooks: occupancy of the retention ring, and conservation
  // counters. Not ports. Safe to read after run() returns.
  int64_t retained_pulses() const { return buffer_.occupancy(); }
  int64_t admitted_pulses() const { return admitted_; }
  int64_t evicted_pulses() const { return evicted_; }
  int64_t dropped_pulses() const { return dropped_; }

 private:
  Parameter<int64_t> max_aperture_pulses_;
  Parameter<int64_t> emit_interval_pulses_;
  Parameter<ApertureEmitMode> emit_mode_;
  Parameter<WindowSpec> aperture_window_;

  int64_t max_{0};
  int64_t emit_interval_{0};
  ApertureEmitMode emit_mode_value_{ApertureEmitMode::kContinue};
  WindowSpec window_{};
  bool windowed_{false};
  bool evict_connected_{false};
  bool geometry_in_connected_{false};
  bool points_in_connected_{false};

  ApertureBuffer buffer_{0};
  int64_t admitted_since_emit_{0};
  int64_t slot_{0};
  int64_t admitted_{0};
  int64_t evicted_{0};
  int64_t dropped_{0};

  std::optional<SampleGeometry> pending_geometry_;
  std::shared_ptr<holoscan::Tensor> pending_points_;
  std::vector<double> stamp_window_;

  std::shared_ptr<holoscan::Tensor> ctrl_continue_;
  std::shared_ptr<holoscan::Tensor> ctrl_reset_;
  std::shared_ptr<holoscan::Tensor> ctrl_pre_emit_;
  std::shared_ptr<holoscan::Tensor> ctrl_no_emit_;

  // Stream-ordered lifetime of tensors used in this tick's concat.
  std::vector<std::shared_ptr<holoscan::Tensor>> keep_;
  std::vector<std::shared_ptr<holoscan::Tensor>> prev_keep_;
};

}  // namespace holoscan::radar
