// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"

namespace holoscan::radar {

namespace cphd {
class CphdFile;  // host-side file reading; defined in cphd_file.hpp
}  // namespace cphd

// Reads a CPHD collection and emits phase history with the per-pulse
// geometry stored alongside it. A source operator: no data inputs. This
// header is the contract reference for the operator's ports, shapes,
// dtypes, and parameters; see docs/planning/SAR/OPERATOR_API.md.
//
// Out: phase_history      — complex64/complex128, (num_pulses, num_samples).
// Out: positions          — float32/float64, (num_pulses, 3). Per-pulse
//                           aperture phase-centre positions.
// Out: reference_position — float32/float64, (num_pulses, 3). Per-pulse
//                           Stabilization Reference Point position.
// Out: sample_axis        — optional, float32/float64, (num_samples,).
// Out: pulse_times        — optional, float64 always, (num_pulses,).
// Out: velocities         — optional, float32/float64, (num_pulses, 3).
// Out: coordinate_frame   — optional, CoordinateFrame; emitted once, on the
//                           first tick of the collection.
//
// num_pulses is pulses_per_tick except on the final tick, where it is the
// remainder — downstream must read the pulse count off the tensor.
//
// Precision is fixed at compile time by the template instantiation
// (T = float → complex64/float32, T = double → complex128/float64).
// pulse_times is float64 in both, since single precision cannot hold a
// collection-relative timestamp to useful accuracy.
//
// Scope note (SIGPROC_PLAN.md's minimum-per-demo rule): this slice reads
// what the mode 1 collection needs — a single file, one channel, `CF8`
// samples, no `AmpSF`, FX domain. The contract's remaining paths (multi-file
// `path`, `CI4`/`CI2`, `AmpSF`, TOA domain) throw a clear error rather than
// guessing, and land with the mode 3 milestone whose collections force them.
template <typename T>
class CPHDReaderOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CPHDReaderOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CPHDReaderOp)

  CPHDReaderOp() = default;
  ~CPHDReaderOp() override;

  // Registers the argument setters for the custom parameter types and
  // attaches the boolean scheduling term the operator stops itself with.
  void initialize() override;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Parameters per the contract. All static: a reader's file, frame and
  // batching are build-time configuration, not in-flight overrides.
  Parameter<std::string> path_;
  Parameter<CoordinateFrame> coordinate_frame_;
  Parameter<int64_t> pulses_per_tick_;
  Parameter<int64_t> channel_;
  Parameter<std::vector<int64_t>> pulse_range_;

  // Lets the operator stop itself once the selected range is exhausted. Held
  // directly rather than as a parameter: a condition passed to add_arg is
  // registered as a scheduling term, not as a parameter value, so exposing
  // it as one would add a contract-surface parameter that never binds.
  std::shared_ptr<holoscan::BooleanCondition> stop_condition_;

  std::unique_ptr<cphd::CphdFile> file_;

  // Resolved at start() from pulse_range_ and the file's vector count.
  int64_t first_pulse_{0};
  int64_t last_pulse_{0};  // exclusive
  int64_t next_pulse_{0};
  bool frame_emitted_{false};
  // Real collections carry occasional non-finite ephemeris; warned about
  // once per instance rather than per tick.
  bool nonfinite_warned_{false};
};

// Explicit instantiations live in cphd_reader_op.cu.
extern template class CPHDReaderOp<float>;
extern template class CPHDReaderOp<double>;

}  // namespace holoscan::radar
