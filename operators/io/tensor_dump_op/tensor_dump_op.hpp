// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::radar {

// On-disk representation for a dumped tensor. Static per instance: the two
// formats produce different files and a caller wanting both instantiates two
// sinks.
enum class TensorDumpFormat {
  kNpy,  // NumPy .npy v1.0 — carries shape and dtype
  kRaw,  // raw element bytes, no header
};

constexpr std::string_view to_string(TensorDumpFormat format) {
  switch (format) {
    case TensorDumpFormat::kNpy: return "npy";
    case TensorDumpFormat::kRaw: return "raw";
  }
  return "unknown";
}

// Writes the tensor arriving on its input port to disk, for offline
// inspection of any port boundary. A sink: no outputs. This header is the
// contract reference for ports and parameters; see
// docs/planning/SAR/OPERATOR_API.md.
//
// In: tensor — any dtype, any shape.
//
// Deliberately untyped, and so **not** a class template, unlike every other
// operator in the module. Its purpose is inspecting a port whose contract is
// still under construction, which a dtype constraint would defeat; invariant
// 3's precision pairing has nothing to pair here, there being no output. The
// dtype is read off the tensor and recorded in the `npy` header rather than
// assumed.
//
// Parameters:
//   path      — output path pattern; `{index}` is replaced by the tick
//               counter (0-based). Static, required.
//   format    — npy | raw. Static, default npy.
//   every_n   — [live] write only every nth message, >= 1. Default 1.
//   max_files — stop after this many files, warning once. Default unlimited.
//
// Not a product writer: `npy` is chosen so the Python tests can read a dump
// directly. It carries no geometry, frame or collection metadata.
class TensorDumpOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorDumpOp)

  TensorDumpOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<std::string> path_;
  Parameter<TensorDumpFormat> format_;
  Parameter<int64_t> every_n_;
  Parameter<int64_t> max_files_;

  // Sticky effective value of the [live] parameter, seeded from every_n_ at
  // start() and replaced by each arriving override.
  int64_t effective_every_n_{1};

  // Ticks seen, which is what `{index}` interpolates -- not the number of
  // files written. Under every_n = 3 the files are ..._0, ..._3, ..._6, so a
  // dump names the tick it came from rather than its position in the series.
  int64_t tick_{0};
  int64_t files_written_{0};
  bool limit_warned_{false};

  // Last directory created, so a run writing thousands of dumps into one
  // directory makes one filesystem call rather than one per tick.
  std::string last_parent_;

  // Reused across ticks so a long dump is not a per-tick host allocation.
  std::vector<uint8_t> host_;
};

}  // namespace holoscan::radar
