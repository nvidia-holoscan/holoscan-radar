// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-specific operators for fmcw_aoa_estimators, colocated with the app:
//   SnapshotK1SliceOp  - center-slice a (N, C, K) snapshot to (N, C, 1) for the
//                        single-snapshot Beamscan branch.
//   EstimatorRouterOp  - pass-through of snapshot/is_valid that owns the
//                        estimator-selection policy. Its selection is read each
//                        tick by the app's set_dynamic_flows() callback, which
//                        activates exactly one downstream estimator path, so
//                        only the selected estimator ticks.
//
// The router separates the selection *policy* (rotate / hold / file) from the
// *transport*: with no external signal it rotates on a fixed dwell (headless,
// deterministic); a control file/FIFO latches a live override (real-time,
// still unattended-capable). Kept app-local (plain-tensor-ports invariant),
// not a shared operator or contract change.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>  // NOLINT(build/include_order)
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"

namespace radar = holoscan::radar;

// Center-slice the trailing snapshot axis: (num_items, num_channels,
// num_snapshots) -> (num_items, num_channels, 1), taking the center look
// (k = num_snapshots / 2). Beamscan is single-snapshot only, so the multi-look
// gather feeding the covariance/gridless estimators is narrowed to one look
// here. is_valid (num_items, num_snapshots) is sliced the same way. A snapshot
// that is already single-look (rank 2, or trailing dim 1) is forwarded as-is.
class SnapshotK1SliceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnapshotK1SliceOp)
  SnapshotK1SliceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("snapshot");
    spec.input<holoscan::TensorMap>("is_valid")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("snapshot");
    spec.output<holoscan::TensorMap>("is_valid")
        .condition(holoscan::ConditionType::kNone);
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "SnapshotK1SliceOp";
    auto snapshot = radar::receive_single(op_input, "snapshot", who);
    auto is_valid = radar::receive_optional(op_input, "is_valid", who);

    const auto& shape = snapshot->shape();
    if (shape.size() != 2 && shape.size() != 3) {
      throw std::runtime_error(
          "SnapshotK1SliceOp: snapshot must be (num_items, num_channels) or "
          "(num_items, num_channels, num_snapshots)");
    }
    const int64_t num_items = shape[0];
    const int64_t num_channels = shape[1];
    const int64_t num_snapshots = (shape.size() == 3) ? shape[2] : 1;

    if (num_snapshots <= 1) {
      // Already single-look: forward unchanged (still one entry per port).
      radar::emit_single(op_output, "snapshot", snapshot);
      if (is_valid) {
        radar::emit_single(op_output, "is_valid", is_valid);
      }
      return;
    }

    const int64_t center = num_snapshots / 2;
    auto sliced = slice_center(*snapshot, num_items * num_channels,
                               num_snapshots, center, who, "snapshot");
    radar::emit_single(op_output, "snapshot", sliced);

    if (is_valid) {
      const auto& vshape = is_valid->shape();
      if (vshape.size() == 2 && vshape[1] == num_snapshots) {
        auto vsliced = slice_center(*is_valid, vshape[0], num_snapshots, center,
                                    who, "is_valid");
        radar::emit_single(op_output, "is_valid", vsliced);
      } else {
        radar::emit_single(op_output, "is_valid", is_valid);
      }
    }
  }

 private:
  // Gather element `center` out of each length-`inner` contiguous row of a
  // rank-collapsed (rows, inner) tensor into a contiguous (rows, 1) tensor, on
  // device, preserving dtype. rows = product of leading dims.
  static std::shared_ptr<holoscan::Tensor> slice_center(
      const holoscan::Tensor& src, int64_t rows, int64_t inner, int64_t center,
      const char* who, const char* what) {
    const DLDataType dtype = src.dtype();
    const size_t elem_bytes =
        static_cast<size_t>(dtype.bits / 8) * static_cast<size_t>(dtype.lanes);
    std::vector<int64_t> out_shape;
    for (auto d : src.shape()) {
      out_shape.push_back(d);
    }
    out_shape.back() = 1;
    auto out = radar::make_device_tensor(out_shape, dtype);
    if (rows > 0) {
      const auto* base = static_cast<const char*>(src.data());
      const cudaError_t err = cudaMemcpy2D(
          out->data(), elem_bytes, base + static_cast<size_t>(center) *
                                              elem_bytes,
          static_cast<size_t>(inner) * elem_bytes, elem_bytes,
          static_cast<size_t>(rows), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string(who) + ": " + what +
                                 " center-slice D2D failed");
      }
    }
    return out;
  }
};

// Estimator selection policy for the dynamic-flow router. Structural (chosen at
// build time); the *which* it yields per tick is what varies.
enum class RouterMode { kRotate, kHold, kFile };

// Pass-through router that decides which single downstream estimator runs each
// tick. compute() forwards snapshot/is_valid unchanged and advances the policy;
// the app's set_dynamic_flows() callback reads active_target() and activates
// only that estimator's edges. The selection sources, in resolution order:
//   kHold   - a fixed estimator (hold_index), ignores external input.
//   kRotate - fixed dwell: active = (tick / rotate_period) % N.
//   kFile   - poll control_path each tick; a recognized name latches (sticky)
//             until changed; an absent/empty/unrecognized file falls back to
//             the rotation. Headless- and SSH-friendly ("echo music > path").
class EstimatorRouterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(EstimatorRouterOp)
  EstimatorRouterOp() = default;

  // Configured by the app in compose() (plain members, not Holoscan params, so
  // the same object is visible to the dynamic-flow callback).
  RouterMode mode{RouterMode::kRotate};
  int64_t rotate_period{40};
  int64_t hold_index{0};
  int64_t log_every{0};  // 0 = log only when the active estimator changes
  std::string control_path{};
  std::vector<std::string> display_names{};  // human names, index-aligned
  std::vector<std::string> target_op_names{};  // downstream op names, aligned

  const std::string& active_target() const {
    return target_op_names[static_cast<size_t>(active_index_)];
  }

  // The estimator selected on the most recent tick (index into display_names /
  // target_op_names). Read by the viz label each frame; a one-frame lag versus
  // the estimator that actually ran is immaterial for a status label.
  int64_t active_index() const { return active_index_; }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("snapshot");
    spec.input<holoscan::TensorMap>("is_valid")
        .condition(holoscan::ConditionType::kNone);
    spec.output<holoscan::TensorMap>("snapshot");
    spec.output<holoscan::TensorMap>("is_valid")
        .condition(holoscan::ConditionType::kNone);
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    const char* who = "EstimatorRouterOp";
    if (target_op_names.empty()) {
      throw std::runtime_error(
          "EstimatorRouterOp: no estimator targets configured");
    }
    auto snapshot = radar::receive_single(op_input, "snapshot", who);
    auto is_valid = radar::receive_optional(op_input, "is_valid", who);

    active_index_ = resolve_selection();

    radar::emit_single(op_output, "snapshot", snapshot);
    if (is_valid) {
      radar::emit_single(op_output, "is_valid", is_valid);
    }

    const bool changed = (active_index_ != last_logged_index_);
    const bool periodic =
        log_every > 0 && (tick_ % log_every == 0);
    if (changed || periodic) {
      HOLOSCAN_LOG_INFO("EstimatorRouterOp: frame {} -> {} estimator", tick_,
                        display_names[static_cast<size_t>(active_index_)]);
      last_logged_index_ = active_index_;
    }
    ++tick_;
  }

 private:
  int64_t resolve_selection() {
    const int64_t n = static_cast<int64_t>(target_op_names.size());
    const int64_t period = rotate_period > 0 ? rotate_period : 1;
    const int64_t rotated = (tick_ / period) % n;
    if (mode == RouterMode::kHold) {
      return clamp_index(hold_index, n);
    }
    if (mode == RouterMode::kRotate) {
      return rotated;
    }
    // kFile: poll and latch. A recognized value sticks until the next
    // recognized value; an absent/empty/unrecognized poll leaves the prior
    // latch in place (so a one-shot `echo music > path` persists). Rotation is
    // the fallback only until the first override latches.
    const int64_t requested = poll_control_file();
    if (requested >= 0) {
      latched_index_ = requested;
    }
    return latched_index_ >= 0 ? latched_index_ : rotated;
  }

  static int64_t clamp_index(int64_t idx, int64_t n) {
    if (idx < 0) {
      return 0;
    }
    if (idx >= n) {
      return n - 1;
    }
    return idx;
  }

  // Read the first whitespace-delimited token from control_path and map it to
  // an estimator index. Returns -1 when the file is absent, empty, or does not
  // name a known estimator (1-based digits are also accepted).
  int64_t poll_control_file() {
    if (control_path.empty()) {
      return -1;
    }
    std::ifstream in(control_path);
    if (!in) {
      return -1;
    }
    std::string token;
    in >> token;
    if (token.empty()) {
      return -1;
    }
    std::string norm;
    norm.reserve(token.size());
    for (char c : token) {
      if (c == '-' || c == '_') {
        continue;
      }
      norm.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(
          c))));
    }
    for (size_t i = 0; i < display_names.size(); ++i) {
      std::string cand;
      for (char c : display_names[i]) {
        if (c == '-' || c == '_') {
          continue;
        }
        cand.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
      }
      if (norm == cand) {
        return static_cast<int64_t>(i);
      }
    }
    if (norm.size() == 1 && norm[0] >= '1' &&
        norm[0] <= '9') {  // 1-based digit
      const int64_t idx = norm[0] - '1';
      if (idx < static_cast<int64_t>(display_names.size())) {
        return idx;
      }
    }
    return -1;
  }

  int64_t tick_{0};
  int64_t active_index_{0};
  int64_t latched_index_{-1};
  int64_t last_logged_index_{-1};
};
