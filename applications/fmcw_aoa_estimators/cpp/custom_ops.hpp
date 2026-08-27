// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-local Graph operators for fmcw_aoa_estimators:
//   SnapshotK1SliceOp  — center-slice (N, C, K) → (N, C, 1) for Beamscan
//   EstimatorRouterOp  — policy + N snapshot outs; emit only the selected one
//   AoaMergeOp         — OnAnyInput fan-in of five aoa_estimates → one out
//
// Replaces Application set_dynamic_flows(): idle estimators never receive a
// snapshot (OnEach), and the merge reacts to whichever aoa port fires.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>  // NOLINT(build/include_order)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;

// Async D2D from `src` into a new loan. Caller must keep `src` alive until the
// next tick (retained_input_ pattern) — the fill only enqueues on the stream.
inline holoscan::expected<holoscan::TensorOutputLoan, holoscan::Error>
prepare_tensor_copy(holoscan::Output<holoscan::Tensor>& port,
                    holoscan::ExecutionContext& context,
                    const holoscan::Tensor& src, const char* who) {
  return radar::prepare_device_tensor(
      port, context, src.shape(), src.dtype(),
      [&](void* dst, std::size_t bytes)
          -> holoscan::expected<void, holoscan::Error> {
        if (bytes != static_cast<std::size_t>(src.nbytes())) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kFailure,
              std::string(who) + ": loan size mismatch"});
        }
        if (bytes == 0) {
          return {};
        }
        if (cudaMemcpyAsync(dst, src.data(), bytes, cudaMemcpyDeviceToDevice,
                            context.cuda_stream().get()) != cudaSuccess) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kFailure,
              std::string(who) + ": D2D copy failed"});
        }
        return {};
      });
}

inline holoscan::expected<void, holoscan::Error> emit_tensor_copy(
    holoscan::Output<holoscan::Tensor>& port, holoscan::ExecutionContext& context,
    const holoscan::Tensor& src, const char* who) {
  HOLOSCAN_TRY(auto loan, prepare_tensor_copy(port, context, src, who));
  return radar::emit_prepared_tensor(port, context, std::move(loan));
}

// Center-slice the trailing snapshot axis for single-look Beamscan.
class SnapshotK1SliceOp : public holoscan::Operator<> {
 public:
  explicit SnapshotK1SliceOp(const radar::MaxTensorExtents& extents) {
    const int64_t max_items =
        radar::require_extent(extents.max_items, "max_items",
                              "SnapshotK1SliceOp");
    const int64_t max_ch =
        radar::require_extent(extents.max_channels, "max_channels",
                              "SnapshotK1SliceOp");
    max_byte_span_ = radar::tensor_byte_span(
        std::vector<int64_t>{max_items, max_ch, 1},
        radar::complex_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(snapshot_in).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.output(snapshot_out)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{snapshot_in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "SnapshotK1SliceOp";
    HOLOSCAN_TRY(auto snapshot, radar::receive_tensor(snapshot_in, who));
    // Keep the source alive past compute(); emit only enqueues the D2D.
    retained_input_ = snapshot;
    const auto& shape = snapshot.shape();
    if (shape.size() != 2 && shape.size() != 3) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "SnapshotK1SliceOp: snapshot must be (N, C) or (N, C, K)"});
    }
    const int64_t num_items = shape[0];
    const int64_t num_channels = shape[1];
    const int64_t num_snapshots = (shape.size() == 3) ? shape[2] : 1;
    if (num_snapshots <= 1) {
      return emit_tensor_copy(snapshot_out, context, retained_input_, who);
    }

    const int64_t center = num_snapshots / 2;
    const DLDataType dtype = retained_input_.dtype();
    const std::size_t elem_bytes =
        static_cast<std::size_t>(dtype.bits / 8) *
        static_cast<std::size_t>(dtype.lanes);
    const std::vector<int64_t> out_shape{num_items, num_channels, 1};
    const int64_t rows = num_items * num_channels;
    return radar::emit_device_tensor(
        snapshot_out, context, out_shape, dtype,
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != static_cast<std::size_t>(rows) * elem_bytes) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                "SnapshotK1SliceOp: sliced loan size mismatch"});
          }
          if (rows <= 0) {
            return {};
          }
          const auto* base =
              static_cast<const char*>(retained_input_.data());
          if (cudaMemcpy2DAsync(
                  dst, elem_bytes,
                  base + static_cast<std::size_t>(center) * elem_bytes,
                  static_cast<std::size_t>(num_snapshots) * elem_bytes,
                  elem_bytes, static_cast<std::size_t>(rows),
                  cudaMemcpyDeviceToDevice,
                  context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                "SnapshotK1SliceOp: center-slice D2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> snapshot_in{"snapshot_in"};
  holoscan::Output<holoscan::Tensor> snapshot_out{"snapshot_out"};

 private:
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_input_;
};

enum class RouterMode { kRotate, kHold, kFile };

struct EstimatorRouterConfig {
  RouterMode mode{RouterMode::kRotate};
  int64_t rotate_period{40};
  int64_t hold_index{0};
  int64_t log_every{0};
  std::string control_path{};
  // Index-aligned human names: beamscan, mvdr, music, root_music, esprit.
  std::vector<std::string> display_names{
      "beamscan", "mvdr", "music", "root_music", "esprit"};
};

// Pass-through router: five snapshot outputs, emit only the selected branch.
class EstimatorRouterOp : public holoscan::Operator<> {
 public:
  static constexpr int kNumEstimators = 5;

  EstimatorRouterOp(const radar::MaxTensorExtents& extents,
                    EstimatorRouterConfig config)
      : cfg_(std::move(config)) {
    const int64_t max_items =
        radar::require_extent(extents.max_items, "max_items",
                              "EstimatorRouterOp");
    const int64_t max_ch =
        radar::require_extent(extents.max_channels, "max_channels",
                              "EstimatorRouterOp");
    const int64_t max_snaps =
        radar::require_extent(extents.max_snapshots, "max_snapshots",
                              "EstimatorRouterOp");
    max_byte_span_ = radar::tensor_byte_span(
        std::vector<int64_t>{max_items, max_ch, max_snaps},
        radar::complex_dtype<float>());
    if (cfg_.display_names.size() != static_cast<size_t>(kNumEstimators)) {
      throw std::runtime_error(
          "EstimatorRouterOp: display_names must have 5 entries");
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(snapshot_in).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    auto declare_out = [&](holoscan::Output<holoscan::Tensor>& port) {
      spec.output(port)
          .produces_tensor(holoscan::radar::pooled_tensor_output(
              kDevice, 3U, max_byte_span_));
    };
    declare_out(snapshot_beamscan);
    declare_out(snapshot_mvdr);
    declare_out(snapshot_music);
    declare_out(snapshot_root_music);
    declare_out(snapshot_esprit);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{snapshot_in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "EstimatorRouterOp";
    HOLOSCAN_TRY(auto snapshot, radar::receive_tensor(snapshot_in, who));
    retained_input_ = snapshot;
    active_index_ = resolve_selection();

    holoscan::Output<holoscan::Tensor>* outs[kNumEstimators] = {
        &snapshot_beamscan, &snapshot_mvdr, &snapshot_music,
        &snapshot_root_music, &snapshot_esprit};
    HOLOSCAN_RETURN_IF_ERROR(
        emit_tensor_copy(*outs[active_index_], context, retained_input_, who));

    const bool changed = (active_index_ != last_logged_index_);
    const bool periodic =
        cfg_.log_every > 0 && (tick_ % cfg_.log_every == 0);
    if (changed || periodic) {
      HOLOSCAN_LOG_INFO("EstimatorRouterOp: frame {} -> {} estimator", tick_,
                        cfg_.display_names[static_cast<size_t>(active_index_)]);
      last_logged_index_ = active_index_;
    }
    ++tick_;
    return {};
  }

  int64_t active_index() const { return active_index_; }

  holoscan::Input<holoscan::Tensor> snapshot_in{"snapshot"};
  holoscan::Output<holoscan::Tensor> snapshot_beamscan{"snapshot_beamscan"};
  holoscan::Output<holoscan::Tensor> snapshot_mvdr{"snapshot_mvdr"};
  holoscan::Output<holoscan::Tensor> snapshot_music{"snapshot_music"};
  holoscan::Output<holoscan::Tensor> snapshot_root_music{
      "snapshot_root_music"};
  holoscan::Output<holoscan::Tensor> snapshot_esprit{"snapshot_esprit"};

 private:
  int64_t resolve_selection() {
    const int64_t n = kNumEstimators;
    const int64_t period = cfg_.rotate_period > 0 ? cfg_.rotate_period : 1;
    const int64_t rotated = (tick_ / period) % n;
    if (cfg_.mode == RouterMode::kHold) {
      return clamp_index(cfg_.hold_index, n);
    }
    if (cfg_.mode == RouterMode::kRotate) {
      return rotated;
    }
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

  int64_t poll_control_file() {
    if (cfg_.control_path.empty()) {
      return -1;
    }
    std::ifstream in(cfg_.control_path);
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
      norm.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
    }
    for (size_t i = 0; i < cfg_.display_names.size(); ++i) {
      std::string cand;
      for (char c : cfg_.display_names[i]) {
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
        norm[0] <= static_cast<char>('0' + kNumEstimators)) {
      return static_cast<int64_t>(norm[0] - '1');
    }
    return -1;
  }

  EstimatorRouterConfig cfg_{};
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_input_;
  int64_t tick_{0};
  int64_t active_index_{0};
  int64_t last_logged_index_{-1};
  int64_t latched_index_{-1};
};

// Fan-in of five aoa_estimates ports. Default OnAnyInput: one activation per
// accepted sample on any port; receive only the selected port and forward.
class AoaMergeOp : public holoscan::Operator<> {
 public:
  explicit AoaMergeOp(const radar::MaxTensorExtents& extents) {
    const int64_t max_items =
        radar::require_extent(extents.max_items, "max_items", "AoaMergeOp");
    max_byte_span_ = radar::tensor_byte_span(
        std::vector<int64_t>{max_items, 2}, radar::real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    auto declare_in = [&](holoscan::Input<holoscan::Tensor>& port) {
      spec.input(port).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    };
    declare_in(aoa_beamscan);
    declare_in(aoa_mvdr);
    declare_in(aoa_music);
    declare_in(aoa_root_music);
    declare_in(aoa_esprit);
    spec.output(aoa_estimates)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, max_byte_span_));
    spec.output(estimator_idx)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 1U, sizeof(int64_t)));
  }

  // Empty contract → synthesized OnAnyInput (HSDK 5 default for inputful ops).
  [[nodiscard]] holoscan::Contract contract() const override {
    return holoscan::Contract{};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "AoaMergeOp";
    holoscan::Input<holoscan::Tensor>* ports[5] = {
        &aoa_beamscan, &aoa_mvdr, &aoa_music, &aoa_root_music, &aoa_esprit};
    holoscan::Input<holoscan::Tensor>* selected = nullptr;
    for (int64_t i = 0; i < 5; ++i) {
      auto* p = ports[i];
      if (!p->is_missing()) {
        if (selected != nullptr) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kFailure,
              "AoaMergeOp: more than one aoa port selected"});
        }
        selected = p;
        host_index_ = i;
      }
    }
    if (selected == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "AoaMergeOp: no aoa port selected for this activation"});
    }
    HOLOSCAN_TRY(auto aoa, radar::receive_tensor(*selected, who));
    retained_input_ = aoa;
    HOLOSCAN_TRY(auto aoa_loan,
                 prepare_tensor_copy(aoa_estimates, context, retained_input_,
                                     who));
    // Which estimator produced this frame, for the viewer's header label.
    // host_index_ is a member so the pageable H2D source outlives the copy.
    const std::vector<int64_t> idx_shape{1};
    HOLOSCAN_TRY(
        auto idx_loan,
        radar::prepare_device_tensor(
            estimator_idx, context, idx_shape, DLDataType{kDLInt, 64, 1},
            [&](void* dst, std::size_t bytes)
                -> holoscan::expected<void, holoscan::Error> {
              if (bytes != sizeof(int64_t)) {
                return holoscan::make_unexpected(holoscan::Error{
                    holoscan::ErrorCode::kFailure,
                    "AoaMergeOp: estimator_idx loan size mismatch"});
              }
              if (cudaMemcpyAsync(dst, &host_index_, bytes,
                                  cudaMemcpyHostToDevice,
                                  context.cuda_stream().get()) != cudaSuccess) {
                return holoscan::make_unexpected(holoscan::Error{
                    holoscan::ErrorCode::kFailure,
                    "AoaMergeOp: estimator_idx H2D failed"});
              }
              return {};
            }));
    HOLOSCAN_TRY(radar::emit_prepared_tensor(aoa_estimates, context,
                                             std::move(aoa_loan)));
    return radar::emit_prepared_tensor(estimator_idx, context,
                                       std::move(idx_loan));
  }

  holoscan::Input<holoscan::Tensor> aoa_beamscan{"aoa_beamscan"};
  holoscan::Input<holoscan::Tensor> aoa_mvdr{"aoa_mvdr"};
  holoscan::Input<holoscan::Tensor> aoa_music{"aoa_music"};
  holoscan::Input<holoscan::Tensor> aoa_root_music{"aoa_root_music"};
  holoscan::Input<holoscan::Tensor> aoa_esprit{"aoa_esprit"};
  holoscan::Output<holoscan::Tensor> aoa_estimates{"aoa_estimates"};
  holoscan::Output<holoscan::Tensor> estimator_idx{"estimator_idx"};

 private:
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_input_;
  int64_t host_index_{0};
};
