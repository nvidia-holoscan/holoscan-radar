// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-local Graph operators for fmcw_dense_simo:
//   AnglePeakPickOp — one peak (az,el) beam per detected RD cell
//   CubeAoaLookupOp — (az_bin, el_bin) → aoa_estimates
//   TrailingIdxColumnsOp — dense (N,K>=2) idx → (N,2) trailing columns
// Kept local: the dense detect+angle middle diverges from the sparse contract.

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <limits>
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
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

// Collapse per-beam cube detections to one (az, el, d, r) row per RD cell —
// the beam with the largest CFAR score. Host path (D2H) for EA quality.
class AnglePeakPickOp : public holoscan::Operator<> {
 public:
  explicit AnglePeakPickOp(const MaxTensorExtents& extents) {
    const int64_t max_dets =
        require_extent(extents.max_detections, "max_detections",
                       "AnglePeakPickOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_dets, 4}, DLDataType{kDLInt, 64, 1});
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(detection_idx)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.input(detection_score)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.output(detection_idx_out)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{
            .ports = holoscan::ports(detection_idx, detection_score),
            .window = std::chrono::nanoseconds{0}})
        .input(detection_idx, holoscan::Exact{})
        .input(detection_score, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "AnglePeakPickOp";
    HOLOSCAN_TRY(auto idx_t, receive_tensor(detection_idx, who));
    HOLOSCAN_TRY(auto score_t, receive_tensor(detection_score, who));

    const auto ishape = idx_t.shape();
    if (ishape.size() != 2 || ishape[1] != 4) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": detection_idx must be (num_points, 4)"});
    }
    const auto sshape = score_t.shape();
    if (sshape.size() != 4) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) +
              ": detection_score must be rank-4 "
              "(num_az, num_el, num_doppler, num_range)"});
    }
    const int64_t n = ishape[0];
    const int64_t na = sshape[0];
    const int64_t ne = sshape[1];
    const int64_t nd = sshape[2];
    const int64_t nr = sshape[3];

    std::vector<int64_t> host_idx(static_cast<size_t>(n * 4));
    std::vector<float> host_score(static_cast<size_t>(na * ne * nd * nr));
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string(who) + ": device synchronize failed"});
    }
    if (n > 0 &&
        cudaMemcpy(host_idx.data(), idx_t.data(),
                   static_cast<size_t>(idx_t.nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string(who) + ": detection_idx D2H failed"});
    }
    if (cudaMemcpy(host_score.data(), score_t.data(),
                   static_cast<size_t>(score_t.nbytes()),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string(who) + ": detection_score D2H failed"});
    }

    std::vector<int64_t> seen_flag(static_cast<size_t>(nd * nr), 0);
    std::vector<std::pair<int64_t, int64_t>> cells;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t d = host_idx[static_cast<size_t>(i * 4 + 2)];
      const int64_t r = host_idx[static_cast<size_t>(i * 4 + 3)];
      if (d < 0 || d >= nd || r < 0 || r >= nr) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string(who) +
                ": detection (doppler, range) out of score bounds"});
      }
      const size_t key = static_cast<size_t>(d * nr + r);
      if (seen_flag[key] == 0) {
        seen_flag[key] = 1;
        cells.emplace_back(d, r);
      }
    }

    const int64_t m = static_cast<int64_t>(cells.size());
    std::vector<int64_t> host_out(static_cast<size_t>(m * 4));
    for (int64_t c = 0; c < m; ++c) {
      const int64_t d = cells[static_cast<size_t>(c)].first;
      const int64_t r = cells[static_cast<size_t>(c)].second;
      int64_t best_az = 0;
      int64_t best_el = 0;
      float best = -std::numeric_limits<float>::infinity();
      for (int64_t a = 0; a < na; ++a) {
        for (int64_t e = 0; e < ne; ++e) {
          const float v = host_score[static_cast<size_t>(
              ((a * ne + e) * nd + d) * nr + r)];
          if (v > best) {
            best = v;
            best_az = a;
            best_el = e;
          }
        }
      }
      host_out[static_cast<size_t>(c * 4 + 0)] = best_az;
      host_out[static_cast<size_t>(c * 4 + 1)] = best_el;
      host_out[static_cast<size_t>(c * 4 + 2)] = d;
      host_out[static_cast<size_t>(c * 4 + 3)] = r;
    }

    const std::vector<int64_t> out_shape{m, 4};
    return emit_device_tensor(
        detection_idx_out, context, out_shape, DLDataType{kDLInt, 64, 1},
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != host_out.size() * sizeof(int64_t)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": detection_idx loan size mismatch"});
          }
          if (m == 0) {
            return {};
          }
          if (cudaMemcpyAsync(dst, host_out.data(), bytes,
                              cudaMemcpyHostToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": detection_idx H2D failed"});
          }
          return {};
        });
  }

    // Graph forbids duplicate port names across in/out; app wires this as the
    // peaked detection_idx.
    holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
    holoscan::Input<holoscan::Tensor> detection_score{"detection_score"};
    holoscan::Output<holoscan::Tensor> detection_idx_out{"detection_idx_out"};

 private:
  std::size_t max_byte_span_{0U};
};

// Maps cube (az_bin, el_bin) columns to physical aoa_estimates (N, 2).
class CubeAoaLookupOp : public holoscan::Operator<> {
 public:
  struct Config {
    std::vector<double> az_grid;
    std::vector<double> el_grid;
  };

  CubeAoaLookupOp(const MaxTensorExtents& extents, Config config)
      : az_grid_(std::move(config.az_grid)),
        el_grid_(std::move(config.el_grid)) {
    const int64_t max_dets =
        require_extent(extents.max_detections, "max_detections",
                       "CubeAoaLookupOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_dets, 2}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(detection_idx)
        .queue_depth(2U)
        .expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.output(aoa_estimates)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{detection_idx});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "CubeAoaLookupOp";
    HOLOSCAN_TRY(auto idx_t, receive_tensor(detection_idx, who));
    const auto shape = idx_t.shape();
    if (shape.size() != 2 || shape[1] < 4) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) +
              ": detection_idx must be (num_points, >= 4) with leading "
              "(az_bin, el_bin) columns"});
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    const int64_t num_az = static_cast<int64_t>(az_grid_.size());
    const int64_t num_el = static_cast<int64_t>(el_grid_.size());

    std::vector<int64_t> host_idx(static_cast<size_t>(n * cols));
    std::vector<float> host_aoa(static_cast<size_t>(n * 2));
    if (n > 0) {
      if (cudaDeviceSynchronize() != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string(who) + ": device synchronize failed"});
      }
      if (cudaMemcpy(host_idx.data(), idx_t.data(),
                     static_cast<size_t>(idx_t.nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string(who) + ": detection_idx D2H failed"});
      }
      for (int64_t i = 0; i < n; ++i) {
        const int64_t az_bin = host_idx[static_cast<size_t>(i * cols + 0)];
        const int64_t el_bin = host_idx[static_cast<size_t>(i * cols + 1)];
        if (az_bin < 0 || az_bin >= num_az || el_bin < 0 ||
            el_bin >= num_el) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kInvalidArgument,
              std::string(who) + ": detection az/el bin out of grid range"});
        }
        host_aoa[static_cast<size_t>(i * 2 + 0)] =
            static_cast<float>(az_grid_[static_cast<size_t>(az_bin)]);
        host_aoa[static_cast<size_t>(i * 2 + 1)] =
            static_cast<float>(el_grid_[static_cast<size_t>(el_bin)]);
      }
    }

    const std::vector<int64_t> out_shape{n, 2};
    return emit_device_tensor(
        aoa_estimates, context, out_shape, real_dtype<float>(),
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != host_aoa.size() * sizeof(float)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": aoa_estimates loan size mismatch"});
          }
          if (n == 0) {
            return {};
          }
          if (cudaMemcpyAsync(dst, host_aoa.data(), bytes,
                              cudaMemcpyHostToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": aoa_estimates H2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Output<holoscan::Tensor> aoa_estimates{"aoa_estimates"};

 private:
  std::vector<double> az_grid_;
  std::vector<double> el_grid_;
  std::size_t max_byte_span_{0U};
};

// Slice trailing two columns of int64 idx (N, K>=2) → (N, 2). Pool sized
// from max_detections (detections) or max_clusters (representatives).
class TrailingIdxColumnsOp : public holoscan::Operator<> {
 public:
  explicit TrailingIdxColumnsOp(const MaxTensorExtents& extents) {
    const int64_t max_dets = extents.max_detections.value_or(0);
    const int64_t max_clust = extents.max_clusters.value_or(0);
    const int64_t max_rows = std::max(max_dets, max_clust);
    if (max_rows <= 0) {
      throw std::invalid_argument(
          "TrailingIdxColumnsOp: MaxTensorExtents.max_detections or "
          "max_clusters must be a positive value");
    }
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rows, 2}, DLDataType{kDLInt, 64, 1});
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(idx).queue_depth(2U).expects_tensor(holoscan::radar::tensor_input(kDevice));
    spec.output(idx_out)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 2U, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{idx});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "TrailingIdxColumnsOp";
    HOLOSCAN_TRY(auto idx_t, receive_tensor(idx, who));
    const auto shape = idx_t.shape();
    if (shape.size() != 2 || shape[1] < 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": expected (num_points, >= 2) int64 idx"});
    }
    const int64_t n = shape[0];
    const int64_t cols = shape[1];
    std::vector<int64_t> host(static_cast<size_t>(n * cols));
    std::vector<int64_t> trimmed(static_cast<size_t>(n * 2));
    if (n > 0) {
      if (cudaDeviceSynchronize() != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string(who) + ": device sync failed"});
      }
      if (cudaMemcpy(host.data(), idx_t.data(),
                     static_cast<size_t>(idx_t.nbytes()),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            std::string(who) + ": idx D2H failed"});
      }
      for (int64_t i = 0; i < n; ++i) {
        trimmed[static_cast<size_t>(i * 2 + 0)] =
            host[static_cast<size_t>(i * cols + (cols - 2))];
        trimmed[static_cast<size_t>(i * 2 + 1)] =
            host[static_cast<size_t>(i * cols + (cols - 1))];
      }
    }

    const std::vector<int64_t> out_shape{n, 2};
    return emit_device_tensor(
        idx_out, context, out_shape, DLDataType{kDLInt, 64, 1},
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != trimmed.size() * sizeof(int64_t)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": idx loan size mismatch"});
          }
          if (n == 0) {
            return {};
          }
          if (cudaMemcpyAsync(dst, trimmed.data(), bytes,
                              cudaMemcpyHostToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": idx H2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> idx{"idx"};
  holoscan::Output<holoscan::Tensor> idx_out{"idx_out"};

 private:
  std::size_t max_byte_span_{0U};
};

}  // namespace holoscan::radar::app
