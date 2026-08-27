// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// [EA HSDK5 ONLY] FeatureAssemblyOp cannot wire optional range_bins /
// velocity_bins on EA (RUNTIME_OPERATOR_REPLAY_MISMATCH — HSDK5_PORT §1 /
// feature_assembly OPEN_ITEMS). This Graph op takes those tensors as
// required OnAligned Exact inputs and emits (N, 2) physical [range, velocity]
// points for SISO clustering. Remove when FeatureAssembly optional wiring
// works on GA.

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
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

namespace physical_points_detail {

inline holoscan::expected<void, holoscan::Error> copy_d2h_async(
    const char* name, void* dst, const holoscan::Tensor& t, size_t bytes,
    cudaStream_t stream) {
  if (bytes == 0) {
    return {};
  }
  if (static_cast<size_t>(t.nbytes()) != bytes) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        std::string(name) + " byte size mismatch"});
  }
  if (cudaMemcpyAsync(dst, t.data(), bytes, cudaMemcpyDeviceToHost, stream) !=
      cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, std::string(name) + " D2H failed"});
  }
  return {};
}

}  // namespace physical_points_detail

class PhysicalRangeVelocityPointsOp : public holoscan::Operator<> {
 public:
  explicit PhysicalRangeVelocityPointsOp(const MaxTensorExtents& extents) {
    const int64_t max_points =
        require_extent(extents.max_points, "max_points",
                       "PhysicalRangeVelocityPointsOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_points, 2}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(detection_idx).queue_depth(2U).expects_tensor(kDevice);
    spec.input(range_bins).queue_depth(2U).expects_tensor(kDevice);
    spec.input(velocity_bins).queue_depth(2U).expects_tensor(kDevice);
    spec.output(points)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 2U, .max_byte_span = max_byte_span_});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{
            .ports = holoscan::ports(detection_idx, range_bins, velocity_bins),
            .window = std::chrono::nanoseconds{0}})
        .input(detection_idx, holoscan::Exact{})
        .input(range_bins, holoscan::Exact{})
        .input(velocity_bins, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "PhysicalRangeVelocityPointsOp";
    HOLOSCAN_TRY(auto idx_t, receive_tensor(detection_idx, who));
    HOLOSCAN_TRY(auto range_t, receive_tensor(range_bins, who));
    HOLOSCAN_TRY(auto vel_t, receive_tensor(velocity_bins, who));
    retained_idx_ = idx_t;
    retained_range_ = range_t;
    retained_vel_ = vel_t;

    const auto idx_shape = retained_idx_.shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": detection_idx must be (N, 2)"});
    }
    const int64_t n = idx_shape[0];
    const int64_t num_range = retained_range_.shape()[0];
    const int64_t num_doppler = retained_vel_.shape()[0];

    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported,
          std::string(who) + ": no CUDA compute stream"});
    }

    idx_host_.resize(static_cast<size_t>(n * 2));
    ranges_host_.resize(static_cast<size_t>(num_range));
    vels_host_.resize(static_cast<size_t>(num_doppler));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "detection_idx", idx_host_.data(), retained_idx_,
        sizeof(int64_t) * idx_host_.size(), stream));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "range_bins", ranges_host_.data(), retained_range_,
        sizeof(float) * ranges_host_.size(), stream));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "velocity_bins", vels_host_.data(), retained_vel_,
        sizeof(float) * vels_host_.size(), stream));
    // Host gather needs the D2H results; sync only this compute stream.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string(who) + ": stream sync failed"});
    }

    host_points_.resize(static_cast<size_t>(n * 2));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t dd = idx_host_[static_cast<size_t>(i * 2 + 0)];
      const int64_t rr = idx_host_[static_cast<size_t>(i * 2 + 1)];
      if (dd < 0 || dd >= num_doppler || rr < 0 || rr >= num_range) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string(who) + ": detection_idx out of bounds"});
      }
      host_points_[static_cast<size_t>(i * 2 + 0)] =
          ranges_host_[static_cast<size_t>(rr)];
      host_points_[static_cast<size_t>(i * 2 + 1)] =
          vels_host_[static_cast<size_t>(dd)];
    }

    const std::vector<int64_t> out_shape{n, 2};
    return emit_device_tensor(
        points, context, out_shape, real_dtype<float>(),
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != host_points_.size() * sizeof(float)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": points loan size mismatch"});
          }
          if (n == 0) {
            return {};
          }
          if (cudaMemcpyAsync(dst, host_points_.data(), bytes,
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": points H2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Output<holoscan::Tensor> points{"points"};

 private:
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_idx_;
  holoscan::Tensor retained_range_;
  holoscan::Tensor retained_vel_;
  std::vector<int64_t> idx_host_;
  std::vector<float> ranges_host_;
  std::vector<float> vels_host_;
  std::vector<float> host_points_;  // H2D source; must outlive emit
};

// [EA HSDK5 ONLY] Same FeatureAssembly workaround for the sparse SIMO /
// MIMO / AoA path: required OnAligned Exact ports emit (N, 4) physical
// [range, velocity, az, el] points. Remove when FeatureAssembly optional
// wiring works on GA.
class PhysicalRangeVelocityAzElPointsOp : public holoscan::Operator<> {
 public:
  explicit PhysicalRangeVelocityAzElPointsOp(const MaxTensorExtents& extents) {
    const int64_t max_points =
        require_extent(extents.max_points, "max_points",
                       "PhysicalRangeVelocityAzElPointsOp");
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_points, 4}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(detection_idx).queue_depth(2U).expects_tensor(kDevice);
    spec.input(range_bins).queue_depth(2U).expects_tensor(kDevice);
    spec.input(velocity_bins).queue_depth(2U).expects_tensor(kDevice);
    spec.input(aoa_estimates).queue_depth(2U).expects_tensor(kDevice);
    spec.output(points)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 2U, .max_byte_span = max_byte_span_});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{
            .ports = holoscan::ports(detection_idx, range_bins, velocity_bins,
                                     aoa_estimates),
            .window = std::chrono::nanoseconds{0}})
        .input(detection_idx, holoscan::Exact{})
        .input(range_bins, holoscan::Exact{})
        .input(velocity_bins, holoscan::Exact{})
        .input(aoa_estimates, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "PhysicalRangeVelocityAzElPointsOp";
    HOLOSCAN_TRY(auto idx_t, receive_tensor(detection_idx, who));
    HOLOSCAN_TRY(auto range_t, receive_tensor(range_bins, who));
    HOLOSCAN_TRY(auto vel_t, receive_tensor(velocity_bins, who));
    HOLOSCAN_TRY(auto aoa_t, receive_tensor(aoa_estimates, who));
    retained_idx_ = idx_t;
    retained_range_ = range_t;
    retained_vel_ = vel_t;
    retained_aoa_ = aoa_t;

    const auto idx_shape = retained_idx_.shape();
    if (idx_shape.size() != 2 || idx_shape[1] != 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": detection_idx must be (N, 2)"});
    }
    const int64_t n = idx_shape[0];
    const auto aoa_shape = retained_aoa_.shape();
    if (aoa_shape.size() != 2 || aoa_shape[0] != n || aoa_shape[1] != 2) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          std::string(who) + ": aoa_estimates must be (N, 2)"});
    }
    const int64_t num_range = retained_range_.shape()[0];
    const int64_t num_doppler = retained_vel_.shape()[0];

    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported,
          std::string(who) + ": no CUDA compute stream"});
    }

    idx_host_.resize(static_cast<size_t>(n * 2));
    ranges_host_.resize(static_cast<size_t>(num_range));
    vels_host_.resize(static_cast<size_t>(num_doppler));
    aoa_host_.resize(static_cast<size_t>(n * 2));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "detection_idx", idx_host_.data(), retained_idx_,
        sizeof(int64_t) * idx_host_.size(), stream));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "range_bins", ranges_host_.data(), retained_range_,
        sizeof(float) * ranges_host_.size(), stream));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "velocity_bins", vels_host_.data(), retained_vel_,
        sizeof(float) * vels_host_.size(), stream));
    HOLOSCAN_RETURN_IF_ERROR(physical_points_detail::copy_d2h_async(
        "aoa_estimates", aoa_host_.data(), retained_aoa_,
        sizeof(float) * aoa_host_.size(), stream));
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string(who) + ": stream sync failed"});
    }

    host_points_.resize(static_cast<size_t>(n * 4));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t dd = idx_host_[static_cast<size_t>(i * 2 + 0)];
      const int64_t rr = idx_host_[static_cast<size_t>(i * 2 + 1)];
      if (dd < 0 || dd >= num_doppler || rr < 0 || rr >= num_range) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kInvalidArgument,
            std::string(who) + ": detection_idx out of bounds"});
      }
      host_points_[static_cast<size_t>(i * 4 + 0)] =
          ranges_host_[static_cast<size_t>(rr)];
      host_points_[static_cast<size_t>(i * 4 + 1)] =
          vels_host_[static_cast<size_t>(dd)];
      host_points_[static_cast<size_t>(i * 4 + 2)] =
          aoa_host_[static_cast<size_t>(i * 2 + 0)];
      host_points_[static_cast<size_t>(i * 4 + 3)] =
          aoa_host_[static_cast<size_t>(i * 2 + 1)];
    }

    const std::vector<int64_t> out_shape{n, 4};
    return emit_device_tensor(
        points, context, out_shape, real_dtype<float>(),
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != host_points_.size() * sizeof(float)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": points loan size mismatch"});
          }
          if (n == 0) {
            return {};
          }
          if (cudaMemcpyAsync(dst, host_points_.data(), bytes,
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                std::string(who) + ": points H2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Input<holoscan::Tensor> aoa_estimates{"aoa_estimates"};
  holoscan::Output<holoscan::Tensor> points{"points"};

 private:
  std::size_t max_byte_span_{0U};
  holoscan::Tensor retained_idx_;
  holoscan::Tensor retained_range_;
  holoscan::Tensor retained_vel_;
  holoscan::Tensor retained_aoa_;
  std::vector<int64_t> idx_host_;
  std::vector<float> ranges_host_;
  std::vector<float> vels_host_;
  std::vector<float> aoa_host_;
  std::vector<float> host_points_;
};

}  // namespace holoscan::radar::app
