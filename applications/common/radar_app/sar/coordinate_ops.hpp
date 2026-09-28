// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar::app::sar {
// App-owned NVRadar scene transform. The explicit host round trip preserves
// the source branch's float32 phase-centre quantization diagnostic.
class EcefToSceneEnuOp : public Operator<> {
 public:
  EcefToSceneEnuOp(std::array<double, 3> origin, bool quantize, int64_t max_pulses)
  : origin_(origin), quantize_(quantize), max_pulses_(max_pulses) {
    if (max_pulses_ < 1)
      throw std::invalid_argument("EcefToSceneEnuOp: invalid pulse extent");
    constexpr double kA = 6378137.0;
    constexpr double kF = 1.0 / 298.257223563;
    const double b = kA * (1.0 - kF);
    const double e2 = 1.0 - b * b / (kA * kA);
    const auto [x, y, z] = origin_;
    const double lon = std::atan2(y, x);
    const double p = std::hypot(x, y);
    const double theta = std::atan2(z * kA, p * b);
    const double slt = std::sin(theta), clt = std::cos(theta);
    const double lat =
        std::atan2(z + e2 * (kA * kA / b) * slt * slt * slt, p - e2 * kA * clt * clt * clt);
    const double sl = std::sin(lat), cl = std::cos(lat);
    const double so = std::sin(lon), co = std::cos(lon);
    rotation_ = {-so, co, 0.0, -sl * co, -sl * so, cl, cl * co, cl * so, sl};
  }
  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = DLDataType{kDLFloat, 64U, 1U}, .rank = 2U};
    constexpr TensorRepresentation any_device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(positions).queue_depth(4U).expects_tensor(tensor_input(any_device));
    spec.input(reference_position).queue_depth(4U).expects_tensor(tensor_input(any_device));
    const auto bounds =
        pooled_tensor_output(device, 2U, static_cast<uint64_t>(max_pulses_) * 3U * sizeof(double));
    spec.output(positions_out).produces_tensor(bounds);
    spec.output(reference_position_out).produces_tensor(bounds);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{.ports = ports(positions, reference_position),
                           .window = std::chrono::nanoseconds{0}})
        .input(positions, Exact{})
        .input(reference_position, Exact{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto in_pos, receive_tensor(positions, "EcefToSceneEnuOp"));
    HOLOSCAN_TRY(auto in_ref, receive_tensor(reference_position, "EcefToSceneEnuOp"));
    HOLOSCAN_TRY(auto pos, transform(positions_out, in_pos, quantize_, context));
    HOLOSCAN_TRY(auto ref, transform(reference_position_out, in_ref, false, context));
    HOLOSCAN_RETURN_IF_ERROR(emit_prepared_tensor(reference_position_out, context, std::move(ref)));
    return emit_prepared_tensor(positions_out, context, std::move(pos));
  }
  Input<Tensor> positions{"positions"};
  Input<Tensor> reference_position{"reference_position"};
  Output<Tensor> positions_out{"positions_out"};
  Output<Tensor> reference_position_out{"reference_position_out"};

 private:
  expected<TensorOutputLoan, Error> transform(Output<Tensor>& output, const Tensor& input,
                                              bool quantize, ExecutionContext& context) const {
    const auto shape = input.shape();
    if (shape.size() != 2U || shape[1] != 3 || shape[0] < 1 || shape[0] > max_pulses_ ||
        (input.dtype().code != kDLFloat || input.dtype().bits != 64U ||
         input.dtype().lanes != 1U)) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "EcefToSceneEnuOp: expected float64 (num_pulses,3)"});
    }
    std::vector<double> host(static_cast<std::size_t>(shape[0]) * 3U);
    const auto bytes = host.size() * sizeof(double);
    const auto stream = context.cuda_stream().get();
    if (cudaMemcpyAsync(host.data(), input.data(), bytes, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "EcefToSceneEnuOp: D2H failed"});
    }
    for (int64_t i = 0; i < shape[0]; ++i) {
      const double dx = host[static_cast<std::size_t>(3 * i)] - origin_[0];
      const double dy = host[static_cast<std::size_t>(3 * i + 1)] - origin_[1];
      const double dz = host[static_cast<std::size_t>(3 * i + 2)] - origin_[2];
      const std::array<double, 3> d{dx, dy, dz};
      for (int j = 0; j < 3; ++j) {
        double value =
            rotation_[3 * j] * d[0] + rotation_[3 * j + 1] * d[1] + rotation_[3 * j + 2] * d[2];
        if (quantize)
          value = static_cast<double>(static_cast<float>(value));
        host[static_cast<std::size_t>(3 * i + j)] = value;
      }
    }
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
    if (!loan)
      return make_unexpected(std::move(loan).error());
    auto writer = loan->write(context.cuda_stream());
    if (!writer)
      return make_unexpected(std::move(writer).error());
    if (cudaMemcpyAsync(writer->data(), host.data(), bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "EcefToSceneEnuOp: H2D failed"});
    }
    HOLOSCAN_RETURN_IF_ERROR(std::move(*writer).commit());
    return std::move(*loan);
  }
  std::array<double, 3> origin_{};
  std::array<double, 9> rotation_{};
  bool quantize_{false};
  int64_t max_pulses_{0};
};
}  // namespace holoscan::radar::app::sar
