// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"
#include "radar_app/sar/staging_detail.hpp"

namespace holoscan::radar::app::sar {
class ScrollGeometryStagerOp : public Operator<> {
 public:
  ScrollGeometryStagerOp(SampleGeometry parent, SampleGeometry window, int64_t anchor, bool loop,
                         int64_t slice_pulses, int64_t max_pulses, int64_t max_bins,
                         std::shared_ptr<QueueWaitPolicy> queue_wait = {})
  : parent_(std::move(parent)),
    window_(std::move(window)),
    anchor_(anchor),
    loop_(loop),
    slice_pulses_(slice_pulses),
    max_pulses_(max_pulses),
    max_bins_(max_bins),
    queue_wait_(std::move(queue_wait)) {
    if (!parent_.is_valid() || !window_.is_valid() || window_.num_v() > parent_.num_v() ||
        window_.num_u() != parent_.num_u() || anchor_ < 0 || anchor_ >= window_.num_v() ||
        max_pulses_ < 1 || max_bins_ < 1) {
      throw std::invalid_argument("ScrollGeometryStagerOp: invalid geometry or extents");
    }
    if (slice_pulses_ < 1)
      slice_pulses_ = max_pulses_;
    capacity_ = static_cast<std::uint32_t>((max_pulses_ + slice_pulses_ - 1) / slice_pulses_);
  }
  void setup(OperatorSpec& spec) override {
    const auto queue = std::max<std::uint32_t>(4U, capacity_ + 2U);
    spec.input(range_profiles).queue_depth(queue).expects_tensor(tensor_input(detail::kDevice));
    spec.input(positions).queue_depth(queue).expects_tensor(tensor_input(detail::kDevice));
    spec.input(reference_position).queue_depth(queue).expects_tensor(tensor_input(detail::kDevice));
    spec.input(stream_boundary).queue_depth(queue);
    spec.output(range_profiles_out)
        .max_emits_per_compute(capacity_)
        .produces_tensor(pooled_tensor_output(
            detail::kDevice, 2U,
            static_cast<uint64_t>(slice_pulses_) * static_cast<uint64_t>(max_bins_) * 16U));
    for (auto* port : {&positions_out, &reference_position_out}) {
      spec.output(*port).max_emits_per_compute(capacity_).produces_tensor(pooled_tensor_output(
          detail::kDevice, 2U, static_cast<uint64_t>(slice_pulses_) * 3U * sizeof(double)));
    }
    spec.output(sample_geometry)
        .max_emits_per_compute(capacity_)
        .produces_tensor(pooled_tensor_output(detail::kHostGeometry, 1U, 11U * sizeof(double)));
    spec.output(stream_boundary_out).max_emits_per_compute(capacity_);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{
            .ports = ports(range_profiles, positions, reference_position, stream_boundary),
            .window = std::chrono::nanoseconds{0}})
        .input(range_profiles, Exact{})
        .input(positions, Exact{})
        .input(reference_position, Exact{})
        .input(stream_boundary, Exact{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles, "ScrollGeometryStagerOp"));
    HOLOSCAN_TRY(auto pos, receive_tensor(positions, "ScrollGeometryStagerOp"));
    HOLOSCAN_TRY(auto ref, receive_tensor(reference_position, "ScrollGeometryStagerOp"));
    HOLOSCAN_TRY(auto boundary, stream_boundary.receive_data());
    const auto ps = profiles.shape(), xs = pos.shape(), rs = ref.shape();
    if (ps.size() != 2U || xs.size() != 2U || rs.size() != 2U || ps[0] < 1 || ps[0] > max_pulses_ ||
        ps[1] < 1 || ps[1] > max_bins_ || xs[0] != ps[0] || rs[0] != ps[0] || xs[1] != 3 ||
        rs[1] != 3 || pos.dtype().bits != 64U || ref.dtype().bits != 64U) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "ScrollGeometryStagerOp: pulse tensor mismatch"});
    }
    const auto stream = context.cuda_stream().get();
    const int64_t np = ps[0];
    const int64_t step = std::min(np, slice_pulses_);
    for (int64_t first = 0; first < np; first += step) {
      const int64_t count = std::min(step, np - first);
      const bool last = first + count == np;
      std::array<double, 3> srp{};
      auto* source = static_cast<const char*>(ref.data()) +
                     static_cast<std::size_t>(first + count - 1) * 3U * sizeof(double);
      if (cudaMemcpyAsync(srp.data(), source, 3U * sizeof(double), cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess ||
          cudaStreamSynchronize(stream) != cudaSuccess) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "ScrollGeometryStagerOp: SRP read failed"});
      }
      if (loop_ && last && boundary == 1U) {
        v0_ = 0;
      } else {
        const auto& origin = parent_.origin();
        const auto& av = parent_.axis_v();
        double project = 0.0;
        for (int k = 0; k < 3; ++k)
          project += (srp[k] - origin[k]) * av[k];
        const int64_t desired =
            std::clamp(static_cast<int64_t>(std::llround(project / parent_.spacing()[0])) - anchor_,
                       int64_t{0}, parent_.num_v() - window_.num_v());
        v0_ = std::max(v0_, desired);
      }
      auto slice_time = context.activation_time();
      slice_time.timestamp_ns += first / step;
      const auto time = EmitOptions{.capture_time = slice_time};
      const bool wait = wait_queues(queue_wait_);
      SampleGeometrySpec spec{.origin = parent_.sample_position(v0_, 0),
                              .axis_u = parent_.axis_u(),
                              .axis_v = parent_.axis_v(),
                              .spacing = parent_.spacing(),
                              .counts = {window_.num_v(), parent_.num_u()}};
      HOLOSCAN_RETURN_IF_ERROR(detail::emit_geometry(sample_geometry, context, SampleGeometry(spec),
                                                     slice_time, wait,
                                                     "ScrollGeometryStagerOp.sample_geometry"));
      HOLOSCAN_RETURN_IF_ERROR(emit_required_tensor(
          positions_out, context, "ScrollGeometryStagerOp.positions_out", wait,
          [&] {
            return slice(positions_out, pos, first, count, context);
          },
          time));
      HOLOSCAN_RETURN_IF_ERROR(emit_required_tensor(
          reference_position_out, context, "ScrollGeometryStagerOp.reference_position_out", wait,
          [&] {
            return slice(reference_position_out, ref, first, count, context);
          },
          time));
      HOLOSCAN_RETURN_IF_ERROR(emit_required_value(stream_boundary_out, context,
                                                   "ScrollGeometryStagerOp.stream_boundary_out",
                                                   wait, last ? boundary : uint8_t{0}, time));
      HOLOSCAN_RETURN_IF_ERROR(emit_required_tensor(
          range_profiles_out, context, "ScrollGeometryStagerOp.range_profiles_out", wait,
          [&] {
            return slice(range_profiles_out, profiles, first, count, context);
          },
          time));
    }
    return {};
  }
  Input<Tensor> range_profiles{"range_profiles"};
  Input<Tensor> positions{"positions"};
  Input<Tensor> reference_position{"reference_position"};
  Input<uint8_t> stream_boundary{"stream_boundary"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<Tensor> positions_out{"positions_out"};
  Output<Tensor> reference_position_out{"reference_position_out"};
  Output<WireSampleGeometry> sample_geometry{"sample_geometry"};
  Output<uint8_t> stream_boundary_out{"stream_boundary_out"};

 private:
  expected<TensorOutputLoan, Error> slice(Output<Tensor>& output, const Tensor& input,
                                          int64_t first, int64_t count,
                                          ExecutionContext& context) const {
    const auto shape = input.shape();
    const auto dtype = input.dtype();
    const auto element = static_cast<std::size_t>(dtype.bits / 8U) * dtype.lanes;
    const auto row_bytes = static_cast<std::size_t>(shape[1]) * element;
    const auto* source =
        static_cast<const char*>(input.data()) + static_cast<std::size_t>(first) * row_bytes;
    return prepare_device_tensor(
        output, context, std::vector<int64_t>{count, shape[1]}, dtype,
        [&](void* destination, std::size_t bytes) -> expected<void, Error> {
          if (bytes != static_cast<std::size_t>(count) * row_bytes ||
              cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice,
                              context.cuda_stream().get()) != cudaSuccess)
            return make_unexpected(
                Error{ErrorCode::kFailure, "ScrollGeometryStagerOp: slice copy failed"});
          return {};
        });
  }
  SampleGeometry parent_, window_;
  int64_t anchor_{0};
  bool loop_{false};
  int64_t slice_pulses_{0}, max_pulses_{0}, max_bins_{0};
  std::shared_ptr<QueueWaitPolicy> queue_wait_{};
  std::uint32_t capacity_{1U};
  int64_t v0_{0};
};
}  // namespace holoscan::radar::app::sar
