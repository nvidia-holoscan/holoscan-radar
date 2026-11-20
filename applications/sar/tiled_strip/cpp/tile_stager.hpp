// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "radar_app/sar/staging_detail.hpp"

namespace holoscan::radar::app::sar {
class TileGeometryStagerOp : public Operator<> {
 public:
  TileGeometryStagerOp(std::vector<SampleGeometry> geometries, int64_t tile_pulses)
  : geometries_(std::move(geometries)), tile_pulses_(tile_pulses) {
    if (geometries_.size() < 2U || tile_pulses_ < 1)
      throw std::invalid_argument("TileGeometryStagerOp: requires tiles and positive cadence");
  }
  void setup(OperatorSpec& spec) override {
    spec.input(range_profiles).queue_depth(4U).expects_tensor(tensor_input(detail::kDevice));
    spec.input(stream_boundary).queue_depth(4U);
    spec.output(range_profiles_out).produces_tensor(tensor_output(detail::kDevice));
    spec.output(sample_geometry)
        .produces_tensor(pooled_tensor_output(detail::kHostGeometry, 1U, 11U * sizeof(double)));
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnAligned{.ports = ports(range_profiles, stream_boundary),
                           .window = std::chrono::nanoseconds{0}})
        .input(range_profiles, Exact{})
        .input(stream_boundary, Exact{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles, "TileGeometryStagerOp"));
    HOLOSCAN_TRY(auto boundary, stream_boundary.receive_data());
    const auto shape = profiles.shape();
    if (shape.size() != 2U || shape[0] < 1)
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "TileGeometryStagerOp: expected pulse x range tensor"});
    // The cadence closes on a batch edge. Stage the next tile with this
    // batch's EmitAndReset so backprojection forms the closing tile first.
    const int64_t end = pulses_seen_ + shape[0];
    const auto current = std::min<std::size_t>(static_cast<std::size_t>((end - 1) / tile_pulses_),
                                               geometries_.size() - 1U);
    std::size_t staged = current;
    if (boundary == 1U) {
      staged = 0U;  // A looping collection restarts at tile zero.
    } else if (boundary == 0U && end % tile_pulses_ == 0 && current + 1U < geometries_.size()) {
      staged = current + 1U;
    }
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_geometry(sample_geometry, context, geometries_[staged],
                                                   context.activation_time()));
    HOLOSCAN_RETURN_IF_ERROR(range_profiles_out.emit(
        std::move(profiles), EmitOptions{.capture_time = context.activation_time()}));
    pulses_seen_ += shape[0];
    if (boundary != 0U)
      pulses_seen_ = 0;
    return {};
  }
  Input<Tensor> range_profiles{"range_profiles"};
  Input<uint8_t> stream_boundary{"stream_boundary"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<WireSampleGeometry> sample_geometry{"sample_geometry"};

 private:
  std::vector<SampleGeometry> geometries_;
  int64_t tile_pulses_{0};
  int64_t pulses_seen_{0};
};
}  // namespace holoscan::radar::app::sar
