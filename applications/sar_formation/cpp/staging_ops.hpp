// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
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
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"

namespace holoscan::radar::app::sar {
namespace detail {
inline expected<void, Error> emit_geometry(Output<WireSampleGeometry>& output,
                                           ExecutionContext& context,
                                           const SampleGeometry& geometry, TimePoint capture_time) {
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = std::vector<int64_t>{11}, .dtype = DLDataType{kDLFloat, 64U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  const auto packed = pack_sample_geometry(geometry);
  auto written = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    std::copy(packed.begin(), packed.end(), writer.data_as<double>());
  });
  if (!written)
    return make_unexpected(std::move(written).error());
  return output.emit_tensor(std::move(*loan), to_wire(geometry),
                            EmitOptions{.capture_time = capture_time});
}

inline constexpr TensorRepresentation kDevice{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = std::nullopt,
    .rank = std::nullopt};  // NOLINT(whitespace/indent_namespace)
inline constexpr TensorRepresentation kHostGeometry{
    .memory_kind = MemoryKind::kHost,
    .dtype = DLDataType{kDLFloat, 64U, 1U},
    .rank = 1U};  // NOLINT(whitespace/indent_namespace)
}  // namespace detail

// The boundary describes the current batch. Publish the current collection's
// geometry before the data trigger, then advance to the next collection.
class CollectionBoundaryStagerOp : public Operator<> {
 public:
  CollectionBoundaryStagerOp(std::vector<SampleGeometry> geometries, std::vector<double> spacings,
                             std::vector<double> frequencies)
  : geometries_(std::move(geometries)),
    spacings_(std::move(spacings)),
    frequencies_(std::move(frequencies)) {
    if (geometries_.empty() || geometries_.size() != spacings_.size() ||
        geometries_.size() != frequencies_.size()) {
      throw std::invalid_argument("CollectionBoundaryStagerOp: collection metadata size mismatch");
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.input(range_profiles).queue_depth(4U).expects_tensor(tensor_input(detail::kDevice));
    spec.input(stream_boundary).queue_depth(4U);
    spec.output(range_profiles_out).produces_tensor(tensor_output(detail::kDevice));
    spec.output(sample_geometry)
        .produces_tensor(pooled_tensor_output(detail::kHostGeometry, 1U, 11U * sizeof(double)));
    spec.output(range_bin_spacing);
    spec.output(reference_frequency);
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
    HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles, "CollectionBoundaryStagerOp"));
    HOLOSCAN_TRY(auto boundary, stream_boundary.receive_data());
    const auto time = EmitOptions{.capture_time = context.activation_time()};
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_geometry(sample_geometry, context, geometries_[index_],
                                                   context.activation_time()));
    HOLOSCAN_RETURN_IF_ERROR(range_bin_spacing.emit(spacings_[index_], time));
    HOLOSCAN_RETURN_IF_ERROR(reference_frequency.emit(frequencies_[index_], time));
    HOLOSCAN_RETURN_IF_ERROR(range_profiles_out.emit(std::move(profiles), time));
    if (boundary != 0U)
      index_ = (index_ + 1U) % geometries_.size();
    return {};
  }

  Input<Tensor> range_profiles{"range_profiles"};
  Input<uint8_t> stream_boundary{"stream_boundary"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<WireSampleGeometry> sample_geometry{"sample_geometry"};
  Output<double> range_bin_spacing{"range_bin_spacing"};
  Output<double> reference_frequency{"reference_frequency"};

 private:
  std::vector<SampleGeometry> geometries_;
  std::vector<double> spacings_, frequencies_;
  std::size_t index_{0U};
};

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
    // The cadence must close on a batch edge, so each image uses one geometry.
    const int64_t last = pulses_seen_ + shape[0] - 1;
    const auto index = std::min<std::size_t>(static_cast<std::size_t>(last / tile_pulses_),
                                             geometries_.size() - 1U);
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_geometry(sample_geometry, context, geometries_[index],
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
