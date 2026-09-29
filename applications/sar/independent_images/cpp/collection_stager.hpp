// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include "radar_app/sar/staging_detail.hpp"

namespace holoscan::radar::app::sar {
// The boundary describes the current batch. Stage the next collection's
// geometry and frequency on its closing tick, alongside EmitAndReset; the
// backprojector applies those overrides after forming the closing image.
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
    const std::size_t staged_index = boundary == 1U ? (index_ + 1U) % geometries_.size() : index_;
    HOLOSCAN_RETURN_IF_ERROR(detail::emit_geometry(
        sample_geometry, context, geometries_[staged_index], context.activation_time()));
    HOLOSCAN_RETURN_IF_ERROR(range_bin_spacing.emit(spacings_[staged_index], time));
    HOLOSCAN_RETURN_IF_ERROR(reference_frequency.emit(frequencies_[staged_index], time));
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

}  // namespace holoscan::radar::app::sar
