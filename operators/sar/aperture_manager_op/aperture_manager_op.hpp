// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/sar/aperture_buffer.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"

namespace holoscan::radar {
struct ApertureManagerParams {
  int64_t max_aperture_pulses{0};
  int64_t emit_interval_pulses{0};
  ApertureEmitMode emit_mode{ApertureEmitMode::kContinue};
  WindowSpec aperture_window{};
  bool wait_for_downstream{false};
};
HOLOSCAN_PARAM_SCHEMA(
    ApertureManagerParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.ApertureManagerParams", 2),
    holoscan::field<&ApertureManagerParams::max_aperture_pulses>("max_aperture_pulses"),
    holoscan::field<&ApertureManagerParams::emit_interval_pulses>("emit_interval_pulses"),
    holoscan::field<&ApertureManagerParams::emit_mode>("emit_mode"),
    holoscan::field<&ApertureManagerParams::aperture_window>("aperture_window"),
    holoscan::field<&ApertureManagerParams::wait_for_downstream>("wait_for_downstream")
        .description("When true, required Tensor emits wait on kBackpressured / "
                     "kResourceExhausted instead of failing compute(). Static. Default false."));

template <typename T>
class ApertureManagerOp : public holoscan::Operator<ApertureManagerParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit ApertureManagerOp(const MaxTensorExtents& extents);
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  int64_t retained_pulses() const { return buffer_.occupancy(); }
  int64_t admitted_pulses() const { return admitted_; }
  int64_t evicted_pulses() const { return evicted_; }
  int64_t dropped_pulses() const { return dropped_; }

  Input<Tensor> range_profiles{"range_profiles"};
  Input<Tensor> positions{"positions"};
  Input<Tensor> reference_position{"reference_position"};
  Input<uint8_t> stream_boundary{"stream_boundary"};
  Input<WireSampleGeometry> sample_geometry{"sample_geometry"};
  Input<Tensor> sample_points{"sample_points"};
  Input<Tensor> control{"control"};
  Input<Tensor> evict{"evict"};

  Output<Tensor> range_profiles_out{"range_profiles_out"};
  Output<Tensor> positions_out{"positions_out"};
  Output<Tensor> reference_position_out{"reference_position_out"};
  Output<Tensor> weight{"weight"};
  Output<WireSampleGeometry> sample_geometry_out{"sample_geometry_out"};
  Output<Tensor> sample_points_out{"sample_points_out"};
  Output<Tensor> control_out{"control_out"};

 private:
  MaxTensorExtents extents_;
  ApertureBuffer buffer_{0};
  int64_t admitted_since_emit_{0}, slot_{0};
  int64_t admitted_{0}, evicted_{0}, dropped_{0};
  std::optional<SampleGeometry> pending_geometry_;
  std::shared_ptr<Tensor> pending_points_;
  std::vector<double> stamp_window_;
  LatchedOutput geometry_output_, points_output_, control_output_, weight_output_;
  std::vector<std::shared_ptr<Tensor>> keep_;
};

extern template class ApertureManagerOp<float>;
extern template class ApertureManagerOp<double>;
}  // namespace holoscan::radar
