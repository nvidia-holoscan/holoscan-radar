// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar {

struct DopplerNotchParams {
  double center_velocity{0.0};
  double stop_half_width{0.0};
  double transition_width{0.0};
  bool cyclic_velocity_axis{true};
};

HOLOSCAN_PARAM_SCHEMA(
    DopplerNotchParams,
    holoscan::param_schema("com.nvidia.holoscan.radar.DopplerNotchParams", 1),
    holoscan::field<&DopplerNotchParams::center_velocity>("center_velocity")
        .description("Notch center velocity in m/s. Default: 0."),
    holoscan::field<&DopplerNotchParams::stop_half_width>("stop_half_width")
        .description(
            "Full-rejection half-width in m/s. The nearest bin is always "
            "nulled. Default: 0."),
    holoscan::field<&DopplerNotchParams::transition_width>("transition_width")
        .description(
            "Raised-cosine transition width in m/s; 0 selects a hard notch. "
            "Default: 0."),
    holoscan::field<&DopplerNotchParams::cyclic_velocity_axis>(
        "cyclic_velocity_axis")
        .description(
            "Measure distance cyclically over the aliased velocity interval. "
            "Static. Default: true."));

static_assert(holoscan::param_schema_availability<DopplerNotchParams>() ==
                  holoscan::ParamSchemaAvailability::kDeclared,
              "HOLOSCAN_PARAM_SCHEMA must be in DopplerNotchParams's "
              "namespace");

template <typename T>
class DopplerNotchOp : public holoscan::Operator<DopplerNotchParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "DopplerNotchOp supports float and double precision");

 public:
  explicit DopplerNotchOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  Input<Tensor> velocity_bins{"velocity_bins"};
  Output<Tensor> rd_maps_out{"rd_maps_out"};

  Input<double> center_velocity{"center_velocity"};
  Input<double> stop_half_width{"stop_half_width"};
  Input<double> transition_width{"transition_width"};

 private:
  Tensor retained_rd_maps_;
  Tensor retained_velocity_bins_;
  MaxTensorExtents extents_;
  std::uint32_t max_rank_{2U};
  std::size_t max_byte_span_{0U};
};

extern template class DopplerNotchOp<float>;
extern template class DopplerNotchOp<double>;

}  // namespace holoscan::radar
