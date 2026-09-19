// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

struct RankOneResidualScreenParams {
  ArrayGeometry array_geometry{};
  AngleConvention angle_convention{};
  double residual_threshold{0.10};
};

HOLOSCAN_PARAM_SCHEMA(
    RankOneResidualScreenParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.RankOneResidualScreenParams", 1),
    holoscan::field<&RankOneResidualScreenParams::array_geometry>(
        "array_geometry")
        .description("Element positions and wavelength; live Exact override"),
    holoscan::field<&RankOneResidualScreenParams::angle_convention>(
        "angle_convention")
        .description("Static convention of the input baseline angles"),
    holoscan::field<&RankOneResidualScreenParams::residual_threshold>(
        "residual_threshold")
        .description("Rank-one residual threshold in [0,1]; static"));

static_assert(
    holoscan::param_schema_availability<RankOneResidualScreenParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in RankOneResidualScreenParams namespace");

template <typename T>
class RankOneResidualScreenOp
    : public holoscan::Operator<RankOneResidualScreenParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "RankOneResidualScreenOp supports float and double precision");

 public:
  explicit RankOneResidualScreenOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> snapshot{"snapshot"};
  Input<Tensor> baseline_aoa{"baseline_aoa"};
  Input<Tensor> is_valid{"is_valid"};
  Input<WireArrayGeometry> array_geometry{"array_geometry"};

  Output<Tensor> snapshot_out{"snapshot_out"};
  Output<Tensor> selected_idx{"selected_idx"};
  Output<Tensor> is_valid_out{"is_valid_out"};
  Output<Tensor> residual_score{"residual_score"};
  Output<Tensor> selection_mask{"selection_mask"};

 private:
  struct RuntimeState;
  std::shared_ptr<RuntimeState> state_;
  Tensor retained_snapshot_;
  Tensor retained_aoa_;
  Tensor retained_valid_;
  std::int64_t max_items_{0};
  std::int64_t max_channels_{0};
  std::int64_t max_snapshots_{0};
  std::size_t snapshot_max_byte_span_{0U};
  std::size_t selected_idx_max_byte_span_{0U};
  std::size_t valid_max_byte_span_{0U};
  std::size_t score_max_byte_span_{0U};
  std::size_t mask_max_byte_span_{0U};
};

extern template class RankOneResidualScreenOp<float>;
extern template class RankOneResidualScreenOp<double>;

}  // namespace holoscan::radar
