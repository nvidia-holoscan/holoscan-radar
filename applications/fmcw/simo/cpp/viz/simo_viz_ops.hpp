// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar::app::simo::viz {

struct RdHeatmapConfig {
  float dynamic_range_db{60.0F};
};

struct ClusterGeometryConfig {
  float detection_radius{0.006F};
  float representative_size{0.014F};
};

class SimoRdHeatmapOp : public holoscan::Operator<> {
 public:
  SimoRdHeatmapOp(const MaxTensorExtents& extents, RdHeatmapConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> cfar_power{"cfar_power"};
  holoscan::Output<holoscan::Tensor> rgba{"rgba"};

 private:
  RdHeatmapConfig config_;
  std::size_t max_byte_span_{0U};
  std::shared_ptr<holoscan::Tensor> peak_power_;
  holoscan::Tensor retained_input_;
};

class SimoClusterGeometryOp : public holoscan::Operator<> {
 public:
  SimoClusterGeometryOp(const MaxTensorExtents& extents,
                        ClusterGeometryConfig config);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Input<holoscan::Tensor> labels{"labels"};
  holoscan::Input<holoscan::Tensor> rep_idx{"rep_idx"};
  holoscan::Input<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Input<holoscan::Tensor> velocity_bins{"velocity_bins"};
  holoscan::Output<holoscan::Tensor> detection_ovals{"detection_ovals"};
  holoscan::Output<holoscan::Tensor> representative_crosses{
      "representative_crosses"};

 private:
  ClusterGeometryConfig config_;
  std::size_t max_ovals_byte_span_{0U};
  std::size_t max_crosses_byte_span_{0U};
};

namespace detail {

std::vector<std::uint8_t> make_heatmap_rgba_reference(
    const std::vector<float>& power, const std::vector<std::int64_t>& shape,
    float dynamic_range_db);

std::vector<float> make_cluster_ovals(
    const std::vector<std::int64_t>& detection_idx,
    const std::vector<std::int64_t>& labels, std::int64_t idx_cols,
    std::int64_t num_doppler, std::int64_t num_range, float radius);

std::vector<float> make_representative_crosses(
    const std::vector<std::int64_t>& rep_idx, std::int64_t idx_cols,
    std::int64_t num_doppler, std::int64_t num_range, float size);

}  // namespace detail

}  // namespace holoscan::radar::app::simo::viz
