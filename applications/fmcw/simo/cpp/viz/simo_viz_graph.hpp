// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/operators/holoviz/holoviz_op.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/sink_desc.hpp>

#include "config_schema.hpp"  // NOLINT(build/include_subdir): app-private header
#include "radar_common/sim/scene.hpp"
#include "viz/simo_viz_ops.hpp"  // NOLINT(build/include_subdir)

namespace holoscan::radar::app::simo::viz {

inline std::array<float, 2> axis_bounds(const std::vector<double>& grid) {
  const auto bounds = std::minmax_element(grid.begin(), grid.end());
  const float lower = static_cast<float>(*bounds.first);
  const float upper = static_cast<float>(*bounds.second);
  return lower < upper ? std::array<float, 2>{lower, upper}
                       : std::array<float, 2>{lower - 0.5F, upper + 0.5F};
}

inline std::vector<TruthTarget> make_truth_targets(
    std::vector<sim::PointTarget> targets) {
  std::stable_sort(
      targets.begin(), targets.end(),
      [](const sim::PointTarget& lhs, const sim::PointTarget& rhs) {
        return sim::norm(lhs.position) < sim::norm(rhs.position);
      });
  std::vector<TruthTarget> result;
  result.reserve(targets.size());
  for (std::size_t rank = 0; rank < targets.size(); ++rank) {
    result.push_back(TruthTarget{
        .position = targets[rank].position,
        .velocity = targets[rank].velocity,
        .color_bucket = static_cast<int>(rank),
    });
  }
  return result;
}

struct SimoVizInputs {
  const holoscan::Output<holoscan::Tensor>& cfar_power;
  const holoscan::Output<holoscan::Tensor>& detection_idx;
  const holoscan::Output<holoscan::Tensor>& points;
  const holoscan::Output<holoscan::Tensor>& labels;
  const holoscan::Output<holoscan::Tensor>& sizes;
  const holoscan::Output<holoscan::Tensor>& range_bins;
  const holoscan::Output<holoscan::Tensor>& velocity_bins;
  const holoscan::Output<holoscan::Tensor>& reduced_points;
};

inline void add_simo_visualization(
    holoscan::Graph& graph, holoscan::PartitionHandle& application,
    holoscan::CompileOptions& options, const VisualizationConfig& config,
    double frame_period, const std::vector<double>& az_grid,
    const std::vector<double>& el_grid,
    const std::vector<sim::PointTarget>& truth_targets,
    std::string workflow_title,
    const SimoVizInputs& inputs) {
  auto display = graph.op<SimoDisplayOp>(
      "display_compositor",
      SimoDisplayConfig{
          .dynamic_range_db =
              static_cast<float>(config.dynamic_range_db),
          .detection_radius =
              static_cast<float>(config.detection_radius),
          .frame_period = frame_period,
          .history_frames = config.history_frames,
          .width = config.window_width,
          .height = config.window_height,
          .az_bounds = axis_bounds(az_grid),
          .el_bounds = axis_bounds(el_grid),
          .title = workflow_title,
          .truth_targets = make_truth_targets(truth_targets),
      });
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "display_compositor",
          .output_port = "rgba",
          .device = holoscan::DeviceId{0},
      });

  ::holoviz::LayerStack layers;
  layers.add_image_layer(
      "image",
      ::holoviz::ImageLayerDesc{
          .format = ::holoviz::ImageFormat::kR8G8B8A8Unorm,
          .max_width = static_cast<std::uint32_t>(config.window_width),
          .max_height = static_cast<std::uint32_t>(config.window_height)});

  const auto display_period =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>{1.0 / config.display_rate_hz});
  const auto max_input_age =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>{
              std::max(2.0 * frame_period, 2.0 / config.display_rate_hz)});
  auto holoviz = graph.op<holoscan::holoviz::HolovizOp>(
      "holoviz", std::move(layers),
      ::holoviz::WindowSinkDesc{
          .title = config.window_title + " - " + std::move(workflow_title),
          .width = static_cast<std::uint32_t>(config.window_width),
          .height = static_cast<std::uint32_t>(config.window_height)},
      display_period, max_input_age);

  // Keep the independently clock-driven display in its own partition.
  application.add(display);
  graph.partition("visualization").add(holoviz);

  const holoscan::ConnectionOptions processing_queue{.queue_depth = 4U};
  const holoscan::ConnectionOptions display_queue{.queue_depth = 8U};

  graph.add_flow(inputs.cfar_power, display->cfar_power, processing_queue);
  graph.add_flow(inputs.detection_idx, display->detection_idx,
                 processing_queue);
  graph.add_flow(inputs.points, display->points, processing_queue);
  graph.add_flow(inputs.labels, display->labels, processing_queue);
  graph.add_flow(inputs.sizes, display->sizes, processing_queue);
  graph.add_flow(inputs.range_bins, display->range_bins, processing_queue);
  graph.add_flow(inputs.velocity_bins, display->velocity_bins,
                 processing_queue);
  graph.add_flow(inputs.reduced_points, display->reduced_points,
                 processing_queue);
  graph.add_flow(display->rgba, holoviz->layer_input(0), display_queue);
}

}  // namespace holoscan::radar::app::simo::viz
