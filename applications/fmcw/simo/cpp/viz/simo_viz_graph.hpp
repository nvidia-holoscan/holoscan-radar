// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/operators/holoviz/holoviz_op.hpp>
#include <holoviz/geometry_topology.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/sink_desc.hpp>

#include "config_schema.hpp"  // NOLINT(build/include_subdir): app-private header
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "viz/simo_viz_ops.hpp"  // NOLINT(build/include_subdir)

namespace holoscan::radar::app::simo::viz {

struct SimoVizInputs {
  const holoscan::Output<holoscan::Tensor>& cfar_power;
  const holoscan::Output<holoscan::Tensor>& detection_idx;
  const holoscan::Output<holoscan::Tensor>& labels;
  const holoscan::Output<holoscan::Tensor>& rep_idx;
  const holoscan::Output<holoscan::Tensor>& range_bins;
  const holoscan::Output<holoscan::Tensor>& velocity_bins;
};

inline void add_simo_visualization(
    holoscan::Graph& graph, holoscan::PartitionHandle& application,
    holoscan::CompileOptions& options, const MaxTensorExtents& extents,
    const VisualizationConfig& config, double frame_period,
    std::string workflow_title, const SimoVizInputs& inputs) {
  auto heatmap = graph.op<SimoRdHeatmapOp>(
      "rd_heatmap", extents,
      RdHeatmapConfig{static_cast<float>(config.dynamic_range_db)});
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "rd_heatmap",
          .output_port = "rgba",
          .device = holoscan::DeviceId{0},
      });

  auto geometry = graph.op<SimoClusterGeometryOp>(
      "cluster_geometry", extents,
      ClusterGeometryConfig{
          static_cast<float>(config.detection_radius),
          static_cast<float>(config.representative_size)});

  ::holoviz::LayerStack layers;
  layers.add_image_layer(
      "image",
      ::holoviz::ImageLayerDesc{
          .format = ::holoviz::ImageFormat::kR8G8B8A8Unorm,
          .max_width =
              static_cast<std::uint32_t>(*extents.max_range_bins),
          .max_height =
              static_cast<std::uint32_t>(*extents.max_doppler_bins)});
  layers.add_geometry_layer(
      "clustered_detections",
      ::holoviz::GeometryLayerDesc{
          {::holoviz::GeometryTopology::kOvalList},
          static_cast<std::uint32_t>(*extents.max_detections), 2.0F});
  layers.add_geometry_layer(
      "cluster_representatives",
      ::holoviz::GeometryLayerDesc{
          {::holoviz::GeometryTopology::kCrossList},
          static_cast<std::uint32_t>(*extents.max_clusters), 3.0F});

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
  application.add(heatmap).add(geometry);
  graph.partition("visualization").add(holoviz);

  const holoscan::ConnectionOptions processing_queue{.queue_depth = 4U};
  const holoscan::ConnectionOptions display_queue{.queue_depth = 8U};

  graph.add_flow(inputs.cfar_power, heatmap->cfar_power, processing_queue);
  graph.add_flow(inputs.detection_idx, geometry->detection_idx,
                 processing_queue);
  graph.add_flow(inputs.labels, geometry->labels, processing_queue);
  graph.add_flow(inputs.rep_idx, geometry->rep_idx, processing_queue);
  graph.add_flow(inputs.range_bins, geometry->range_bins, processing_queue);
  graph.add_flow(inputs.velocity_bins, geometry->velocity_bins,
                 processing_queue);
  graph.add_flow(heatmap->rgba, holoviz->layer_input(0), display_queue);
  graph.add_flow(geometry->detection_ovals, holoviz->layer_input(1),
                 display_queue);
  graph.add_flow(geometry->representative_crosses, holoviz->layer_input(2),
                 display_queue);
}

}  // namespace holoscan::radar::app::simo::viz
