// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
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

#include "radar_app/app/angle_presentation.hpp"
#include "radar_app/app/visualization_config.hpp"
#include "radar_app/viz/fmcw_dashboard_op.hpp"
#include "radar_common/sim/scene.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar::app::viz {

inline std::array<float, 2> display_bounds(const std::array<double, 2>& bounds) {
  const float lower = static_cast<float>(bounds[0]);
  const float upper = static_cast<float>(bounds[1]);
  return lower < upper ? std::array<float, 2>{lower, upper}
                       : std::array<float, 2>{lower - 0.5F, upper + 0.5F};
}

inline std::vector<TruthTarget> make_truth_targets(std::vector<sim::PointTarget> targets) {
  std::stable_sort(targets.begin(), targets.end(),
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

struct RawDashboardInputs {
  const holoscan::Output<holoscan::Tensor>& cfar_power;
  const holoscan::Output<holoscan::Tensor>& detection_idx;
  const holoscan::Output<holoscan::Tensor>& points;
  const holoscan::Output<holoscan::Tensor>& range_bins;
  const holoscan::Output<holoscan::Tensor>& velocity_bins;
};

struct ClusterDashboardInputs {
  const holoscan::Output<holoscan::Tensor>& cfar_power;
  const holoscan::Output<holoscan::Tensor>& detection_idx;
  const holoscan::Output<holoscan::Tensor>& points;
  const holoscan::Output<holoscan::Tensor>& labels;
  const holoscan::Output<holoscan::Tensor>& sizes;
  const holoscan::Output<holoscan::Tensor>& range_bins;
  const holoscan::Output<holoscan::Tensor>& velocity_bins;
  const holoscan::Output<holoscan::Tensor>& reduced_points;
};

struct ProfiledClusterDashboardInputs {
  const holoscan::Output<holoscan::Tensor>& cfar_power;
  const holoscan::Output<holoscan::Tensor>& detection_idx;
  const holoscan::Output<holoscan::Tensor>& points;
  const holoscan::Output<holoscan::Tensor>& labels;
  const holoscan::Output<holoscan::Tensor>& sizes;
  const holoscan::Output<holoscan::Tensor>& range_bins;
  const holoscan::Output<holoscan::Tensor>& velocity_bins;
  const holoscan::Output<holoscan::Tensor>& reduced_points;
  const holoscan::Output<wire::FMCWWaveformT>& waveform;
  FMCWWaveformDescriptor selected_waveform;
  std::size_t max_profile_gap_frames{1U};
};

template <typename Inputs>
inline void add_fmcw_dashboard(holoscan::Graph& graph, holoscan::PartitionHandle& application,
                               holoscan::CompileOptions& options, const VisualizationConfig& config,
                               double frame_period, bool show_angles,
                               const std::vector<double>& az_grid,
                               const std::vector<double>& el_grid,
                               const AngleConvention& angle_convention,
                               const std::vector<sim::PointTarget>& truth_targets,
                               const sim::SceneVisibility& scene_visibility,
                               std::string workflow_title, const Inputs& inputs) {
  static_assert(std::is_same_v<Inputs, RawDashboardInputs> ||
                    std::is_same_v<Inputs, ClusterDashboardInputs> ||
                    std::is_same_v<Inputs, ProfiledClusterDashboardInputs>,
                "unsupported FMCW dashboard input bundle");
  constexpr bool kClustered = std::is_same_v<Inputs, ClusterDashboardInputs> ||
                              std::is_same_v<Inputs, ProfiledClusterDashboardInputs>;
  constexpr bool kProfiled = std::is_same_v<Inputs, ProfiledClusterDashboardInputs>;
  std::string selected_profile;
  FMCWWaveformDescriptor selected_waveform;
  std::size_t max_profile_gap_frames = 1U;
  if constexpr (kProfiled) {
    selected_profile = config.profile;
    selected_waveform = inputs.selected_waveform;
    max_profile_gap_frames = inputs.max_profile_gap_frames;
  }
  const PhysicalAngleBounds physical_bounds =
      show_angles ? physical_scan_grid_bounds(angle_convention, az_grid, el_grid)
                  : PhysicalAngleBounds{{-1.0, 1.0}, {-1.0, 1.0}};
  auto display = graph.op<FmcwDashboardOp>(
      "display_compositor",
      FmcwDashboardConfig{
          .mode = kClustered ? DashboardMode::kClusters : DashboardMode::kRawDetections,
          .show_angles = show_angles,
          .dynamic_range_db = static_cast<float>(config.dynamic_range_db),
          .detection_radius = static_cast<float>(config.detection_radius),
          .frame_period = frame_period,
          .history_frames = config.history_frames,
          .detection_trail_frames = config.detection_trail_frames,
          .width = config.window_width,
          .height = config.window_height,
          .az_bounds = display_bounds(physical_bounds.az),
          .el_bounds = display_bounds(physical_bounds.el),
          .angle_convention = angle_convention,
          .title =
              workflow_title + (selected_profile.empty() ? "" : " | PROFILE: " + selected_profile),
          .selected_profile = std::move(selected_profile),
          .selected_waveform = std::move(selected_waveform),
          .truth_targets = make_truth_targets(truth_targets),
          .scene_visibility = scene_visibility,
      });
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "display_compositor",
      .output_port = "rgba",
      .device = holoscan::DeviceId{0},
  });

  ::holoviz::LayerStack layers;
  layers.add_image_layer(
      "image",
      ::holoviz::ImageLayerDesc{.format = ::holoviz::ImageFormat::kR8G8B8A8Unorm,
                                .max_width = static_cast<std::uint32_t>(config.window_width),
                                .max_height = static_cast<std::uint32_t>(config.window_height)});

  const auto display_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>{1.0 / config.display_rate_hz});
  const auto max_input_age =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>{
          std::max(static_cast<double>(max_profile_gap_frames + 1U) * frame_period,
                   2.0 / config.display_rate_hz)});
  auto holoviz = graph.op<holoscan::holoviz::HolovizOp>(
      "holoviz", std::move(layers),
      ::holoviz::WindowSinkDesc{.title = config.window_title + " - " + std::move(workflow_title),
                                .width = static_cast<std::uint32_t>(config.window_width),
                                .height = static_cast<std::uint32_t>(config.window_height)},
      display_period, max_input_age);

  // Keep the independently clock-driven display in its own partition.
  application.add(display);
  graph.partition("visualization").add(holoviz);

  // Visualization is best-effort: discard stale frames instead of
  // backpressuring the radar pipeline when rendering falls behind.
  const holoscan::ConnectionOptions processing_queue{
      .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U};
  const holoscan::ConnectionOptions display_queue{
      .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 8U};

  graph.add_flow(inputs.cfar_power, display->cfar_power, processing_queue);
  graph.add_flow(inputs.detection_idx, display->detection_idx, processing_queue);
  graph.add_flow(inputs.points, display->points, processing_queue);
  if constexpr (kClustered) {
    graph.add_flow(inputs.labels, display->labels, processing_queue);
    graph.add_flow(inputs.sizes, display->sizes, processing_queue);
  }
  graph.add_flow(inputs.range_bins, display->range_bins, processing_queue);
  graph.add_flow(inputs.velocity_bins, display->velocity_bins, processing_queue);
  if constexpr (kClustered) {
    graph.add_flow(inputs.reduced_points, display->reduced_points, processing_queue);
  }
  if constexpr (kProfiled) {
    const holoscan::ConnectionOptions waveform_queue{.queue_depth = max_profile_gap_frames + 2U};
    graph.add_flow(inputs.waveform, display->waveform_in, waveform_queue);
  }
  graph.add_flow(display->rgba, holoviz->layer_input(0), display_queue);
}

}  // namespace holoscan::radar::app::viz
