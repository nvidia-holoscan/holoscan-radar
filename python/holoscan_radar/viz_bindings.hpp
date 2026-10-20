// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

#include "radar_app/viz/fmcw_dashboard_graph.hpp"

namespace holoscan::radar::python_viz {

namespace nb = nanobind;
namespace authoring = holoscan::authoring;
namespace viz = holoscan::radar::app::viz;

struct VizPath {
  std::string path;
};
HOLOSCAN_PARAM_SCHEMA(VizPath, holoscan::param_schema("com.nvidia.holoscan.radar.VizPath", 1),
                      holoscan::field<&VizPath::path>("path"));

template <typename Loader>
struct DashboardThunks : authoring::ManualOperatorRegistrationThunks<viz::FmcwDashboardOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const VizPath*>(params->data());
    auto cfg = Loader::load(std::filesystem::path(input->path));
    auto dashboard = Loader::dashboard(cfg);
    return DashboardThunks::make_operator_ref(
        graph.op<viz::FmcwDashboardOp>(name, std::move(dashboard)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(holoscan::param_type_descriptor<VizPath>(),
                                                    holoscan::param_schema_descriptor<VizPath>(),
                                                    assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return DashboardThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kNotSupported,
                                 "dashboard configuration is constructor-only"};
  }
};

template <typename Loader>
void register_dashboard(std::string_view name) {
  using Thunks = DashboardThunks<Loader>;
  auto result = authoring::authoring_registry().register_operator({
      .name = name,
      .param_type = &holoscan::param_type_descriptor<VizPath>(),
      .param_schema = holoscan::param_schema_descriptor<VizPath>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &Thunks::from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  });
  if (!result) {
    throw holoscan::RuntimeError{result.error()};
  }
}

template <typename Config>
viz::FmcwDashboardConfig dashboard_config(const Config& cfg, std::string title, bool clustered,
                                          bool show_angles, const std::vector<double>& az_grid,
                                          const std::vector<double>& el_grid,
                                          const AngleConvention& angle_convention) {
  const auto bounds = show_angles
                          ? app::physical_scan_grid_bounds(angle_convention, az_grid, el_grid)
                          : app::PhysicalAngleBounds{{-1.0, 1.0}, {-1.0, 1.0}};
  viz::FmcwDashboardConfig result;
  result.mode = clustered ? viz::DashboardMode::kClusters : viz::DashboardMode::kRawDetections;
  result.show_angles = show_angles;
  result.dynamic_range_db = static_cast<float>(cfg.visualization.dynamic_range_db);
  result.range_doppler_width_percent = cfg.visualization.range_doppler_width_percent;
  result.detection_radius = static_cast<float>(cfg.visualization.detection_radius);
  result.frame_period = cfg.frame_period;
  result.display_rate_hz = cfg.visualization.display_rate_hz;
  result.history_frames = cfg.visualization.history_frames;
  result.detection_trail_frames = cfg.visualization.detection_trail_frames;
  result.width = cfg.visualization.window_width;
  result.height = cfg.visualization.window_height;
  result.az_bounds = viz::display_bounds(bounds.az);
  result.el_bounds = viz::display_bounds(bounds.el);
  result.angle_convention = angle_convention;
  result.title = std::move(title);
  result.truth_targets = viz::make_truth_targets(cfg.scene.targets);
  result.scene_visibility = cfg.scene_visibility;
  return result;
}

inline nb::dict visualization_settings(const app::VisualizationConfig& config,
                                       double frame_period) {
  nb::dict result;
  result["window_width"] = config.window_width;
  result["window_height"] = config.window_height;
  result["window_title"] = config.window_title;
  result["display_rate_hz"] = config.display_rate_hz;
  result["frame_period"] = frame_period;
  return result;
}

}  // namespace holoscan::radar::python_viz
