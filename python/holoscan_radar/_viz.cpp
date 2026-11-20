// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <cstdint>
#include <type_traits>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/viz/presentation_graph.hpp"

#include "radar_app/viz/rgba_image_bridge_op.hpp"
#include "radar_app/viz/image_device_bridge_op.hpp"
#include "radar_app/viz/fmcw_holoviz_op.hpp"
#if HOLOSCAN_RADAR_HAS_X11
#include "radar_app/viz/x11_image_presenter_op.hpp"
#endif

namespace authoring = holoscan::authoring;
namespace viz = holoscan::radar::app::viz;

namespace {

struct ImageSize {
  std::int32_t width;
  std::int32_t height;
  double rate{60.0};
  bool fit{false};
  std::string title;
};
HOLOSCAN_PARAM_SCHEMA(ImageSize, holoscan::param_schema("com.nvidia.holoscan.radar.ImageSize", 1),
                      holoscan::field<&ImageSize::width>("width"),
                      holoscan::field<&ImageSize::height>("height"),
                      holoscan::field<&ImageSize::rate>("rate"),
                      holoscan::field<&ImageSize::fit>("fit"),
                      holoscan::field<&ImageSize::title>("title"));

template <typename Op>
struct SizeThunks : authoring::ManualOperatorRegistrationThunks<Op> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const ImageSize*>(params->data());
    if (input->width <= 0 || input->height <= 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                   "dashboard image width and height must be positive"};
    }
    if constexpr (std::is_same_v<Op, viz::CompletedFrameSinkOp>)
      return SizeThunks::make_operator_ref(
          graph.op<Op>(name, holoscan::radar::app::authoring_frame_progress()));
    else if constexpr (std::is_same_v<Op, viz::CompletedFrameReplayOp>)
      return SizeThunks::make_operator_ref(
          graph.op<Op>(name, input->width, input->height, input->rate,
                       holoscan::radar::app::authoring_frame_progress(), input->fit, input->title));
    else
      return SizeThunks::make_operator_ref(graph.op<Op>(name, input->width, input->height));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(holoscan::param_type_descriptor<ImageSize>(),
                                                    holoscan::param_schema_descriptor<ImageSize>(),
                                                    assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return SizeThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kNotSupported,
                                 "image dimensions are constructor-only"};
  }
};

struct DashboardWindow {
  std::int32_t width;
  std::int32_t height;
  std::string title;
  bool fit{false};
};
HOLOSCAN_PARAM_SCHEMA(DashboardWindow,
                      holoscan::param_schema("com.nvidia.holoscan.radar.DashboardWindow", 1),
                      holoscan::field<&DashboardWindow::width>("width"),
                      holoscan::field<&DashboardWindow::height>("height"),
                      holoscan::field<&DashboardWindow::title>("title"),
                      holoscan::field<&DashboardWindow::fit>("fit"));

template <typename Op>
struct ViewerThunks : authoring::ManualOperatorRegistrationThunks<Op> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const DashboardWindow*>(params->data());
    if (input->width <= 0 || input->height <= 0) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                   "dashboard image width and height must be positive"};
    }
#if HOLOSCAN_RADAR_HAS_X11
    if constexpr (std::is_same_v<Op, viz::X11ImagePresenterOp>)
      return ViewerThunks::make_operator_ref(
          graph.op<Op>(name, input->width, input->height, input->title, input->fit,
                       holoscan::radar::app::authoring_frame_progress()));
    else
#endif
      return ViewerThunks::make_operator_ref(
          graph.op<Op>(name, input->width, input->height, input->title));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<DashboardWindow>(),
        holoscan::param_schema_descriptor<DashboardWindow>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return ViewerThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kNotSupported,
                                 "dashboard window dimensions are constructor-only"};
  }
};

template <typename Op>
void register_transfer(std::string_view name) {
  using T = SizeThunks<Op>;
  auto result = authoring::authoring_registry().register_operator({
      .name = name,
      .param_type = &holoscan::param_type_descriptor<ImageSize>(),
      .param_schema = holoscan::param_schema_descriptor<ImageSize>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &T::add_to_graph,
      .add_to_graph_from_assignments = &T::from_assignments,
      .set_params = &T::set_params,
      .port = &T::port,
      .ports = &T::ports,
      .region_add = &T::region_add,
      .partition_add = &T::partition_add,
      .lifecycle_before = &T::lifecycle_before,
      .lifecycle_allow_self_restart = &T::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &T::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &T::lifecycle_allow_restart_request,
  });
  if (!result)
    throw holoscan::RuntimeError{result.error()};
}

template <typename Op>
void register_viewer(std::string_view name) {
  using V = ViewerThunks<Op>;
  auto viewer_result = authoring::authoring_registry().register_operator({
      .name = name,
      .param_type = &holoscan::param_type_descriptor<DashboardWindow>(),
      .param_schema = holoscan::param_schema_descriptor<DashboardWindow>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &V::add_to_graph,
      .add_to_graph_from_assignments = &V::from_assignments,
      .set_params = &V::set_params,
      .port = &V::port,
      .ports = &V::ports,
      .region_add = &V::region_add,
      .partition_add = &V::partition_add,
      .lifecycle_before = &V::lifecycle_before,
      .lifecycle_allow_self_restart = &V::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &V::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &V::lifecycle_allow_restart_request,
  });
  if (!viewer_result) {
    throw holoscan::RuntimeError{viewer_result.error()};
  }
}

}  // namespace

void register_viz_bindings(nanobind::module_&) {
  register_transfer<viz::CompletedFrameSinkOp>("holoscan.radar._app.CompletedFrameSink");
  register_transfer<viz::CompletedFrameReplayOp>("holoscan.radar._app.CompletedFrameReplay");
  register_transfer<viz::ImageDownloadOp>("holoscan.radar._app.ImageDownload");
  register_transfer<viz::ImageUploadOp>("holoscan.radar._app.ImageUpload");
  using T = SizeThunks<viz::RgbaImageBridgeOp>;
  auto result = authoring::authoring_registry().register_operator({
      .name = "holoscan.radar._app.ImageBridge",
      .param_type = &holoscan::param_type_descriptor<ImageSize>(),
      .param_schema = holoscan::param_schema_descriptor<ImageSize>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &T::add_to_graph,
      .add_to_graph_from_assignments = &T::from_assignments,
      .set_params = &T::set_params,
      .port = &T::port,
      .ports = &T::ports,
      .region_add = &T::region_add,
      .partition_add = &T::partition_add,
      .lifecycle_before = &T::lifecycle_before,
      .lifecycle_allow_self_restart = &T::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &T::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &T::lifecycle_allow_restart_request,
  });
  if (!result) {
    throw holoscan::RuntimeError{result.error()};
  }
  using F = SizeThunks<viz::FmcwFrameComposerOp>;
  auto composer_result = authoring::authoring_registry().register_operator({
      .name = "holoscan.radar._app.FrameComposer",
      .param_type = &holoscan::param_type_descriptor<ImageSize>(),
      .param_schema = holoscan::param_schema_descriptor<ImageSize>(),
      .parameter_introspection = authoring::OperatorParameterIntrospection::kConstructorReplay,
      .add_to_graph = &F::add_to_graph,
      .add_to_graph_from_assignments = &F::from_assignments,
      .set_params = &F::set_params,
      .port = &F::port,
      .ports = &F::ports,
      .region_add = &F::region_add,
      .partition_add = &F::partition_add,
      .lifecycle_before = &F::lifecycle_before,
      .lifecycle_allow_self_restart = &F::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &F::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &F::lifecycle_allow_restart_request,
  });
  if (!composer_result) {
    throw holoscan::RuntimeError{composer_result.error()};
  }
  register_viewer<viz::FmcwHolovizOp>("holoscan.radar._app.DashboardViewer");
#if HOLOSCAN_RADAR_HAS_X11
  register_viewer<viz::X11ImagePresenterOp>("holoscan.radar._app.X11DashboardViewer");
#endif
}
