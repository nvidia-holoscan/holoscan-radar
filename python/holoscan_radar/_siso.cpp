// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/graph.hpp>
#include "applications/fmcw/siso/cpp/config.hpp"
#include "applications/fmcw/simo/cpp/config.hpp"
#include "applications/fmcw/dense_beamforming/cpp/config.hpp"
#include "radar_app/app/source.hpp"
#include "radar_app/report/report_ops.hpp"
#include "python/holoscan_radar/viz_bindings.hpp"

namespace nb = nanobind;
namespace radar = holoscan::radar;
namespace authoring = holoscan::authoring;
namespace app = holoscan::radar::app;

namespace {

struct AppConfigPath {
  std::string path;
};
HOLOSCAN_PARAM_SCHEMA(AppConfigPath,
                      holoscan::param_schema("com.nvidia.holoscan.radar.AppConfigPath", 1),
                      holoscan::field<&AppConfigPath::path>("path"));

template <typename Op>
struct AppThunks : authoring::ManualOperatorRegistrationThunks<Op> {
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return AppThunks::invoke_add([&] {
      return AppThunks::add_to_graph(graph, name, &*params);
    });
  }
  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::ErrorCode::kNotSupported,
                                 "app node configuration is constructor-only"};
  }
};

struct SisoSourceThunks : AppThunks<app::SimAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    radar::MaxTensorExtents extents;
    app::SimAdcSourceConfig source;
    if (input->path.empty()) {
      extents.max_chirps = 2;
      extents.max_samples_per_chirp = 2;
    } else {
      auto cfg = app::siso::load_config(std::filesystem::path(input->path));
      extents = cfg.extents;
      source.scene = std::move(cfg.scene);
      source.scene_visibility = std::move(cfg.scene_visibility);
      source.frame_period = cfg.frame_period;
      source.num_frames = cfg.num_frames;
    }
    return SisoSourceThunks::make_operator_ref(
        graph.op<app::SimAdcSourceOp>(name, extents, std::move(source)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return SisoSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct SisoReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    if (!input->path.empty()) {
      (void)app::siso::load_config(std::filesystem::path(input->path));
    }
    app::DetectionReportOp::Config config;
    config.mode = app::DetectionReportMode::kClusters;
    config.use_reduced_points = true;
    return SisoReportThunks::make_operator_ref(
        graph.op<app::DetectionReportOp>(name, std::move(config)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return SisoReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct SimoSourceThunks : AppThunks<app::SimAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::simo::load_config(std::filesystem::path(input->path));
    app::SimAdcSourceConfig source;
    source.scene = std::move(cfg.scene);
    source.scene_visibility = std::move(cfg.scene_visibility);
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    return SimoSourceThunks::make_operator_ref(
        graph.op<app::SimAdcSourceOp>(name, cfg.extents, std::move(source)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return SimoSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct SimoReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::simo::load_config(std::filesystem::path(input->path));
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kClusters;
    report.point_index_columns = app::simo::report_index_columns(*cfg.assembly_params);
    report.angle_convention = cfg.beamscan_params->angle_convention;
    return SimoReportThunks::make_operator_ref(
        graph.op<app::DetectionReportOp>(name, std::move(report)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return SimoReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct DenseSourceThunks : AppThunks<app::SimAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::dense_beamforming::load_config(std::filesystem::path(input->path));
    app::SimAdcSourceConfig source;
    source.scene = std::move(cfg.scene);
    source.scene_visibility = std::move(cfg.scene_visibility);
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    return DenseSourceThunks::make_operator_ref(
        graph.op<app::SimAdcSourceOp>(name, cfg.extents, std::move(source)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return DenseSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct DenseReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::dense_beamforming::load_config(std::filesystem::path(input->path));
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kRawDetections;
    report.angle_convention = cfg.beamformer_params->angle_convention;
    return DenseReportThunks::make_operator_ref(
        graph.op<app::DetectionReportOp>(name, std::move(report)));
  }
  static holoscan::expected<authoring::OpaqueOperatorRef, holoscan::Error> from_assignments(
      holoscan::Graph& graph, std::string_view name,
      std::span<const authoring::ParameterAssignment> assignments) {
    auto params = authoring::materialize_parameters(
        holoscan::param_type_descriptor<AppConfigPath>(),
        holoscan::param_schema_descriptor<AppConfigPath>(), assignments);
    if (!params) {
      return holoscan::make_unexpected(std::move(params).error());
    }
    return DenseReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

template <typename Thunks>
void register_app_node(std::string_view name) {
  auto result = authoring::authoring_registry().register_operator({
      .name = name,
      .param_type = &holoscan::param_type_descriptor<AppConfigPath>(),
      .param_schema = holoscan::param_schema_descriptor<AppConfigPath>(),
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

template <typename Params>
nb::object encode_params(const Params& params) {
  auto value = holoscan::config::encode_config(params);
  auto document = holoscan::config::Document::from_value(std::move(value));
  auto json = holoscan::config_io::serialize(document, holoscan::config_io::Format::kJson);
  return nb::module_::import_("json").attr("loads")(json);
}

nb::dict load_siso(std::string path) {
  auto cfg = app::siso::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_clusters"] = *cfg.extents.max_clusters;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["feature_assembly"] = encode_params(*cfg.assembly_params);
  result["cluster_dbscan"] = encode_params(*cfg.dbscan_params);
  result["detection_reduction"] = encode_params(*cfg.reduction_params);
  result["num_frames"] = cfg.num_frames;
  result["visualize"] = cfg.visualize;
  result["visualization"] =
      radar::python_viz::visualization_settings(cfg.visualization, cfg.frame_period);
  return result;
}

nb::dict load_simo(std::string path) {
  auto cfg = app::simo::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_snapshots"] = *cfg.extents.max_snapshots;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  extents["max_clusters"] = *cfg.extents.max_clusters;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["snapshot_gather"] = encode_params(*cfg.gather_params);
  result["beamscan_estimator"] = encode_params(*cfg.beamscan_params);
  result["feature_assembly"] = encode_params(*cfg.assembly_params);
  result["cluster_dbscan"] = encode_params(*cfg.dbscan_params);
  result["detection_reduction"] = encode_params(*cfg.reduction_params);
  result["num_frames"] = cfg.num_frames;
  result["visualize"] = cfg.visualize;
  result["visualization"] =
      radar::python_viz::visualization_settings(cfg.visualization, cfg.frame_period);
  return result;
}

nb::dict load_dense(std::string path) {
  auto cfg = app::dense_beamforming::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  nb::dict cfar_extents = nb::dict(extents);
  cfar_extents["max_tx"] = *cfg.cfar_extents.max_tx;
  cfar_extents["max_rx"] = *cfg.cfar_extents.max_rx;
  cfar_extents.attr("pop")("max_channels");
  nb::dict result;
  result["extents"] = std::move(extents);
  result["cfar_extents"] = std::move(cfar_extents);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["phase_shift_beamformer"] = encode_params(*cfg.beamformer_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["feature_assembly"] = encode_params(*cfg.assembly_params);
  result["num_frames"] = cfg.num_frames;
  result["visualize"] = cfg.visualize;
  result["visualization"] =
      radar::python_viz::visualization_settings(cfg.visualization, cfg.frame_period);
  return result;
}

struct SisoDashboardLoader {
  static auto load(const std::filesystem::path& path) { return app::siso::load_config(path); }
  static app::viz::FmcwDashboardConfig dashboard(const app::siso::Config& cfg) {
    return radar::python_viz::dashboard_config(cfg, "SISO", true, false, {}, {},
                                               radar::AngleConvention{});
  }
};

struct SimoDashboardLoader {
  static auto load(const std::filesystem::path& path) { return app::simo::load_config(path); }
  static app::viz::FmcwDashboardConfig dashboard(const app::simo::Config& cfg) {
    return radar::python_viz::dashboard_config(
        cfg, "SIMO", true, true, cfg.beamscan_params->az_coordinates,
        cfg.beamscan_params->el_coordinates, cfg.beamscan_params->angle_convention);
  }
};

struct DenseDashboardLoader {
  static auto load(const std::filesystem::path& path) {
    return app::dense_beamforming::load_config(path);
  }
  static app::viz::FmcwDashboardConfig dashboard(const app::dense_beamforming::Config& cfg) {
    return radar::python_viz::dashboard_config(
        cfg, "SIMO Dense Beamforming", false, true, cfg.beamformer_params->az_coordinates,
        cfg.beamformer_params->el_coordinates, cfg.beamformer_params->angle_convention);
  }
};

}  // namespace

void register_siso_bindings(nb::module_& module) {
  register_app_node<SisoSourceThunks>("holoscan.radar._app.SisoSource");
  register_app_node<SisoReportThunks>("holoscan.radar._app.SisoReport");
  module.def("_load_siso", &load_siso);
  module.def("_reset_app_frames", [](int frames) {
    app::AppFrameBank::reset(frames);
  });
  module.def("_app_frames_captured", [] {
    const std::lock_guard lock(app::AppFrameBank::mutex);
    return app::AppFrameBank::captured;
  });
  register_app_node<SimoSourceThunks>("holoscan.radar._app.SimoSource");
  register_app_node<SimoReportThunks>("holoscan.radar._app.SimoReport");
  module.def("_load_simo", &load_simo);
  register_app_node<DenseSourceThunks>("holoscan.radar._app.DenseSource");
  register_app_node<DenseReportThunks>("holoscan.radar._app.DenseReport");
  module.def("_load_dense", &load_dense);
  radar::python_viz::register_dashboard<SisoDashboardLoader>("holoscan.radar._app.SisoDashboard");
  radar::python_viz::register_dashboard<SimoDashboardLoader>("holoscan.radar._app.SimoDashboard");
  radar::python_viz::register_dashboard<DenseDashboardLoader>("holoscan.radar._app.DenseDashboard");
}
