// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <mutex>
#include <unordered_map>
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
#include "radar_common/sdk/config_codecs.hpp"
#include "applications/fmcw/mimo_tdm/cpp/config.hpp"
#include "applications/fmcw/adaptive_aoa/cpp/config.hpp"
#include "applications/fmcw/multi_waveform/cpp/config.hpp"
#include "radar_app/fmcw/mimo_sources.hpp"
#include "radar_app/report/report_ops.hpp"
#include "python/holoscan_radar/viz_bindings.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"

namespace holoscan::config {
template <>
struct ConfigCodec<radar::DopplerNotchParams>
: params::DefaultedParamSchemaCodec<radar::DopplerNotchParams> {};
template <>
struct ConfigCodec<radar::RankOneResidualScreenParams>
: params::DefaultedParamSchemaCodec<radar::RankOneResidualScreenParams> {};
template <>
struct ConfigCodec<radar::MUSICParams> : params::DefaultedParamSchemaCodec<radar::MUSICParams> {};
template <>
struct ConfigCodec<radar::RootMUSICParams>
: params::DefaultedParamSchemaCodec<radar::RootMUSICParams> {};
template <>
struct ConfigCodec<radar::ESPRITParams> : params::DefaultedParamSchemaCodec<radar::ESPRITParams> {};
template <>
struct ConfigCodec<radar::MVDRParams> : params::DefaultedParamSchemaCodec<radar::MVDRParams> {};
}  // namespace holoscan::config

namespace nb = nanobind;
namespace radar = holoscan::radar;
namespace authoring = holoscan::authoring;
namespace app = holoscan::radar::app;
namespace mimo = holoscan::radar::app::mimo;

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

struct TdmSourceThunks : AppThunks<mimo::MimoTdmAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::mimo_tdm::load_config(std::filesystem::path(input->path));
    mimo::MimoTdmAdcSourceConfig source;
    source.scene = std::move(cfg.scene);
    source.scene_visibility = std::move(cfg.scene_visibility);
    source.tx_schedule = std::move(cfg.tx_schedule);
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    return TdmSourceThunks::make_operator_ref(
        graph.op<mimo::MimoTdmAdcSourceOp>(name, cfg.extents, std::move(source)));
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
    return TdmSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct TdmReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::mimo_tdm::load_config(std::filesystem::path(input->path));
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kClusters;
    report.point_index_columns = std::move(cfg.report_index_columns);
    report.use_reduced_points = true;
    report.angle_convention = cfg.beamscan_params->angle_convention;
    return TdmReportThunks::make_operator_ref(
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
    return TdmReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct DdmSourceThunks : AppThunks<mimo::MimoDdmAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = mimo::load_ddm_config(std::filesystem::path(input->path));
    mimo::MimoDdmAdcSourceConfig source;
    source.scene = std::move(cfg.scene);
    source.scene_visibility = std::move(cfg.scene_visibility);
    source.doppler_offsets = std::move(cfg.doppler_offsets);
    if (cfg.correlate) {
      source.phase_codes = cfg.ddm_code.phase_codes;
    }
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    return DdmSourceThunks::make_operator_ref(
        graph.op<mimo::MimoDdmAdcSourceOp>(name, cfg.extents, std::move(source)));
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
    return DdmSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct DdmReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = mimo::load_ddm_config(std::filesystem::path(input->path));
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kClusters;
    report.point_index_columns = std::move(cfg.report_index_columns);
    report.use_reduced_points = true;
    report.angle_convention = cfg.beamscan_params->angle_convention;
    return DdmReportThunks::make_operator_ref(
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
    return DdmReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct AdaptiveSourceThunks : AppThunks<mimo::MimoTdmAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::adaptive_aoa::load_config(std::filesystem::path(input->path));
    mimo::MimoTdmAdcSourceConfig source;
    source.scene = std::move(cfg.scene);
    source.scene_visibility = std::move(cfg.scene_visibility);
    source.tx_schedule = std::move(cfg.tx_schedule);
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    return AdaptiveSourceThunks::make_operator_ref(
        graph.op<mimo::MimoTdmAdcSourceOp>(name, cfg.extents, std::move(source)));
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
    return AdaptiveSourceThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct AdaptiveReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto cfg = app::adaptive_aoa::load_config(std::filesystem::path(input->path));
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kRawDetections;
    report.angle_convention = cfg.beamscan_params->angle_convention;
    return AdaptiveReportThunks::make_operator_ref(
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
    return AdaptiveReportThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct MultiContext {
  explicit MultiContext(const std::string& path)
  : cfg(app::multi_waveform::load_config(std::filesystem::path(path))) {
    coordinator = std::make_shared<app::TransmitFrameCoordinator>(cfg.max_frames_in_flight);
    link = std::make_shared<app::SimulatedTdmLink>(
        app::SimulatedTdmLinkConfig{.profiles = cfg.profiles,
                                    .scene = cfg.scene,
                                    .scene_visibility = cfg.scene_visibility,
                                    .tx_schedule = cfg.tx_schedule,
                                    .frame_period = cfg.frame_period,
                                    .polls_before_ready = cfg.simulated_receive_polls});
  }
  app::multi_waveform::Config cfg;
  std::shared_ptr<app::TransmitFrameCoordinator> coordinator;
  std::shared_ptr<app::SimulatedTdmLink> link;
};

std::shared_ptr<MultiContext> multi_context(const std::string& path) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<MultiContext>> contexts;
  std::lock_guard<std::mutex> lock(mutex);
  auto& context = contexts[path];
  if (!context) {
    context = std::make_shared<MultiContext>(path);
  }
  return context;
}

struct MultiTxThunks : AppThunks<app::FMCWTransmitSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto context = multi_context(input->path);
    const auto& cfg = context->cfg;
    app::FMCWTransmitSourceConfig source;
    source.profiles = cfg.profiles;
    source.profile_sequence = cfg.profile_sequence;
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    source.driver = context->link;
    source.coordinator = context->coordinator;
    return MultiTxThunks::make_operator_ref(
        graph.op<app::FMCWTransmitSourceOp>(name, std::move(source)));
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
    return MultiTxThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct MultiRxThunks : AppThunks<app::CorrelatedMimoTdmAdcSourceOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto context = multi_context(input->path);
    const auto& cfg = context->cfg;
    app::CorrelatedMimoTdmAdcSourceConfig source;
    source.poll_period = cfg.rx_poll_period;
    source.frame_period = cfg.frame_period;
    source.num_frames = cfg.num_frames;
    source.pending_capacity = cfg.max_frames_in_flight;
    source.driver = context->link;
    source.coordinator = context->coordinator;
    return MultiRxThunks::make_operator_ref(
        graph.op<app::CorrelatedMimoTdmAdcSourceOp>(name, cfg.extents, std::move(source)));
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
    return MultiRxThunks::invoke_add([&] {
      return add_to_graph(graph, name, &*params);
    });
  }
};

struct MultiReportThunks : AppThunks<app::DetectionReportOp> {
  static authoring::OpaqueOperatorRef add_to_graph(
      holoscan::Graph& graph, std::string_view name,
      const authoring::MaterializedParameters* params) {
    const auto* input = static_cast<const AppConfigPath*>(params->data());
    auto context = multi_context(input->path);
    const auto& cfg = context->cfg;
    app::DetectionReportOp::Config report;
    report.mode = app::DetectionReportMode::kClusters;
    report.point_index_columns = cfg.report_index_columns;
    report.use_reduced_points = true;
    report.angle_convention = cfg.beamscan_params->angle_convention;
    return MultiReportThunks::make_operator_ref(
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
    return MultiReportThunks::invoke_add([&] {
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

nb::dict load_tdm(std::string path) {
  auto cfg = app::mimo_tdm::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_tx"] = *cfg.extents.max_tx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_interleaved_chirps"] = *cfg.extents.max_interleaved_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_virtual_channels"] = *cfg.extents.max_virtual_channels;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_snapshots"] = *cfg.extents.max_snapshots;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  extents["max_clusters"] = *cfg.extents.max_clusters;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["mimo_deinterleave"] = encode_params(*cfg.deinterleave_params);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["tdm_sparse_compensation"] = encode_params(*cfg.compensation_params);
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

nb::dict load_ddm(std::string path) {
  auto cfg = mimo::load_ddm_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_tx"] = *cfg.extents.max_tx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_virtual_channels"] = *cfg.extents.max_virtual_channels;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_snapshots"] = *cfg.extents.max_snapshots;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  extents["max_clusters"] = *cfg.extents.max_clusters;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["ddm_dense_demultiplex"] = encode_params(*cfg.demultiplex_params);
  result["doppler_notch"] = encode_params(*cfg.notch_params);
  result["virtual_array_assembly"] = encode_params(*cfg.vaa_params);
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

nb::dict load_adaptive(std::string path) {
  auto cfg = app::adaptive_aoa::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_tx"] = *cfg.extents.max_tx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_interleaved_chirps"] = *cfg.extents.max_interleaved_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_virtual_channels"] = *cfg.extents.max_virtual_channels;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_snapshots"] = *cfg.extents.max_snapshots;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  extents["max_sources"] = *cfg.extents.max_sources;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["mimo_deinterleave"] = encode_params(*cfg.deinterleave_params);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["tdm_sparse_compensation"] = encode_params(*cfg.compensation_params);
  result["beamscan_estimator"] = encode_params(*cfg.beamscan_params);
  result["rank_one_residual_screen"] = encode_params(*cfg.screen_params);
  result["feature_assembly"] = encode_params(*cfg.assembly_params);
  result["num_frames"] = cfg.num_frames;
  result["visualize"] = cfg.visualize;
  result["visualization"] =
      radar::python_viz::visualization_settings(cfg.visualization, cfg.frame_period);
  switch (cfg.algorithm) {
    case app::adaptive_aoa::Algorithm::kMusic:
      result["estimator_name"] = "music_estimator";
      result["estimator_params"] = encode_params(**cfg.music_params);
      break;
    case app::adaptive_aoa::Algorithm::kRootMusic:
      result["estimator_name"] = "root_music_estimator";
      result["estimator_params"] = encode_params(**cfg.root_music_params);
      break;
    case app::adaptive_aoa::Algorithm::kEsprit:
      result["estimator_name"] = "esprit_estimator";
      result["estimator_params"] = encode_params(**cfg.esprit_params);
      break;
    case app::adaptive_aoa::Algorithm::kMvdr:
      result["estimator_name"] = "mvdr_estimator";
      result["estimator_params"] = encode_params(**cfg.mvdr_params);
      break;
  }
  return result;
}

nb::dict load_multi(std::string path) {
  auto cfg = app::multi_waveform::load_config(std::filesystem::path(path));
  nb::dict extents;
  extents["max_rx"] = *cfg.extents.max_rx;
  extents["max_tx"] = *cfg.extents.max_tx;
  extents["max_chirps"] = *cfg.extents.max_chirps;
  extents["max_interleaved_chirps"] = *cfg.extents.max_interleaved_chirps;
  extents["max_samples_per_chirp"] = *cfg.extents.max_samples_per_chirp;
  extents["max_range_bins"] = *cfg.extents.max_range_bins;
  extents["max_doppler_bins"] = *cfg.extents.max_doppler_bins;
  extents["max_virtual_channels"] = *cfg.extents.max_virtual_channels;
  extents["max_channels"] = *cfg.extents.max_channels;
  extents["max_snapshots"] = *cfg.extents.max_snapshots;
  extents["max_detections"] = *cfg.extents.max_detections;
  extents["max_items"] = *cfg.extents.max_items;
  extents["max_points"] = *cfg.extents.max_points;
  extents["max_features"] = *cfg.extents.max_features;
  extents["max_az"] = *cfg.extents.max_az;
  extents["max_el"] = *cfg.extents.max_el;
  extents["max_clusters"] = *cfg.extents.max_clusters;
  nb::dict result;
  result["extents"] = std::move(extents);
  result["mimo_deinterleave"] = encode_params(*cfg.deinterleave_params);
  result["range_processor"] = encode_params(*cfg.range_params);
  result["clutter_filter"] = encode_params(*cfg.clutter_params);
  result["doppler_processor"] = encode_params(*cfg.doppler_params);
  result["cfar_detector"] = encode_params(*cfg.cfar_params);
  result["tdm_sparse_compensation"] = encode_params(*cfg.compensation_params);
  result["beamscan_estimator"] = encode_params(*cfg.beamscan_params);
  result["feature_assembly"] = encode_params(*cfg.assembly_params);
  result["cluster_dbscan"] = encode_params(*cfg.dbscan_params);
  result["detection_reduction"] = encode_params(*cfg.reduction_params);
  result["num_frames"] = cfg.num_frames;
  result["visualize"] = cfg.visualize;
  result["visualization"] =
      radar::python_viz::visualization_settings(cfg.visualization, cfg.frame_period);
  result["max_frames_in_flight"] = cfg.max_frames_in_flight;
  result["visualization_profile_max_gap_frames"] = cfg.visualization_profile_max_gap_frames;
  result["initial_waveform"] =
      encode_params(cfg.profiles.at(cfg.profile_sequence.front()).waveform);
  return result;
}

struct TdmDashboardLoader {
  static auto load(const std::filesystem::path& path) { return app::mimo_tdm::load_config(path); }
  static app::viz::FmcwDashboardConfig dashboard(const app::mimo_tdm::Config& cfg) {
    return radar::python_viz::dashboard_config(
        cfg, "MIMO TDM", true, true, cfg.beamscan_params->az_coordinates,
        cfg.beamscan_params->el_coordinates, cfg.beamscan_params->angle_convention);
  }
};

struct DdmDashboardLoader {
  static auto load(const std::filesystem::path& path) { return mimo::load_ddm_config(path); }
  static app::viz::FmcwDashboardConfig dashboard(const mimo::DdmConfig& cfg) {
    return radar::python_viz::dashboard_config(
        cfg, "MIMO DDM", true, true, cfg.beamscan_params->az_coordinates,
        cfg.beamscan_params->el_coordinates, cfg.beamscan_params->angle_convention);
  }
};

struct AdaptiveDashboardLoader {
  static auto load(const std::filesystem::path& path) {
    return app::adaptive_aoa::load_config(path);
  }
  static app::viz::FmcwDashboardConfig dashboard(const app::adaptive_aoa::Config& cfg) {
    return radar::python_viz::dashboard_config(
        cfg, std::string("Adaptive AoA | ") + app::adaptive_aoa::algorithm_name(cfg.algorithm),
        false, true, cfg.beamscan_params->az_coordinates, cfg.beamscan_params->el_coordinates,
        cfg.beamscan_params->angle_convention);
  }
};

struct MultiDashboardLoader {
  static auto load(const std::filesystem::path& path) {
    return app::multi_waveform::load_config(path);
  }
  static app::viz::FmcwDashboardConfig dashboard(const app::multi_waveform::Config& cfg) {
    auto result = radar::python_viz::dashboard_config(
        cfg, "Multi-Waveform | PROFILE: " + cfg.visualization.profile, true, true,
        cfg.beamscan_params->az_coordinates, cfg.beamscan_params->el_coordinates,
        cfg.beamscan_params->angle_convention);
    result.selected_profile = cfg.visualization.profile;
    result.selected_waveform = cfg.profiles.at(cfg.visualization_profile_index).waveform;
    return result;
  }
};

}  // namespace

void register_tdm_bindings(nb::module_& module) {
  register_app_node<TdmSourceThunks>("holoscan.radar._app.TdmSource");
  register_app_node<TdmReportThunks>("holoscan.radar._app.TdmReport");
  module.def("_load_tdm", &load_tdm);
  register_app_node<DdmSourceThunks>("holoscan.radar._app.DdmSource");
  register_app_node<DdmReportThunks>("holoscan.radar._app.DdmReport");
  module.def("_load_ddm", &load_ddm);
  register_app_node<AdaptiveSourceThunks>("holoscan.radar._app.AdaptiveSource");
  register_app_node<AdaptiveReportThunks>("holoscan.radar._app.AdaptiveReport");
  module.def("_load_adaptive", &load_adaptive);
  register_app_node<MultiTxThunks>("holoscan.radar._app.MultiTx");
  register_app_node<MultiRxThunks>("holoscan.radar._app.MultiRx");
  register_app_node<MultiReportThunks>("holoscan.radar._app.MultiReport");
  module.def("_load_multi", &load_multi);
  radar::python_viz::register_dashboard<TdmDashboardLoader>("holoscan.radar._app.TdmDashboard");
  radar::python_viz::register_dashboard<DdmDashboardLoader>("holoscan.radar._app.DdmDashboard");
  radar::python_viz::register_dashboard<AdaptiveDashboardLoader>(
      "holoscan.radar._app.AdaptiveDashboard");
  radar::python_viz::register_dashboard<MultiDashboardLoader>("holoscan.radar._app.MultiDashboard");
}
