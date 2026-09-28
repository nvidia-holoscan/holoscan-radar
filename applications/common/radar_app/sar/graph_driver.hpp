// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/cphd_reader_op/params_config.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_app/app/config_utils.hpp"
#ifdef HOLOSCAN_RADAR_HAS_X11
#include "radar_app/viz/x11_display.hpp"
#endif
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "sar/aperture_manager_op/params_config.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/backprojection_op/params_config.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "sar/sar_range_compression_op/params_config.hpp"
#include "radar_app/sar/config.hpp"
#include "radar_app/sar/custom_ops.hpp"
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
#include <holoscan/visualization/holoviz_op.hpp>
#include "radar_app/sar/sar_image_viz.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;

inline void bind_outputs(holoscan::CompileOptions& options, const char* op,
                         std::initializer_list<const char*> names) {
  for (const char* port : names) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
}

inline radar::MaxTensorExtents make_extents(const AppConfig& cfg,
                                            const radar::ApertureManagerParams& manager) {
  int64_t max_samples = 0;
  for (const auto& path : cfg.cphd_paths) {
    radar::cphd::CphdFile file(path);
    max_samples = std::max(max_samples, file.channel(0).num_samples);
  }
  if (cfg.reader_pulses_per_tick < 1 || max_samples < 1)
    throw std::invalid_argument("SAR: invalid CPHD pulse/sample extents");
  radar::MaxTensorExtents extents;
  extents.max_samples_per_pulse = max_samples;
  extents.max_pulses = cfg.reader_pulses_per_tick + manager.max_aperture_pulses;
  extents.max_range_bins = cfg.nfft > 0 ? cfg.nfft : max_samples;
  int64_t max_v = cfg.sample_geometry.num_v();
  int64_t max_u = cfg.sample_geometry.num_u();
  for (const auto& geom : cfg.collection_geometries) {
    max_v = std::max(max_v, geom.num_v());
    max_u = std::max(max_u, geom.num_u());
  }
  for (const auto& geom : cfg.tile_geometries) {
    max_v = std::max(max_v, geom.num_v());
    max_u = std::max(max_u, geom.num_u());
  }
  extents.max_image_v = max_v;
  extents.max_image_u = max_u;
  return extents;
}
template <typename LoadConfig, typename BuildGraph>
int run_sar_graph_app(int argc, char** argv, const char* name, LoadConfig load_config,
                      BuildGraph build_graph) {
  try {
    const CliPaths cli = parse_cli(argc, argv, (std::string(name) + ".yaml").c_str());
    AppConfig cfg = load_config(cli.config, cli.cphd_override);
    auto document = app::config::load_document(cli.config);
    document = app::config::normalize_enum_field<radar::FftNorm>(std::move(document),
                                                                 "/range_compression/fft_norm");
    document = app::config::normalize_enum_field<radar::ApertureEmitMode>(
        std::move(document), "/aperture_manager/emit_mode");
    document = app::config::normalize_enum_field<radar::BackprojectionBackend>(
        std::move(document), "/backprojection/backend");
    document = app::config::normalize_enum_field<radar::GeometryChange>(
        std::move(document), "/backprojection/geometry_change");
    document = app::config::normalize_enum_field<radar::RangeInterpolation>(
        std::move(document), "/backprojection/interpolation");

    radar::CphdReaderParams reader_seed;
    reader_seed.path = radar::CollectionPaths(cfg.cphd_paths);
    auto reader_params = app::config::decode_params(document, "/cphd_reader", reader_seed);
    radar::SARRangeCompressionParams range_seed;
    range_seed.nfft = cfg.nfft;
    auto range_params = app::config::decode_params(document, "/range_compression", range_seed);
    radar::ApertureManagerParams manager_seed;
    manager_seed.emit_interval_pulses = cfg.tile_pulses;
    auto manager_params = app::config::decode_params(document, "/aperture_manager", manager_seed);
    radar::BackprojectionParams bp_seed;
    bp_seed.sample_geometry = cfg.sample_geometry;
    bp_seed.range_bin_spacing = cfg.range_bin_spacing;
    bp_seed.reference_frequency = cfg.reference_frequency;
    auto bp_params = app::config::decode_params(document, "/backprojection", bp_seed);
    const radar::MaxTensorExtents extents = make_extents(cfg, *manager_params);

    holoscan::Graph graph{name};
    holoscan::CompileOptions options;
    const holoscan::ConnectionOptions queue{.queue_depth = 4U};
    auto reader = graph.op<radar::CPHDReaderOp<double>>("cphd_reader", extents);
    reader.set_params(std::move(reader_params));
    bind_outputs(options, "cphd_reader",
                 {"phase_history", "positions", "reference_position", "sample_axis", "pulse_times",
                  "velocities"});
    auto range = graph.op<radar::SARRangeCompressionOp<double>>("range_compression", extents);
    range.set_params(std::move(range_params));
    bind_outputs(options, "range_compression", {"range_profiles", "range_bins"});
    auto manager = graph.op<radar::ApertureManagerOp<double>>("aperture_manager", extents);
    manager.set_params(std::move(manager_params));
    bind_outputs(options, "aperture_manager",
                 {"range_profiles_out", "positions_out", "reference_position_out", "weight",
                  "sample_points_out", "control_out"});
    auto bp = graph.op<radar::BackprojectionOp<double>>("backprojection", extents);
    bp.set_params(std::move(bp_params));
    bind_outputs(options, "backprojection", {"image"});
    auto counter = std::make_shared<app::sar::ImageCount>();
    auto report = graph.op<app::sar::SarImageReportOp>("image_report", counter);
    auto application = graph.partition("application");
    application.add(reader).add(range).add(manager).add(bp).add(report);
    graph.add_flow(reader->phase_history, range->phase_history, queue);
    const auto* formed =
        build_graph(graph, options, queue, cfg, reader, range, manager, bp, application, extents);
    graph.add_flow(*formed, report->image, queue);
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
    bool visualize = cfg.visualize;
#ifdef HOLOSCAN_RADAR_HAS_X11
    if (visualize && !cfg.headless && !cfg.skip_display_check && !app::gui_available()) {
      HOLOSCAN_LOG_WARN("SAR: no X11 display; skipping visualization");
      visualize = false;
    }
#endif
    if (visualize) {
      const auto& viz_geometry = cfg.scroll                       ? cfg.scroll_parent
                                 : cfg.parent_geometry.is_valid() ? cfg.parent_geometry
                                                                  : cfg.sample_geometry;
      auto raster = graph.op<app::sar::SarImageVizOp>("sar_image_viz", viz_geometry.num_v(),
                                                      viz_geometry.num_u(), cfg.visualize_size,
                                                      cfg.headless ? 0.0 : cfg.visualize_hold_s);
      application.add(raster);
      bind_outputs(options, "sar_image_viz", {"frame"});
      holoscan::visualization::LayerStack layers;
      layers.add_image_layer("sar_image",
                             holoscan::visualization::ImageLayerDesc{
                                 .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm,
                                 .max_width = static_cast<uint32_t>(cfg.visualize_size),
                                 .max_height = static_cast<uint32_t>(cfg.visualize_size)});
      const int64_t max_side = std::max(viz_geometry.num_v(), viz_geometry.num_u());
      const auto width = static_cast<uint32_t>(
          std::max<int64_t>(1, cfg.visualize_size * viz_geometry.num_u() / max_side));
      const auto height = static_cast<uint32_t>(
          std::max<int64_t>(1, cfg.visualize_size * viz_geometry.num_v() / max_side));
      if (cfg.headless) {
        auto viz = graph.op<holoscan::visualization::HolovizOp>(
            "holoviz", std::move(layers),
            holoscan::visualization::CudaSinkDesc{
                .width = width,
                .height = height,
                .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm});
        auto sink = graph.op<app::sar::SarImageSchemaSinkOp>("viz_sink");
        application.add(viz).add(sink);
        bind_outputs(options, "holoviz", {"output"});
        graph.add_flow(*formed, raster->image, queue);
        graph.add_flow(raster->frame, viz->image_layer_input(0), queue);
        graph.add_flow(viz->output(), sink->frame, queue);
      } else {
        auto viz = graph.op<holoscan::visualization::HolovizOp>(
            "holoviz", std::move(layers),
            holoscan::visualization::WindowSinkDesc{
                .title = "SAR formation", .width = width, .height = height});
        graph.partition("visualization").add(viz);
        graph.add_flow(*formed, raster->image, queue);
        graph.add_flow(raster->frame, viz->image_layer_input(0), queue);
      }
    }
#else
    if (cfg.visualize)
      HOLOSCAN_LOG_WARN("SAR visualization component is unavailable");
#endif
    if (cfg.dump) {
      auto dump = graph.op<radar::TensorDumpOp>("tensor_dump");
      radar::TensorDumpParams params;
      params.path = cfg.dump_path;
      if (cfg.dump_max_files >= 0)
        params.max_files = cfg.dump_max_files;
      dump.set_params(params);
      application.add(dump);
      graph.add_flow(*formed, dump->tensor, queue);
    }
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    const auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok())
      throw std::runtime_error(std::string("SAR graph compile failed: ") +
                               std::string(plan.json()));
    auto session = holoscan::run_async(plan);
    session.wait();
    if (!session.diagnostics().empty()) {
      std::string error = "SAR graph run diagnostics:";
      for (const auto& diagnostic : session.diagnostics())
        error += " [" + holoscan::format_diagnostic(diagnostic, plan.graph()) + "]";
      throw std::runtime_error(error);
    }
    if (counter->count.load() == 0)
      throw std::runtime_error("SAR graph ended without an image");
    return 0;
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("{}: {}", name, error.what());
    return 1;
  }
}
