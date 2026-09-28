// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
#include "coordinate_ops.hpp"  // NOLINT(build/include_subdir)
#include "config.hpp"          // NOLINT(build/include_subdir)
#include "custom_ops.hpp"      // NOLINT(build/include_subdir)
#include "mosaic_op.hpp"       // NOLINT(build/include_subdir)
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
#include <holoscan/visualization/holoviz_op.hpp>
#include "sar_image_viz.hpp"  // NOLINT(build/include_subdir)
#endif
#include "scroll_op.hpp"    // NOLINT(build/include_subdir)
#include "staging_ops.hpp"  // NOLINT(build/include_subdir)

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;

namespace {
void bind_outputs(holoscan::CompileOptions& options, const char* op,
                  std::initializer_list<const char*> names) {
  for (const char* port : names) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  }
}

radar::MaxTensorExtents make_extents(const AppConfig& cfg,
                                     const radar::ApertureManagerParams& manager) {
  int64_t max_samples = 0;
  for (const auto& path : cfg.cphd_paths) {
    radar::cphd::CphdFile file(path);
    max_samples = std::max(max_samples, file.channel(0).num_samples);
  }
  if (cfg.reader_pulses_per_tick < 1 || max_samples < 1)
    throw std::invalid_argument("sar_formation: invalid CPHD pulse/sample extents");
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
}  // namespace

int main(int argc, char** argv) {
  try {
    const CliPaths cli = parse_cli(argc, argv, "mode01.yaml");
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

    holoscan::Graph graph{"sar_formation"};
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
    if (cfg.collection_geometries.size() > 1U) {
      auto stager = graph.op<app::sar::CollectionBoundaryStagerOp>(
          "collection_stager", cfg.collection_geometries, cfg.collection_spacings,
          cfg.collection_frequencies);
      application.add(stager);
      graph.add_flow(range->range_profiles, stager->range_profiles, queue);
      graph.add_flow(reader->stream_boundary, stager->stream_boundary, queue);
      graph.add_flow(stager->range_profiles_out, manager->range_profiles, queue);
      graph.add_flow(stager->sample_geometry, manager->sample_geometry, queue);
      graph.add_flow(stager->range_bin_spacing, bp->range_bin_spacing, queue);
      graph.add_flow(stager->reference_frequency, bp->reference_frequency, queue);
      graph.add_flow(manager->sample_geometry_out, bp->sample_geometry, queue);
    } else if (cfg.tile_geometries.size() > 1U) {
      auto stager = graph.op<app::sar::TileGeometryStagerOp>("tile_stager", cfg.tile_geometries,
                                                             cfg.tile_pulses);
      application.add(stager);
      graph.add_flow(range->range_profiles, stager->range_profiles, queue);
      graph.add_flow(reader->stream_boundary, stager->stream_boundary, queue);
      graph.add_flow(stager->range_profiles_out, manager->range_profiles, queue);
      graph.add_flow(stager->sample_geometry, manager->sample_geometry, queue);
      graph.add_flow(manager->sample_geometry_out, bp->sample_geometry, queue);
    } else if (cfg.scroll) {
      const int64_t slice =
          cfg.emit_interval_pulses > 0 && cfg.reader_pulses_per_tick > cfg.emit_interval_pulses
              ? cfg.emit_interval_pulses
              : cfg.reader_pulses_per_tick;
      auto scroll = graph.op<app::sar::ScrollGeometryStagerOp>(
          "scroll_stager", cfg.scroll_parent, cfg.sample_geometry, cfg.scroll_anchor,
          cfg.reader_loop, slice, cfg.reader_pulses_per_tick, *extents.max_range_bins);
      application.add(scroll);
      bind_outputs(options, "scroll_stager",
                   {"range_profiles_out", "positions_out", "reference_position_out"});
      graph.add_flow(range->range_profiles, scroll->range_profiles, queue);
      graph.add_flow(reader->positions, scroll->positions, queue);
      graph.add_flow(reader->reference_position, scroll->reference_position, queue);
      graph.add_flow(reader->stream_boundary, scroll->stream_boundary, queue);
      graph.add_flow(scroll->range_profiles_out, manager->range_profiles, queue);
      graph.add_flow(scroll->positions_out, manager->positions, queue);
      graph.add_flow(scroll->reference_position_out, manager->reference_position, queue);
      graph.add_flow(scroll->stream_boundary_out, manager->stream_boundary, queue);
      graph.add_flow(scroll->sample_geometry, bp->sample_geometry, queue);
    } else {
      graph.add_flow(range->range_profiles, manager->range_profiles, queue);
    }
    if (cfg.scroll) {
      // The scroll stage owns positions, reference positions and boundary wiring.
    } else if (cfg.transform_to_scene_enu) {
      auto enu = graph.op<app::sar::EcefToSceneEnuOp>("ecef_to_enu", cfg.enu_origin_ecef,
                                                      cfg.quantize_phase_center_f32,
                                                      cfg.reader_pulses_per_tick);
      application.add(enu);
      bind_outputs(options, "ecef_to_enu", {"positions_out", "reference_position_out"});
      graph.add_flow(reader->positions, enu->positions, queue);
      graph.add_flow(reader->reference_position, enu->reference_position, queue);
      graph.add_flow(enu->positions_out, manager->positions, queue);
      graph.add_flow(enu->reference_position_out, manager->reference_position, queue);
    } else {
      graph.add_flow(reader->positions, manager->positions, queue);
      graph.add_flow(reader->reference_position, manager->reference_position, queue);
    }
    if (!cfg.scroll)
      graph.add_flow(reader->stream_boundary, manager->stream_boundary, queue);
    graph.add_flow(manager->range_profiles_out, bp->range_profiles, queue);
    graph.add_flow(manager->positions_out, bp->positions, queue);
    graph.add_flow(manager->reference_position_out, bp->reference_position, queue);
    graph.add_flow(manager->control_out, bp->control, queue);
    if (cfg.wire_weight)
      graph.add_flow(manager->weight, bp->weight, queue);
    const holoscan::Output<holoscan::Tensor>* formed = &bp->image;
    if (cfg.parent_geometry.is_valid() || cfg.scroll) {
      auto mosaic = graph.op<app::sar::SarMosaicOp>(
          "sar_mosaic", cfg.scroll ? cfg.scroll_parent : cfg.parent_geometry, cfg.reader_loop);
      application.add(mosaic);
      bind_outputs(options, "sar_mosaic", {"image_out"});
      graph.add_flow(bp->image, mosaic->image, queue);
      graph.add_flow(bp->sample_geometry_out, mosaic->sample_geometry, queue);
      formed = &mosaic->image_out;
    }
    graph.add_flow(*formed, report->image, queue);
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
    bool visualize = cfg.visualize;
#ifdef HOLOSCAN_RADAR_HAS_X11
    if (visualize && !cfg.headless && !cfg.skip_display_check && !app::gui_available()) {
      HOLOSCAN_LOG_WARN("sar_formation: no X11 display; skipping visualization");
      visualize = false;
    }
#endif
    if (visualize) {
      const auto& viz_geometry = cfg.scroll                       ? cfg.scroll_parent
                                 : cfg.parent_geometry.is_valid() ? cfg.parent_geometry
                                                                  : cfg.sample_geometry;
      auto raster = graph.op<app::sar::SarImageVizOp>("sar_image_viz", viz_geometry.num_v(),
                                                      viz_geometry.num_u(), cfg.visualize_size);
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
    HOLOSCAN_LOG_ERROR("sar_formation: {}", error.what());
    return 1;
  }
}
