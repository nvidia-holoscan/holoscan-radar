// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/cphd_reader_op/params_config.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "io/tensor_dump_op/output_paths.hpp"
#include "radar_app/app/config_utils.hpp"
#include "radar_app/app/config_loader.hpp"
#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/app/shutdown_signals.hpp"
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
#include "radar_app/sar/diagnostics.hpp"
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
#include <holoscan/visualization/holoviz_op.hpp>
#include "radar_app/sar/sar_image_viz.hpp"
#include "radar_app/viz/image_device_bridge_op.hpp"
#include "radar_app/viz/frame_composer_op.hpp"
#include "radar_app/viz/presentation_graph.hpp"
#include "radar_app/sar/timeline_raster.hpp"
#include "radar_app/sar/window_place.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;

inline void bind_outputs(holoscan::CompileOptions& options, const char* op,
                         std::initializer_list<const char*> names,
                         holoscan::DeviceId device = app::application_device()) {
  for (const char* port : names) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = device});
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
  // Reform sends no pulses between boundaries (OPERATOR_API.md step 5a), so
  // the manager's largest output is the replay alone -- never a tick's
  // admissions plus the evictions they displace, which is what the sum below
  // bounds. The reader's batch still has to fit, since ingest arrives on the
  // same pools. On Kourou that distinction is 6.3 GiB of reservation.
  if (manager.emit_mode == radar::ApertureEmitMode::kReform) {
    extents.max_pulses = std::max(cfg.reader_pulses_per_tick, manager.max_aperture_pulses);
  } else {
    extents.max_pulses = cfg.reader_pulses_per_tick + manager.max_aperture_pulses;
  }
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

// The aperture-wide `max_pulses` above bounds what the manager can replay or
// evict, and every operator downstream of it has to be sized for that. The
// ingest side never does: the reader emits one `pulses_per_tick` batch and
// range compression is 1:1 on it. Sizing those pools for the retained aperture
// costs nothing under a per-tick allocator but is real memory once HSDK 5
// materializes pools up front — on a 23k-pulse by 24k-sample collect it is
// 17.45 GiB per pool against 0.73 GiB needed. Apps with their own ingest-edge
// operators should size them from this, not from the aperture extents.
inline radar::MaxTensorExtents make_ingest_extents(const radar::MaxTensorExtents& extents,
                                                   const AppConfig& cfg) {
  radar::MaxTensorExtents ingest = extents;
  ingest.max_pulses = cfg.reader_pulses_per_tick;
  return ingest;
}

// Handed to the graph-building callback so an app can name the precisions the
// driver instantiated. Apps that form at one precision ignore it.
template <typename TSample, typename TGeometry>
struct Precision {
  using Sample = TSample;
  using Geometry = TGeometry;
};

// Keep prepared contexts independent of later thread-local device discovery.
struct SarDeviceSelection {
  bool visualize;
  holoscan::DeviceId compute;
  holoscan::DeviceId display;
  holoscan::DeviceId composer;
};

inline SarDeviceSelection select_sar_device(AppConfig& cfg, const std::string& path) {
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
  app::prepare_presentation(cfg, path, cfg.presenter, cfg.choose_presenter);
  cfg.visualize_hold_s = cfg.visualization.final_image_hold_s;
  const bool visualize = cfg.visualize;
#else
  app::select_application_device(false, false);
  const bool visualize = false;
#endif
  return {visualize, app::application_device(),
          holoscan::DeviceId{app::application_display_device()},
          holoscan::DeviceId{app::application_composer_device()}};
}

inline radar::MaxTensorExtents prepare_sar_extents(const AppConfig& cfg,
                                                   const std::filesystem::path& path) {
  auto document = app::config::normalize_enum_field<radar::ApertureEmitMode>(
      app::config::load_inherited_document(path), "/aperture_manager/emit_mode");
  radar::ApertureManagerParams seed;
  seed.emit_interval_pulses = cfg.tile_pulses;
  return make_extents(cfg, *app::config::decode_params(document, "/aperture_manager", seed));
}

// Prepare external configuration and device discovery before entering a registry
// constructor transaction. Attaching a Python context then only authors nodes.
struct SarRuntimeInputs {
  holoscan::config::Document document;
  radar::MaxTensorExtents extents;
  SarDeviceSelection devices;
};

inline SarRuntimeInputs prepare_sar_runtime(AppConfig& cfg, const std::filesystem::path& path) {
  return {app::config::load_inherited_document(path), prepare_sar_extents(cfg, path),
          select_sar_device(cfg, path.string())};
}

// Shared native scaffolding for C++ and Python-authored SAR graphs. Each context
// owns its HUD and completion counters independently; operators capture shared
// owners so their lifetime extends through execution-plan teardown.
template <typename T = float, typename TGeom = double>
struct SarGraphRuntime {
  std::string name;
  CliPaths cli;
  AppConfig cfg;
  const SarDeviceSelection devices;
  bool visualize;
  std::shared_ptr<holoscan::Graph> owned_graph;
  holoscan::Graph& graph;
  holoscan::CompileOptions options;
  holoscan::ConnectionOptions queue{.queue_depth = 4U};
  radar::MaxTensorExtents extents, ingest_extents;
  std::shared_ptr<radar::FormationHud> hud{std::make_shared<radar::FormationHud>()};
  std::shared_ptr<app::sar::ImageCount> counter{std::make_shared<app::sar::ImageCount>()};
  holoscan::OperatorHandle<radar::CPHDReaderOp<T, TGeom>> reader{
      graph.op<radar::CPHDReaderOp<T, TGeom>>("cphd_reader", ingest_extents, hud)};
  holoscan::OperatorHandle<radar::SARRangeCompressionOp<T>> range{
      graph.op<radar::SARRangeCompressionOp<T>>("range_compression", ingest_extents)};
  holoscan::OperatorHandle<radar::ApertureManagerOp<T, TGeom>> manager{
      graph.op<radar::ApertureManagerOp<T, TGeom>>("aperture_manager", extents, hud)};
  holoscan::OperatorHandle<radar::BackprojectionOp<T, TGeom>> bp{
      graph.op<radar::BackprojectionOp<T, TGeom>>("backprojection", extents, hud)};
  std::array<int64_t, 2> product_extent{
      cfg.scroll && cfg.scroll_parent.is_valid()
          ? std::array<int64_t, 2>{cfg.scroll_parent.num_v(), cfg.scroll_parent.num_u()}
          : std::array<int64_t, 2>{0, 0}};
  holoscan::OperatorHandle<app::sar::SarImageReportOp> report{
      graph.op<app::sar::SarImageReportOp>("image_report", counter, hud, product_extent)};
  holoscan::PartitionHandle source{graph.partition("source")};
  holoscan::PartitionHandle application{graph.partition("application")};
  holoscan::PartitionHandle formation{graph.partition("formation")};
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
  std::shared_ptr<app::FrameProgress> display_progress{std::make_shared<app::FrameProgress>()};
#endif
  bool has_visualization{false}, has_timeline{false};
  bool finished{false}, started{false};

  SarGraphRuntime(std::string app_name, CliPaths paths, AppConfig config,
                  holoscan::Graph* external_graph = nullptr,
                  const SarRuntimeInputs* prepared = nullptr)
  : name(std::move(app_name)),
    cli(std::move(paths)),
    cfg(std::move(config)),
    devices(prepared == nullptr ? select_sar_device(cfg, cli.config.string()) : prepared->devices),
    visualize(devices.visualize),
    owned_graph(external_graph == nullptr ? std::make_shared<holoscan::Graph>(name) : nullptr),
    graph(external_graph == nullptr ? *owned_graph : *external_graph),
    extents(prepared == nullptr ? prepare_sar_extents(cfg, cli.config) : prepared->extents),
    ingest_extents(make_ingest_extents(extents, cfg)) {
    auto document =
        prepared == nullptr ? app::config::load_inherited_document(cli.config) : prepared->document;
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
    if (visualize && !cfg.headless && cfg.visualization.timeline.enabled)
      hud->enable_progress(manager_params->emit_interval_pulses, reader_params->pulses_per_tick);
    radar::BackprojectionParams bp_seed;
    bp_seed.sample_geometry = cfg.sample_geometry;
    bp_seed.range_bin_spacing = cfg.range_bin_spacing;
    bp_seed.reference_frequency = cfg.reference_frequency;
    auto bp_params = app::config::decode_params(document, "/backprojection", bp_seed);

    reader.set_params(std::move(reader_params));
    bind_outputs(options, "cphd_reader",
                 {"phase_history", "positions", "reference_position", "sample_axis", "pulse_times",
                  "velocities"},
                 devices.compute);
    range.set_params(std::move(range_params));
    bind_outputs(options, "range_compression", {"range_profiles", "range_bins"}, devices.compute);
    manager.set_params(std::move(manager_params));
    bind_outputs(options, "aperture_manager",
                 {"range_profiles_out", "positions_out", "reference_position_out", "weight",
                  "sample_points_out", "control_out"},
                 devices.compute);
    bp.set_params(std::move(bp_params));
    bind_outputs(options, "backprojection", {"image"}, devices.compute);
    // A scroll mosaic publishes a decimated view of its accumulator, so the
    // tensor reaching the report is not the formed strip. Report the strip.
    // Authored partitions only get independent workers when each one has a
    // thread-placement binding. Without that, HSDK 5 runs callbacks inline on
    // the controller — the HSDK 4 greedy default that cannot overlap
    // reader.compute() with BP.compute(). The old EventBased pool of N workers
    // is this binding.
    // serial_baseline exact-cover: each authored partition must be one directed
    // chain. A join of two producers (range and ecef_to_enu both into the
    // manager) in `application` fails compile; put the ECEF transform on
    // `source` with the reader.
    source.add(reader);
    application.add(range).add(manager);
    formation.add(bp).add(report);
  }

  void finish(const holoscan::Output<holoscan::Tensor>* formed) {
    if (finished)
      throw std::logic_error("SAR graph was already finalized");
    graph.add_flow(*formed, report->image, queue);
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
    if (visualize) {
      // Visualization is best-effort: discard stale images when rendering falls behind
      // instead of backpressuring formation, reporting, or an optional tensor dump.
      // The raster edge carries the formed image, which on tiled_strip is the
      // whole parent lattice -- 3.62 GiB a slot on the Imperial strip. Every
      // declared slot is a full copy of it, and a drop-oldest consumer only ever
      // renders the newest, so depth beyond 1 is reserved and never read. At
      // depth 4 the continuous_strip pools came to 100.7 GiB against 95.6 GiB of
      // GPU and the run died at PROFILE_MATERIALIZATION_FAILED before start();
      // depth 1 was 68.2 GiB. Those two figures predate continuous_strip's
      // decimated mosaic view, which is now ~16 MiB a slot, so they now bound
      // tiled_strip rather than both. Depth 1 still holds for the reason above:
      // the extra slots are never read. Read the slot counts with SAR_DUMP_PLAN
      // before raising this.
      const holoscan::ConnectionOptions raster_queue{
          .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
      const holoscan::ConnectionOptions display_queue{
          .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 8U};
      const holoscan::ConnectionOptions overlay_queue{
          .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
      const auto& viz_geometry = cfg.scroll                       ? cfg.scroll_parent
                                 : cfg.parent_geometry.is_valid() ? cfg.parent_geometry
                                                                  : cfg.sample_geometry;
      // Strip mosaics place along-track v horizontally to fit a landscape display.
      const bool transpose_display_axes = cfg.scroll || !cfg.tile_geometries.empty();
      const int64_t raster_max_v = std::max(viz_geometry.num_v(), *extents.max_image_v);
      const int64_t raster_max_u = std::max(viz_geometry.num_u(), *extents.max_image_u);
      const bool mosaic = !cfg.tile_geometries.empty();
      app::sar::OverlayConfig overlay;
      overlay.scene_names = cfg.scene_names;
      overlay.show_playback_lag = cfg.show_playback_lag;
      overlay.diag = [hud = hud, show_lag = cfg.show_playback_lag, mosaic, scroll = cfg.scroll,
                      parent = cfg.parent_geometry, scroll_parent = cfg.scroll_parent,
                      geoms = cfg.collection_geometries, to_enu = cfg.transform_to_scene_enu,
                      origin = cfg.enu_origin_ecef]() {
        const auto snap = hud->snapshot();
        app::sar::FormationDiag d;
        d.bp_compute_s = snap.bp.compute_s;
        d.collection_elapsed_s = snap.bp.collection_elapsed_s;
        d.budget_s = snap.bp.budget_s;
        d.budget_playback_rate = snap.bp.budget_playback_rate;
        d.capture_time_ns = snap.bp.capture_time_ns;
        d.pulses = snap.bp.pulses;
        d.pixels = snap.bp.pixels;
        d.bp_num_v = snap.bp.num_v;
        d.bp_num_u = snap.bp.num_u;
        d.rate_line = snap.rate_line;
        d.emit_at = snap.bp.emit_at;
        d.last_packet_at =
            snap.bp.last_packet_at.has_value() ? snap.bp.last_packet_at : snap.last_packet_at;
        if (show_lag) {
          d.playback_valid = snap.playback.valid;
          d.playback_paced = snap.playback.paced;
          d.playback_behind_s = snap.playback.behind_s;
          d.playback_target_s = snap.playback.target_s();
          d.playback_wall_s = snap.playback.wall_s;
        }
        if (snap.bp.emit_at.has_value() && d.last_packet_at.has_value()) {
          d.form_latency_s =
              std::chrono::duration<double>(*snap.bp.emit_at - *d.last_packet_at).count();
        }
        const auto& apc = snap.apc;
        if (apc.valid) {
          const radar::SampleGeometry* g = nullptr;
          if (mosaic && parent.is_valid()) {
            g = &parent;
          } else if (scroll && scroll_parent.is_valid()) {
            g = &scroll_parent;
          } else if (!geoms.empty()) {
            const size_t i = static_cast<size_t>(apc.file_index) % geoms.size();
            g = &geoms[i];
          }
          if (g != nullptr) {
            const auto uv = apc_raster_uv(apc.ecef, *g, to_enu, origin);
            if (uv.has_value()) {
              d.apc_valid = true;
              d.raster_u = (*uv)[0];
              d.raster_v = (*uv)[1];
            }
          }
        }
        return d;
      };
      const bool strip_fit = cfg.scroll || mosaic;
      const int64_t viz_u = (cfg.viz_crop_nu > 0) ? cfg.viz_crop_nu : viz_geometry.num_u();
      int64_t lut_side = cfg.visualize_size;
      uint32_t width = 0;
      uint32_t height = 0;
      // Fit parent strips within the desktop work area and configured display
      // cap, reserving space for the footer. Forwarded work areas may span
      // multiple laptop monitors; never expand to their combined width.
      if (strip_fit && !cfg.headless) {
        int work_w = 0;
        int work_h = 0;
        if (desktop_work_area(&work_w, &work_h)) {
          const int timeline_reserve = static_cast<int>(app::sar::TimelineRasterOp::kHeight) + 48;
          const int fit_h = std::max(1, work_h - timeline_reserve);
          const auto fit =
              holoviz_fit_wh(viz_geometry.num_v(), viz_u, work_w, fit_h, cfg.visualize_size);
          width = fit[0];
          height = fit[1];
          lut_side = static_cast<int64_t>(std::max(width, height));
          HOLOSCAN_LOG_INFO("SAR: Holoviz {}x{} (desktop work area {}x{}; visualize_size {} cap)",
                            width, height, work_w, work_h, cfg.visualize_size);
        } else {
          const auto wh = holoviz_window_wh(viz_geometry.num_v(), viz_u, cfg.visualize_size);
          width = wh[0];
          height = wh[1];
          HOLOSCAN_LOG_INFO("SAR: Holoviz {}x{} (visualize_size {}; no desktop work area)", width,
                            height, cfg.visualize_size);
        }
      } else {
        const int64_t max_side = std::max(viz_geometry.num_v(), viz_geometry.num_u());
        const int64_t horizontal_extent =
            transpose_display_axes ? viz_geometry.num_v() : viz_geometry.num_u();
        const int64_t vertical_extent =
            transpose_display_axes ? viz_geometry.num_u() : viz_geometry.num_v();
        width =
            static_cast<uint32_t>(std::max<int64_t>(1, lut_side * horizontal_extent / max_side));
        height = static_cast<uint32_t>(std::max<int64_t>(1, lut_side * vertical_extent / max_side));
      }
      if (cfg.explicit_window_size) {
        width = static_cast<uint32_t>(cfg.visualization.window_width);
        height = static_cast<uint32_t>(cfg.visualization.window_height);
      }
      const bool docked =
          cfg.visualization.timeline.enabled && cfg.visualization.timeline.placement == "bottom";
      const int status_width =
          docked && width >= 2 ? std::min(320, static_cast<int>(width) / 2) : 0;
      const int footer_height = status_width > 0 ? cfg.visualization.timeline.window_height : 0;
      overlay.canvas_width = width;
      overlay.image_height = height;
      if (status_width > 0) {
        overlay.status_width = status_width;
        overlay.footer_height = footer_height;
      }
      // Hold is session-level after the image, not a sleep in this operator:
      // raster and Holoviz share `visualization`, so a hold here never presents.
      auto raster = graph.op<app::sar::SarImageVizOp>(
          "sar_image_viz", raster_max_v, raster_max_u, lut_side, transpose_display_axes,
          std::move(overlay), product_extent, cfg.visualization.display_rate_hz,
          [hud = hud] {
            return hud->snapshot().reader_stopped;
          },
          !cfg.headless);
      bind_outputs(options, "sar_image_viz", {"frame"}, devices.compute);
      auto raster_partition = graph.partition("raster");
      raster_partition.add(raster);
      auto visualization = graph.partition("visualization");
      const auto renderer_device = devices.composer;
      auto composed = graph.op<app::viz::FrameComposerOp>(
          "holoviz", width, height + footer_height, app::sar::kHudMaxGlyphs, true,
          app::sar::kHudMaxLines, lut_side, status_width > 0 ? height : 0);
      visualization.add(composed);
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "holoviz", .output_port = "output", .device = renderer_device});
      graph.add_flow(*formed, raster->image, raster_queue);
      graph.add_flow(raster->labels, composed->tensor_layer_input(1), overlay_queue);
      graph.add_flow(raster->overlay, composed->tensor_layer_input(2), overlay_queue);
      app::viz::connect_image_devices(graph, options, raster_partition, visualization,
                                      raster->frame, composed->image_layer_input(0), "sar_composer",
                                      lut_side, lut_side, devices.compute, renderer_device,
                                      overlay_queue);
      if (!docked)
        app::viz::add_image_presentation(graph, options, visualization, composed->output(), "sar",
                                         width, height, cfg.visualization.window_title,
                                         cfg.visualization, display_progress, {}, devices.composer,
                                         devices.display);
      has_visualization = true;
      if (cfg.visualization.timeline.enabled) {
        auto timeline = graph.op<app::sar::TimelineRasterOp>(
            "timeline_raster", hud, cfg.scene_names, cfg.show_playback_lag,
            cfg.visualization.timeline.display_rate_hz,
            docked ? width - status_width : cfg.visualization.timeline.window_width,
            cfg.visualization.timeline.window_height,
            cfg.headless ? std::shared_ptr<app::FrameProgress>{} : display_progress);
        bind_outputs(options, "timeline_raster", {"frame"}, devices.compute);
        auto timeline_part = graph.partition("timeline");
        timeline_part.add(timeline);
        auto timeline_display = graph.partition("timeline_display");
        const auto tw = docked ? width - status_width : cfg.visualization.timeline.window_width;
        const auto th = cfg.visualization.timeline.window_height;
        auto timeline_frame = graph.op<app::viz::FrameComposerOp>("timeline_frame", tw, th, 16384U);
        timeline_display.add(timeline_frame);
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "timeline_frame", .output_port = "output", .device = renderer_device});
        graph.add_flow(timeline->labels, timeline_frame->tensor_layer_input(1), overlay_queue);
        app::viz::connect_image_devices(graph, options, timeline_part, timeline_display,
                                        timeline->frame, timeline_frame->image_layer_input(0),
                                        "timeline_composer", tw, th, devices.compute,
                                        renderer_device, overlay_queue);
        if (docked) {
          app::viz::add_docked_presentation(graph, options, visualization, composed->output(),
                                            timeline_display, timeline_frame->output(), width,
                                            height, th, cfg.visualization, display_progress,
                                            devices.composer, devices.display, status_width);
        } else {
          auto timeline_config = cfg.visualization;
          timeline_config.display_rate_hz = cfg.visualization.timeline.display_rate_hz;
          app::viz::add_image_presentation(graph, options, timeline_display,
                                           timeline_frame->output(), "timeline", tw, th,
                                           "SAR timeline", timeline_config, {}, "bottom_left",
                                           devices.composer, devices.display);
        }
        has_timeline = true;
      }
    }
#else
    if (cfg.visualize)
      HOLOSCAN_LOG_WARN("SAR visualization component is unavailable");
#endif
    if (cfg.dump) {
      auto dump = graph.op<app::sar::TrackedTensorDumpOp>("tensor_dump", counter);
      radar::TensorDumpParams params;
      params.path = cfg.dump_path;
      if (cfg.dump_max_files >= 0)
        params.max_files = cfg.dump_max_files;
      dump.set_params(params);
      formation.add(dump);
      graph.add_flow(*formed, dump->tensor, queue);
    }
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    const holoscan::ThreadPlacement worker_lane{};
    options.deployment.bind_thread_placement("source", worker_lane)
        .bind_thread_placement("application", worker_lane)
        .bind_thread_placement("formation", worker_lane);
    if (has_visualization)
      options.deployment.bind_thread_placement("visualization", worker_lane)
          .bind_thread_placement("raster", worker_lane);
    if (has_timeline)
      options.deployment.bind_thread_placement("timeline", worker_lane)
          .bind_thread_placement("timeline_display", worker_lane);
    finished = true;
  }

  int64_t run(app::ShutdownSignals* signals = nullptr,
              const std::function<void()>& caller_poll = {}) {
    if (!finished || started)
      throw std::logic_error("SAR graph must be finalized and run once");
    started = true;
    const auto plan = holoscan::compile(graph, std::move(options));
    // Pool sizing is the binding constraint on a retained-aperture run and the
    // plan is the only place the slot counts are visible: a declared depth of 2
    // can materialize as 9. SAR_DUMP_PLAN=<file> writes it out.
    if (const char* plan_dump = std::getenv("SAR_DUMP_PLAN")) {
      std::ofstream(plan_dump) << plan.json();
    }
    if (!plan.ok()) {
      std::string error = "SAR graph compile failed:";
      for (const auto& diagnostic : plan.diagnostics()) {
        if (diagnostic.severity == holoscan::DiagnosticSeverity::kError)
          error += " [" + holoscan::format_diagnostic(diagnostic, plan.graph()) + "]";
      }
      throw std::runtime_error(error);
    }
    const auto sched = plan.scheduler_plan();
    HOLOSCAN_LOG_INFO("SAR: scheduler workers={} overlap={} max_concurrent={} lanes={}",
                      sched.worker_count, sched.activation_overlap ? 1 : 0,
                      sched.max_concurrent_activations, sched.execution_lane_count);
    for (const auto& part : plan.execution_partitions()) {
      HOLOSCAN_LOG_INFO("SAR: partition '{}' lane={}",
                        part.authored_name.value_or(std::string("(auto)")), part.execution_lane);
    }
    const bool saving_images = cfg.dump && cfg.dump_max_files != 0;
    HOLOSCAN_LOG_INFO("SAR: display={}; image saving={}",
                      has_visualization ? (cfg.headless ? "headless" : "window") : "off",
                      saving_images ? "on" : "off");
    if (saving_images) {
      const auto output = radar::dump_paths::display_path(cfg.dump_path);
      HOLOSCAN_LOG_INFO("SAR: image output directory{}: {}",
                        output.is_relative() ? " (relative to repository root)" : "",
                        output.parent_path().string());
      HOLOSCAN_LOG_INFO("SAR: image filename template: {}", output.filename().string());
      HOLOSCAN_LOG_INFO("SAR: each saved image will log its exact path and viewer command");
    } else if (cfg.dump) {
      HOLOSCAN_LOG_INFO(
          "SAR: images will not be saved because output.max_files is 0; set a positive limit");
    } else {
      HOLOSCAN_LOG_INFO("SAR: images will not be saved; set dump: true in {} to save .npy images",
                        config_hint_path(cli.config, name.c_str()));
    }
    auto session = holoscan::run_async(plan);
    const std::function<void()> poll = [&] {
      if (signals && signals->requested())
        session.request_stop();
      if (caller_poll)
        caller_poll();
    };
    try {
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
      if (has_visualization && !cfg.headless &&
          !cfg.visualization.x11_presentation.value_or(true)) {
        if (cfg.scroll || !cfg.tile_geometries.empty())
          spawn_place_window_top_left("SAR formation");
        if (has_timeline && cfg.visualization.timeline.placement == "window")
          spawn_place_window_bottom_left("SAR timeline");
      }
#endif
      if (!cfg.reader_loop) {
        // request_stop does not drain queued samples. After reader EOF, wait
        // for the terminal closing image to reach reporting, including fan-out
        // slices. A BP wall timestamp alone can refer to an earlier image.
        constexpr auto kDrainWait = std::chrono::seconds{180};
        constexpr auto kPoll = std::chrono::milliseconds{50};
        std::optional<std::chrono::steady_clock::time_point> drain_deadline;
        auto eos_flushed = [&] {
          const auto snap = hud->snapshot();
          return sar_eof_drained(snap,
                                 counter->reported_capture_ns.load(std::memory_order_acquire)) &&
                 (!cfg.dump ||
                  counter->dumped_capture_ns.load() >= counter->reported_capture_ns.load());
        };
        while (!eos_flushed()) {
          if (poll)
            poll();
          const auto now = std::chrono::steady_clock::now();
          if (!drain_deadline.has_value()) {
            const auto snap = hud->snapshot();
            if (snap.reader_stopped)
              drain_deadline = now + kDrainWait;
          }
          if (drain_deadline.has_value() && now >= *drain_deadline)
            break;
          auto slice_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(kPoll);
          if (drain_deadline.has_value()) {
            const auto remaining =
                std::chrono::duration_cast<std::chrono::nanoseconds>(*drain_deadline - now);
            if (remaining < slice_ns)
              slice_ns = remaining;
          }
          if (session.wait_for(slice_ns))
            break;
        }
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
        if (eos_flushed() && has_visualization) {
          const auto display_deadline =
              std::chrono::steady_clock::now() + app::kDisplayDrainTimeout;
          while (display_progress->capture_ns.load() < counter->reported_capture_ns.load() &&
                 std::chrono::steady_clock::now() < display_deadline && !session.wait_for(kPoll)) {
            if (poll)
              poll();
          }
        }
#endif
        if (eos_flushed() && has_visualization && !cfg.headless && cfg.visualize_hold_s > 0.0) {
          const auto hold = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(cfg.visualize_hold_s));
          const auto hold_end = std::chrono::steady_clock::now() + hold;
          while (std::chrono::steady_clock::now() < hold_end &&
                 !session.wait_for(std::chrono::milliseconds{50})) {
            if (poll)
              poll();
          }
        }
        if (!(has_visualization && !cfg.headless && cfg.visualize_hold_s < 0.0 && eos_flushed()))
          session.request_stop();
      }
      while (!session.wait_for(std::chrono::milliseconds{50})) {
        if (poll)
          poll();
      }
      session.wait();
    } catch (...) {
      session.request_stop();
      session.wait();
      throw;
    }
    if (session.termination() == holoscan::RunTermination::kFailed ||
        !session.diagnostics().empty()) {
      std::string error = "SAR graph run diagnostics:";
      bool full_queue = false;
      bool vulkan_failure = false;
      for (const auto& diagnostic : session.diagnostics()) {
        const std::string formatted = holoscan::format_diagnostic(diagnostic, plan.graph());
        error += " [" + formatted + "]";
        full_queue = full_queue || app::sar::is_backpressure_failure(diagnostic);
        vulkan_failure = vulkan_failure || app::sar::is_vulkan_failure(diagnostic);
      }
      if (session.diagnostics().empty())
        error += " [failed with no diagnostics]";
      if (full_queue || vulkan_failure) {
        const char* term = std::getenv("TERM");
        const bool color = ::isatty(::fileno(stderr)) != 0 && std::getenv("NO_COLOR") == nullptr &&
                           (term == nullptr || std::string(term) != "dumb");
        if (vulkan_failure) {
          error += app::sar::format_failure_banner(
              "VULKAN: VISUALIZATION FAILED",
              "Holoviz could not initialize or use the Vulkan graphics device.\n\n"
              "Checks:\n"
              " 1. Check host NVIDIA drivers, Vulkan support, and container GPU access.\n"
              " 2. Check display forwarding and authorization.\n"
              " 3. On DGX Spark/GB10, check the documented cache/memory-pressure issue.\n"
              "Troubleshooting: ./applications/sar/README.md (Vulkan/display sections).\n",
              color);
        } else {
          const std::string body =
              "Pipeline cannot keep up - output queue or tensor pool is full.\n"
              "This is expected when data playback rate is higher than the compute capacity.\n\n"
              "Options:\n"
              " 1. Lower playback via cphd_reader.playback_rate in " +
              config_hint_path(cli.config, name.c_str()) +
              "\n"
              " 2. Decrease image size\n"
              " 3. Use a platform with a more capable GPU\n";
          error += app::sar::format_failure_banner("BACKPRESSURE: RUN STOPPED", body, color);
        }
      }
      throw std::runtime_error(error);
    }
    const auto images = counter->count.load();
    if (images == 0 && session.termination() != holoscan::RunTermination::kStopped)
      throw std::runtime_error("SAR graph ended without an image");
    if (session.termination() == holoscan::RunTermination::kStopped)
      HOLOSCAN_LOG_INFO("{}: stopped after {} image(s)", name, images);
    return images;
  }
};

template <typename T = float, typename TGeom = double, typename LoadConfig, typename BuildGraph>
int run_sar_graph_app_at(int argc, char** argv, const char* name, LoadConfig load_config,
                         BuildGraph build_graph) {
  try {
    app::ShutdownSignals signals;
    const CliPaths cli = parse_cli(argc, argv, (std::string(name) + ".yaml").c_str());
    auto config = load_config(cli.config, cli.cphd_override);
    config.presenter = cli.presenter;
    config.choose_presenter = cli.choose_presenter;
    SarGraphRuntime<T, TGeom> runtime(name, cli, std::move(config));
    auto& r = runtime;
    r.graph.add_flow(r.reader->phase_history, r.range->phase_history, r.queue);
    const auto* formed = [&] {
      // Workflows with app-owned diagnostics can opt into the shared mailbox.
      if constexpr (requires {
                      build_graph(Precision<T, TGeom>{}, r.graph, r.options, r.queue, r.cfg,
                                  r.reader, r.range, r.manager, r.bp, r.source, r.application,
                                  r.formation, r.extents, r.hud);
                    }) {
        return build_graph(Precision<T, TGeom>{}, r.graph, r.options, r.queue, r.cfg, r.reader,
                           r.range, r.manager, r.bp, r.source, r.application, r.formation,
                           r.extents, r.hud);
      } else {
        return build_graph(Precision<T, TGeom>{}, r.graph, r.options, r.queue, r.cfg, r.reader,
                           r.range, r.manager, r.bp, r.source, r.application, r.formation,
                           r.extents);
      }
    }();
    r.finish(formed);
    r.run(&signals);
    return 0;
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("{}: {}", name, error.what());
    return 1;
  }
}

// Samples default to complex64 with geometry in float64, which is what the
// sensors produce: very few radar systems deliver more than single-precision
// samples, and the FMCW applications have always instantiated at `float`. It is
// also where SAR's memory is -- the sample and image tensors are the dominant
// pools and geometry is a few per cent of them, so this roughly halves every
// SAR graph. The two axes are not one choice: ECEF positions cannot go to
// float32, where one ulp at a 6.4e6 m radius is 0.76 m, some 25 wavelengths at
// X band, so only the sample side moves. `sample_precision: double` opts back
// in for a collect that genuinely carries more. See the precision section of
// SAR/OPERATOR_API.md.
inline bool single_sample_precision(const std::filesystem::path& config) {
  const auto document = app::config::load_inherited_document(config);
  if (!document.find("/sample_precision")) {
    return true;
  }
  const auto value = document.get<std::string>("/sample_precision");
  if (value == "single") {
    return true;
  }
  if (value != "double") {
    throw std::runtime_error("SAR: sample_precision must be 'single' or 'double'");
  }
  return false;
}

// The precision has to be chosen before the driver is instantiated, so this one
// key is read here rather than waiting for `load_config` to run inside it.
template <typename LoadConfig, typename BuildGraph>
int run_sar_graph_app(int argc, char** argv, const char* name, LoadConfig load_config,
                      BuildGraph build_graph) {
  // These two run before the driver, so they are outside its own handler: an
  // unreadable config path or a bad `sample_precision` would otherwise leave
  // main() to terminate on the exception instead of reporting it.
  bool single = true;
  try {
    const CliPaths cli = parse_cli(argc, argv, (std::string(name) + ".yaml").c_str());
    single = single_sample_precision(cli.config);
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("{}: {}", name, error.what());
    return 1;
  }
  if (single) {
    return run_sar_graph_app_at<float, double>(argc, argv, name, load_config, build_graph);
  }
  return run_sar_graph_app_at<double, double>(argc, argv, name, load_config, build_graph);
}
