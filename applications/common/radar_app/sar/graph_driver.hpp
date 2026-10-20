// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
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
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/threading/thread_placement.hpp>
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
#include "radar_app/sar/timeline_raster.hpp"
#include "radar_app/sar/window_place.hpp"
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

// Operator Error text stays as-is. When the run fails because a required emit
// could not land, append the calling demo's teaching paragraph.
inline bool mentions_full_queue(const std::string& text) {
  return text.find("output queue is full") != std::string::npos ||
         text.find("no free tensor buffer") != std::string::npos;
}

template <typename LoadConfig, typename BuildGraph>
int run_sar_graph_app(int argc, char** argv, const char* name, LoadConfig load_config,
                      BuildGraph build_graph, const char* failure_hint = nullptr) {
  try {
    // Own SIGINT/SIGTERM before Holoviz or the runtime start threads. Host
    // `holoscan run` maps Ctrl-C to `docker stop` (SIGTERM); without this
    // domain the looping presets ignore both signals until the CLI SIGKILLs.
    auto created_signals = holoscan::ManagedShutdownSignals::create();
    if (!created_signals) {
      throw std::runtime_error(std::string("SAR: managed SIGINT/SIGTERM domain failed: ") +
                               holoscan::runtime_error_message(created_signals.error()));
    }
    auto managed_signals = std::move(created_signals).value();
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
    auto hud = std::make_shared<radar::FormationHud>();
    auto reader = graph.op<radar::CPHDReaderOp<double>>("cphd_reader", extents, hud);
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
    auto bp = graph.op<radar::BackprojectionOp<double>>("backprojection", extents, hud);
    bp.set_params(std::move(bp_params));
    bind_outputs(options, "backprojection", {"image"});
    auto counter = std::make_shared<app::sar::ImageCount>();
    auto report = graph.op<app::sar::SarImageReportOp>("image_report", counter, hud);
    // Authored partitions only get independent workers when each one has a
    // thread-placement binding. Without that, HSDK 5 runs callbacks inline on
    // the controller — the HSDK 4 greedy default that cannot overlap
    // reader.compute() with BP.compute(). The old EventBased pool of N workers
    // is this binding.
    // serial_baseline exact-cover: each authored partition must be one directed
    // chain. A join of two producers (range and ecef_to_enu both into the
    // manager) in `application` fails compile; put the ECEF transform on
    // `source` with the reader.
    auto source = graph.partition("source");
    source.add(reader);
    auto application = graph.partition("application");
    application.add(range).add(manager);
    auto formation = graph.partition("formation");
    formation.add(bp).add(report);
    bool has_visualization = false;
    bool has_timeline = false;
    graph.add_flow(reader->phase_history, range->phase_history, queue);
    const auto* formed = build_graph(graph, options, queue, cfg, reader, range, manager, bp, source,
                                     application, formation, extents);
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
      // Visualization is best-effort: discard stale images when rendering falls behind
      // instead of backpressuring formation, reporting, or an optional tensor dump.
      const holoscan::ConnectionOptions raster_queue{
          .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U};
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
      overlay.diag = [hud, show_lag = cfg.show_playback_lag, mosaic, scroll = cfg.scroll,
                      parent = cfg.parent_geometry, scroll_parent = cfg.scroll_parent,
                      geoms = cfg.collection_geometries, to_enu = cfg.transform_to_scene_enu,
                      origin = cfg.enu_origin_ecef]() {
        const auto snap = hud->snapshot();
        app::sar::FormationDiag d;
        d.bp_compute_s = snap.bp.compute_s;
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
      // Parent strips fill the desktop work-area width (and leave a band for
      // the Gantt). visualize_size is the square-collect size and the no-DISPLAY
      // fallback. HolovizOp has no position field; strip windows are moved by
      // title after they map.
      if (strip_fit && !cfg.headless) {
        int work_w = 0;
        int work_h = 0;
        if (desktop_work_area(&work_w, &work_h)) {
          const int timeline_reserve = static_cast<int>(app::sar::TimelineRasterOp::kHeight) + 48;
          const int fit_h = std::max(1, work_h - timeline_reserve);
          const auto fit = holoviz_fit_wh(viz_geometry.num_v(), viz_u, work_w, fit_h);
          width = fit[0];
          height = fit[1];
          lut_side = static_cast<int64_t>(std::max(width, height));
          HOLOSCAN_LOG_INFO(
              "SAR: Holoviz {}x{} (desktop work area {}x{}; visualize_size {} fallback)", width,
              height, work_w, work_h, cfg.visualize_size);
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
      // Hold is session-level after the image, not a sleep in this operator:
      // raster and Holoviz share `visualization`, so a hold here never presents.
      auto raster =
          graph.op<app::sar::SarImageVizOp>("sar_image_viz", raster_max_v, raster_max_u, lut_side,
                                            transpose_display_axes, std::move(overlay));
      bind_outputs(options, "sar_image_viz", {"frame"});
      auto make_layers = [&] {
        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("sar_image",
                               holoscan::visualization::ImageLayerDesc{
                                   .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm,
                                   .max_width = static_cast<uint32_t>(lut_side),
                                   .max_height = static_cast<uint32_t>(lut_side)});
        layers.add_geometry_layer(
            "look", holoscan::visualization::GeometryLayerDesc{
                        .topologies = {holoscan::visualization::GeometryTopology::kLineList},
                        .max_primitives = app::sar::kHudMaxLines,
                        .default_line_width = 2.0F});
        layers.add_text_layer(
            "labels", holoscan::visualization::TextLayerDesc{.max_glyphs = app::sar::kHudMaxGlyphs,
                                                             .normalized_coordinates = true});
        return layers;
      };
      const auto hud_age = std::chrono::seconds{5};
      auto wire_overlay = [&](auto& viz) {
        graph.add_flow(*formed, raster->image, raster_queue);
        graph.add_flow(raster->overlay, viz->tensor_layer_input(1), overlay_queue);
        graph.add_flow(raster->labels, viz->tensor_layer_input(2), overlay_queue);
        graph.add_flow(raster->frame, viz->image_layer_input(0), display_queue);
      };
      if (cfg.headless) {
        auto viz = graph.op<holoscan::visualization::HolovizOp>(
            "holoviz", make_layers(),
            holoscan::visualization::CudaSinkDesc{
                .width = width,
                .height = height,
                .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm},
            holoscan::visualization::LayerTrigger{}, hud_age);
        auto sink = graph.op<app::sar::SarImageSchemaSinkOp>("viz_sink");
        auto visualization = graph.partition("visualization");
        visualization.add(raster).add(viz).add(sink);
        has_visualization = true;
        bind_outputs(options, "holoviz", {"output"});
        wire_overlay(viz);
        graph.add_flow(viz->output(), sink->frame, queue);
      } else {
        auto viz = graph.op<holoscan::visualization::HolovizOp>(
            "holoviz", make_layers(),
            holoscan::visualization::WindowSinkDesc{.title = "SAR formation",
                                                    .width = width,
                                                    .height = height,
                                                    .close_on_escape = true},
            holoscan::visualization::LayerTrigger{}, hud_age);
        auto visualization = graph.partition("visualization");
        visualization.add(raster).add(viz);
        has_visualization = true;
        wire_overlay(viz);
        auto timeline = graph.op<app::sar::TimelineRasterOp>(
            "timeline_raster", hud, cfg.scene_names, cfg.show_playback_lag);
        bind_outputs(options, "timeline_raster", {"frame"});
        holoscan::visualization::LayerStack tl_layers;
        tl_layers.add_image_layer(
            "timeline",
            holoscan::visualization::ImageLayerDesc{
                .format = holoscan::visualization::ImageFormat::kR8G8B8A8Unorm,
                .max_width = static_cast<uint32_t>(app::sar::TimelineRasterOp::kWidth),
                .max_height = static_cast<uint32_t>(app::sar::TimelineRasterOp::kHeight)});
        auto timeline_viz = graph.op<holoscan::visualization::HolovizOp>(
            "timeline_holoviz", std::move(tl_layers),
            holoscan::visualization::WindowSinkDesc{
                .title = "SAR timeline",
                .width = static_cast<uint32_t>(app::sar::TimelineRasterOp::kWidth),
                .height = static_cast<uint32_t>(app::sar::TimelineRasterOp::kHeight),
                .close_on_escape = true},
            holoscan::visualization::LayerTrigger{}, std::chrono::milliseconds(500));
        // serial_baseline exact-cover: one connected chain per authored partition.
        // The Gantt is an OnClock producer + window, independent of formation Holoviz.
        auto timeline_part = graph.partition("timeline");
        timeline_part.add(timeline).add(timeline_viz);
        has_timeline = true;
        graph.add_flow(timeline->frame, timeline_viz->image_layer_input(0), overlay_queue);
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
      formation.add(dump);
      graph.add_flow(*formed, dump->tensor, queue);
    }
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    const holoscan::ThreadPlacement worker_lane{};
    options.deployment.bind_thread_placement("source", worker_lane)
        .bind_thread_placement("application", worker_lane)
        .bind_thread_placement("formation", worker_lane);
    if (has_visualization)
      options.deployment.bind_thread_placement("visualization", worker_lane);
    if (has_timeline)
      options.deployment.bind_thread_placement("timeline", worker_lane);
    const auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      const std::string json = std::string(plan.json());
      std::string hint;
      if (json.find("holoscan.partition.exact_cover") != std::string::npos) {
        hint =
            " (partition exact_cover: each authored partition must be one directed chain, "
            "not a join of two producers)";
      }
      throw std::runtime_error(std::string("SAR graph compile failed") + hint);
    }
    const auto sched = plan.scheduler_plan();
    HOLOSCAN_LOG_INFO("SAR: scheduler workers={} overlap={} max_concurrent={} lanes={}",
                      sched.worker_count, sched.activation_overlap ? 1 : 0,
                      sched.max_concurrent_activations, sched.execution_lane_count);
    for (const auto& part : plan.execution_partitions()) {
      HOLOSCAN_LOG_INFO("SAR: partition '{}' lane={}",
                        part.authored_name.value_or(std::string("(auto)")), part.execution_lane);
    }
    auto session = holoscan::run_async(plan, managed_signals);
#ifdef HOLOSCAN_RADAR_SAR_HOLOVIZ
    if (has_visualization && !cfg.headless) {
      if (cfg.scroll || !cfg.tile_geometries.empty())
        spawn_place_window_top_left("SAR formation");
      if (has_timeline)
        spawn_place_window_bottom_left("SAR timeline");
    }
#endif
    if (!cfg.reader_loop) {
      // request_stop does not drain queued samples. The reader parks after D22;
      // wait until that last batch has formed an image (or SIGINT) before stop.
      constexpr auto kDrainWait = std::chrono::seconds{180};
      constexpr auto kPoll = std::chrono::milliseconds{50};
      const auto drain_deadline = std::chrono::steady_clock::now() + kDrainWait;
      auto eos_flushed = [&] {
        const auto snap = hud->snapshot();
        if (snap.spans.empty() || !snap.spans.back().dwell_end.has_value())
          return false;
        return snap.bp.emit_at.has_value() && snap.last_packet_at.has_value() &&
               *snap.bp.emit_at >= *snap.last_packet_at;
      };
      while (!eos_flushed()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= drain_deadline)
          break;
        const auto remaining = drain_deadline - now;
        const auto slice_ns =
            std::min(std::chrono::duration_cast<std::chrono::nanoseconds>(remaining),
                     std::chrono::duration_cast<std::chrono::nanoseconds>(kPoll));
        if (session.wait_for(slice_ns))
          break;
      }
      if (eos_flushed() && has_visualization && !cfg.headless && cfg.visualize_hold_s > 0.0) {
        const auto hold = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(cfg.visualize_hold_s));
        [[maybe_unused]] const bool hold_finished = session.wait_for(hold);
      }
      if (!(has_visualization && cfg.visualize_hold_s < 0.0 && eos_flushed()))
        session.request_stop();
    }
    session.wait();
    if (const auto closed = managed_signals.close(); !closed) {
      throw std::runtime_error(std::string("SAR: managed SIGINT/SIGTERM domain close failed: ") +
                               holoscan::runtime_error_message(closed.error()));
    }
    if (session.termination() == holoscan::RunTermination::kFailed ||
        !session.diagnostics().empty()) {
      std::string error = "SAR graph run diagnostics:";
      bool full_queue = false;
      for (const auto& diagnostic : session.diagnostics()) {
        const std::string formatted = holoscan::format_diagnostic(diagnostic, plan.graph());
        error += " [" + formatted + "]";
        full_queue = full_queue || mentions_full_queue(formatted);
      }
      if (session.diagnostics().empty())
        error += " [failed with no diagnostics]";
      if (full_queue && failure_hint != nullptr && failure_hint[0] != '\0') {
        error += "\nNote: ";
        error += failure_hint;
      }
      throw std::runtime_error(error);
    }
    const auto images = counter->count.load();
    if (images == 0)
      throw std::runtime_error("SAR graph ended without an image");
    if (session.termination() == holoscan::RunTermination::kStopped)
      HOLOSCAN_LOG_INFO("{}: stopped after {} image(s)", name, images);
    return 0;
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("{}: {}", name, error.what());
    return 1;
  }
}
