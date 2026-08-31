// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mode-1 / 2 / 3 / 4 / 5 SAR formation. Holoviz when visualize: true.
// TensorDumpOp when dump: true (off by default). The dump is the
// offline PNG checkpoint (tools/sar_image/form_image.py).
//
//   CPHDReaderOp → SARRangeCompressionOp → ApertureManagerOp
//                                        → BackprojectionOp
//                                        [→ TileMosaicOp]
//                                        [→ ScrollMosaicOp]
//                                        [→ TensorDumpOp]
//                                        [→ SarImageToLutOp
//                                            → HoldLastImageOp
//                                            → HolovizOp]
//                                        [→ TimelineRasterOp
//                                            → HolovizOp]
//   reader.positions / reference_position bypass range compression
//   and go through the manager. Mode 2 wires manager.weight.
//   Mode 5 inserts CollectionBoundaryStagerOp on the
//   range_profiles edge so the next lattice is staged into the
//   manager (and dr / f0 into BP) on collection_boundary.
//   Mode 3 inserts TileGeometryStagerOp on the same edge so the
//   next tile lattice is staged on the pulse cadence, and
//   TileMosaicOp holds finished squares in a parent raster.
//   Mode 4 inserts ScrollGeometryStagerOp on range_profiles +
//   positions + reference_position; geometry goes to BP
//   (geometry_change = scroll), not through the manager.
//   When emit_interval_pulses < pulses_per_tick, the stager
//   slices the reader cube so the manager sees the interval.
//   ScrollMosaicOp blits the window into the parent strip so
//   Holoviz matches mode 3.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/sar_formation/cpp/sar_formation \
//            [mode01.yaml | mode02.yaml | mode03.yaml |
//             mode04.yaml | mode05.yaml] [<collection.cphd>]

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_reader_op/collection_paths.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_common/math/constants.hpp"
#include "sar/aperture_manager_op/aperture_manager_op.hpp"
#include "sar/backprojection_op/backprojection_op.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir)
#include "custom_ops.hpp"   // NOLINT(build/include_subdir)
#include "window_place.hpp"  // NOLINT(build/include_subdir)

namespace radar = holoscan::radar;
using holoscan::radar::app::HolovizLayout;
using holoscan::radar::app::gui_available;
using holoscan::radar::app::make_color_holoviz_op;
using holoscan::radar::app::make_grayscale_lut;
using holoscan::radar::app::make_image_holoviz_op;

class SarFormationApp : public holoscan::Application {
 public:
  AppConfig cfg;

  void compose() override {
    const auto& origin = cfg.sample_geometry.origin();
    const auto& axis_v = cfg.sample_geometry.axis_v();
    const auto& axis_u = cfg.sample_geometry.axis_u();
    HOLOSCAN_LOG_INFO(
        "sar_formation: {} collection(s)  lattice {}x{}  "
        "f0={:.6e} Hz  dump {}",
        cfg.cphd_paths.size(), cfg.sample_geometry.num_v(),
        cfg.sample_geometry.num_u(), cfg.reference_frequency,
        cfg.dump ? cfg.dump_path : std::string("off"));
    if (cfg.lattice_max_side > 0) {
      if (cfg.scroll) {
        HOLOSCAN_LOG_INFO(
            "sar_formation lattice.max_side {} (scroll window "
            "depth along v)",
            cfg.lattice_max_side);
      } else if (cfg.parent_geometry.is_valid()) {
        HOLOSCAN_LOG_INFO(
            "sar_formation lattice.max_side {} (per-tile "
            "formed-pixel cap)",
            cfg.lattice_max_side);
      } else if (cfg.show_playback_lag) {
        HOLOSCAN_LOG_INFO(
            "sar_formation lattice.max_side {} (formed-pixel cap; "
            "lower this if playback lag climbs)",
            cfg.lattice_max_side);
      } else {
        HOLOSCAN_LOG_INFO(
            "sar_formation lattice.max_side {} (formed-pixel cap)",
            cfg.lattice_max_side);
      }
    }
    HOLOSCAN_LOG_INFO(
        "sar_formation lattice origin [{:.3f}, {:.3f}, {:.3f}]",
        origin[0], origin[1], origin[2]);
    HOLOSCAN_LOG_INFO(
        "sar_formation axis_v [{:.6f}, {:.6f}, {:.6f}]",
        axis_v[0], axis_v[1], axis_v[2]);
    HOLOSCAN_LOG_INFO(
        "sar_formation axis_u [{:.6f}, {:.6f}, {:.6f}]",
        axis_u[0], axis_u[1], axis_u[2]);

    // Spaceborne. Atlanta's lattice is scene ENU (near the origin)
    // but the APC is ~770 km out; float32 ULP there is ~9 cm.
    // The graph stays double. TESTING.md: fp32 is degraded at
    // spaceborne ranges.
    HOLOSCAN_LOG_INFO("sar_formation precision double");
    if (cfg.nfft > 0) {
      HOLOSCAN_LOG_INFO(
          "sar_formation range_bin_spacing {:.9f} m (CPHD SCSS, "
          "nfft={}; range_bins unconnected)",
          cfg.range_bin_spacing, cfg.nfft);
    } else {
      HOLOSCAN_LOG_INFO(
          "sar_formation range_bin_spacing {:.9f} m (CPHD SCSS; "
          "range_bins unconnected)",
          cfg.range_bin_spacing);
    }
    if (cfg.tile_geometries.size() <= 1) {
      for (size_t i = 0; i < cfg.collection_geometries.size();
           ++i) {
        const auto& geom = cfg.collection_geometries[i];
        const auto& sp = geom.spacing();
        HOLOSCAN_LOG_INFO(
            "sar_formation collection {} / {}: {}x{}  "
            "spacing {:.4f} x {:.4f} m  pixel_scale {:.4g}",
            i + 1, cfg.collection_geometries.size(), geom.num_v(),
            geom.num_u(), sp[0], sp[1],
            i < cfg.lattice_pixel_scale.size()
                ? cfg.lattice_pixel_scale[i]
                : 1.0);
        if (i < cfg.collection_physics.size()) {
          const auto& phys = cfg.collection_physics[i];
          HOLOSCAN_LOG_INFO(
              "sar_formation collection {} physics: heading "
              "{:.1f} deg CW from N  XY square {:.2f} km  "
              "GSD {:.3f} m (range {:.3f}, azimuth {:.3f})  "
              "native {}x{}",
              i + 1,
              phys.heading_cw_rad * 180.0 / radar::kPi,
              phys.xy_square_m / 1000.0, phys.gsd_m,
              phys.gsd_range_m, phys.gsd_azimuth_m,
              phys.native_count, phys.native_count);
        }
      }
    }
    for (size_t i = 0; i < cfg.tile_geometries.size(); ++i) {
      const auto& geom = cfg.tile_geometries[i];
      const auto& origin = geom.origin();
      const auto& sp = geom.spacing();
      HOLOSCAN_LOG_INFO(
          "sar_formation tile {} / {}: {}x{}  "
          "spacing {:.4f} m  origin [{:.3f}, {:.3f}, {:.3f}]",
          i + 1, cfg.tile_geometries.size(), geom.num_v(),
          geom.num_u(), sp[0], origin[0], origin[1], origin[2]);
    }
    if (cfg.parent_geometry.is_valid()) {
      const auto& p = cfg.parent_geometry;
      const auto& sp = p.spacing();
      HOLOSCAN_LOG_INFO(
          "sar_formation parent {}x{}  spacing {:.4f} x {:.4f} m",
          p.num_v(), p.num_u(), sp[0], sp[1]);
    } else if (cfg.scroll && cfg.scroll_parent.is_valid()) {
      const auto& p = cfg.scroll_parent;
      const auto& sp = p.spacing();
      HOLOSCAN_LOG_INFO(
          "sar_formation parent {}x{}  spacing {:.4f} x {:.4f} m",
          p.num_v(), p.num_u(), sp[0], sp[1]);
    }
    auto reader = make_operator<radar::CPHDReaderOp<double>>(
        "cphd_reader",
        from_config("cphd_reader"),
        holoscan::Arg("path") =
            radar::CollectionPaths(cfg.cphd_paths));

    std::shared_ptr<radar::SARRangeCompressionOp<double>> range;
    if (cfg.nfft > 0) {
      range = make_operator<radar::SARRangeCompressionOp<double>>(
          "range_compression", from_config("range_compression"),
          holoscan::Arg("nfft") = cfg.nfft);
    } else {
      range = make_operator<radar::SARRangeCompressionOp<double>>(
          "range_compression", from_config("range_compression"));
    }

    // Lattice and stored-sample-N/2 frequency: sidecar when present
    // (generated fixtures), otherwise the app-owned YAML keys
    // (Atlanta and any other public CPHD).
    holoscan::ArgList manager_args = from_config("aperture_manager");
    if (cfg.tile_pulses > 0) {
      manager_args.add(holoscan::Arg("emit_interval_pulses") =
                           cfg.tile_pulses);
    }
    int64_t scroll_slice = 0;
    int64_t scroll_cap = 1;
    if (cfg.scroll && cfg.emit_interval_pulses > 0 &&
        cfg.reader_pulses_per_tick > cfg.emit_interval_pulses) {
      scroll_slice = cfg.emit_interval_pulses;
      scroll_cap =
          (cfg.reader_pulses_per_tick + scroll_slice - 1) /
          scroll_slice;
    }
    auto manager = make_operator<radar::ApertureManagerOp<double>>(
        "aperture_manager", manager_args);
    manager->input_queue_capacity = scroll_cap;

    auto bp = make_operator<radar::BackprojectionOp<double>>(
        "backprojection",
        from_config("backprojection"),
        holoscan::Arg("sample_geometry") = cfg.sample_geometry,
        holoscan::Arg("reference_frequency") =
            cfg.reference_frequency,
        holoscan::Arg("range_bin_spacing") = cfg.range_bin_spacing);
    bp->input_queue_capacity = scroll_cap;

    add_flow(reader, range, {{"phase_history", "phase_history"}});

    std::shared_ptr<ScrollGeometryStagerOp> scroll_stager;
    std::shared_ptr<holoscan::Operator> profiles = range;
    if (!cfg.nvradar_vph_path.empty()) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: NVRadar vph override {}",
          cfg.nvradar_vph_path);
      auto vph = make_operator<NvradarVphOverrideOp>(
          "nvradar_vph",
          holoscan::Arg("path") = cfg.nvradar_vph_path);
      add_flow(range, vph, {{"range_profiles", "range_profiles"}});
      profiles = vph;
    }
    if (cfg.collection_geometries.size() > 1) {
      auto stager =
          make_operator<CollectionBoundaryStagerOp>("stager");
      stager->geometries = cfg.collection_geometries;
      stager->spacings = cfg.collection_spacings;
      stager->frequencies = cfg.collection_frequencies;
      add_flow(profiles, stager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, manager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, manager,
               {{"sample_geometry", "sample_geometry"}});
      add_flow(stager, bp,
               {{"range_bin_spacing", "range_bin_spacing"}});
      add_flow(stager, bp,
               {{"reference_frequency", "reference_frequency"}});
      add_flow(manager, bp,
               {{"sample_geometry", "sample_geometry"}});
    } else if (cfg.tile_geometries.size() > 1) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: tiling {} tiles of {} pulses "
          "(live sample_geometry on cadence reset)",
          cfg.tile_geometries.size(), cfg.tile_pulses);
      auto stager =
          make_operator<TileGeometryStagerOp>("tile_stager");
      stager->geometries = cfg.tile_geometries;
      stager->tile_pulses = cfg.tile_pulses;
      stager->loop = cfg.reader_loop;
      add_flow(profiles, stager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, manager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, manager,
               {{"sample_geometry", "sample_geometry"}});
      add_flow(manager, bp,
               {{"sample_geometry", "sample_geometry"}});
    } else if (cfg.scroll) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: scrolling {}x{} window on {}x{} parent "
          "(anchor row {}, live sample_geometry along v)",
          cfg.sample_geometry.num_v(), cfg.sample_geometry.num_u(),
          cfg.scroll_parent.num_v(), cfg.scroll_parent.num_u(),
          cfg.scroll_anchor);
      auto stager =
          make_operator<ScrollGeometryStagerOp>("scroll_stager");
      stager->parent = cfg.scroll_parent;
      stager->window0 = cfg.sample_geometry;
      stager->anchor = cfg.scroll_anchor;
      stager->loop = cfg.reader_loop;
      if (scroll_slice > 0) {
        HOLOSCAN_LOG_INFO(
            "sar_formation: scroll stager slicing {}-pulse "
            "reader ticks into {}-pulse dispatches (up to {} "
            "per tick)",
            cfg.reader_pulses_per_tick, scroll_slice, scroll_cap);
      }
      stager->slice_pulses = scroll_slice;
      stager->dispatch_capacity = scroll_cap;
      add_flow(profiles, stager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, manager,
               {{"range_profiles", "range_profiles"}});
      add_flow(stager, bp,
               {{"sample_geometry", "sample_geometry"}});
      scroll_stager = stager;
    } else {
      add_flow(profiles, manager,
               {{"range_profiles", "range_profiles"}});
    }

    std::shared_ptr<holoscan::Operator> geom_src = reader;
    if (cfg.transform_to_scene_enu) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: ECEF→ENU at IARP for NVRadar grid");
      auto to_enu = make_operator<EcefToSceneEnuOp>(
          "ecef_to_enu",
          holoscan::Arg("origin_ecef") = std::vector<double>{
              cfg.enu_origin_ecef[0], cfg.enu_origin_ecef[1],
              cfg.enu_origin_ecef[2]},
          holoscan::Arg("quantize_phase_center_f32") =
              cfg.quantize_phase_center_f32);
      add_flow(reader, to_enu, {{"positions", "positions"}});
      add_flow(reader, to_enu,
               {{"reference_position", "reference_position"}});
      geom_src = to_enu;
    }
    if (scroll_stager) {
      add_flow(geom_src, scroll_stager, {{"positions", "positions"}});
      add_flow(geom_src, scroll_stager,
               {{"reference_position", "reference_position"}});
      add_flow(scroll_stager, manager, {{"positions", "positions"}});
      add_flow(scroll_stager, manager,
               {{"reference_position", "reference_position"}});
    } else {
      add_flow(geom_src, manager, {{"positions", "positions"}});
      add_flow(geom_src, manager,
               {{"reference_position", "reference_position"}});
    }
    add_flow(manager, bp, {{"range_profiles", "range_profiles"}});
    add_flow(manager, bp, {{"positions", "positions"}});
    add_flow(manager, bp,
             {{"reference_position", "reference_position"}});
    add_flow(manager, bp, {{"control", "control"}});
    if (cfg.wire_weight) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: wiring ApertureManagerOp.weight");
      add_flow(manager, bp, {{"weight", "weight"}});
    }
    std::shared_ptr<holoscan::Operator> formed = bp;
    if (cfg.parent_geometry.is_valid() &&
        cfg.tile_geometries.size() > 1) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: TileMosaicOp parent hold ({} tiles)",
          cfg.tile_geometries.size());
      auto mosaic = make_operator<TileMosaicOp>("tile_mosaic");
      mosaic->parent = cfg.parent_geometry;
      mosaic->tiles = cfg.tile_geometries;
      mosaic->loop = cfg.reader_loop;
      add_flow(bp, mosaic, {{"image", "image"}});
      formed = mosaic;
    } else if (cfg.scroll) {
      HOLOSCAN_LOG_INFO(
          "sar_formation: ScrollMosaicOp parent hold ({}x{})",
          cfg.scroll_parent.num_v(), cfg.scroll_parent.num_u());
      auto mosaic = make_operator<ScrollMosaicOp>("scroll_mosaic");
      mosaic->parent = cfg.scroll_parent;
      mosaic->loop = cfg.reader_loop;
      add_flow(bp, mosaic, {{"image", "image"}});
      add_flow(bp, mosaic,
               {{"sample_geometry", "sample_geometry"}});
      formed = mosaic;
    }
    if (cfg.dump) {
      holoscan::ArgList dump_args{
          holoscan::Arg("path") = cfg.dump_path};
      if (cfg.dump_max_files >= 0) {
        dump_args.add(
            holoscan::Arg("max_files") = cfg.dump_max_files);
      }
      auto dump = make_operator<radar::TensorDumpOp>(
          "tensor_dump", dump_args);
      add_flow(formed, dump, {{"image", "tensor"}});
    }

    if (cfg.visualize) {
      auto to_lut = make_operator<SarImageToLutOp>("sar_image_to_lut");
      // Holoviz window size, not the LUT input lattice. R8_UINT
      // cannot bilinear; box-average the LUT toward this side
      // so the window is near 1:1. BP stays at the formed
      // lattice (tile or scroll window); dump follows `formed`
      // (parent canvas when tiling or scrolling).
      const bool show_lag = cfg.show_playback_lag;
      const bool mosaic = cfg.parent_geometry.is_valid();
      const bool strip_fit = mosaic || cfg.scroll;
      const radar::SampleGeometry viz_geom =
          mosaic ? cfg.parent_geometry
                 : (cfg.scroll ? cfg.scroll_parent
                               : cfg.sample_geometry);
      uint32_t viz_w = 0;
      uint32_t viz_h = 0;
      int64_t lut_max_side = cfg.visualize_size;
      const int64_t viz_u =
          (cfg.viz_crop_nu > 0) ? cfg.viz_crop_nu
                                : viz_geom.num_u();
      if (strip_fit) {
        int work_w = 0;
        int work_h = 0;
        if (desktop_work_area(&work_w, &work_h)) {
          const auto fit = holoviz_fit_wh(
              viz_geom.num_v(), viz_u, work_w, work_h);
          viz_w = fit[0];
          viz_h = fit[1];
          lut_max_side = static_cast<int64_t>(
              std::max(viz_w, viz_h));
          HOLOSCAN_LOG_INFO(
              "sar_formation Holoviz {}x{} (desktop work area "
              "{}x{}; visualize_size {} is the no-DISPLAY "
              "fallback)",
              viz_w, viz_h, work_w, work_h, cfg.visualize_size);
        } else {
          const auto wh = holoviz_window_wh(
              viz_geom.num_v(), viz_u, cfg.visualize_size);
          viz_w = wh[0];
          viz_h = wh[1];
          HOLOSCAN_LOG_INFO(
              "sar_formation Holoviz {}x{} (visualize_size {}; "
              "no desktop work area)",
              viz_w, viz_h, cfg.visualize_size);
        }
      }
      to_lut->max_side = lut_max_side;
      to_lut->crop_u0 = cfg.viz_crop_u0;
      to_lut->crop_nu = cfg.viz_crop_nu;
      if (cfg.viz_crop_nu > 0) {
        HOLOSCAN_LOG_INFO(
            "sar_formation Holoviz crop u [{}, {}) of {} "
            "(range-window intersection)",
            cfg.viz_crop_u0, cfg.viz_crop_u0 + cfg.viz_crop_nu,
            viz_geom.num_u());
      }
      to_lut->scene_names = cfg.scene_names;
      to_lut->formation_diag = [bp, reader, show_lag, mosaic,
                                scroll = cfg.scroll,
                                parent = cfg.parent_geometry,
                                scroll_parent = cfg.scroll_parent,
                                geoms = cfg.collection_geometries,
                                to_enu = cfg.transform_to_scene_enu,
                                origin = cfg.enu_origin_ecef]() {
        const auto t = bp->last_image_timing();
        SarImageToLutOp::FormationDiag d;
        d.bp_compute_s = t.compute_s;
        d.pulses = t.pulses;
        d.pixels = t.pixels;
        d.bp_num_v = t.num_v;
        d.bp_num_u = t.num_u;
        d.emit_at = t.emit_at;
        d.last_packet_at = reader->last_packet_at();
        if (show_lag) {
          d.playback = reader->playback_status();
        }
        const auto apc = reader->last_mean_apc();
        if (apc.valid) {
          const radar::SampleGeometry* g = nullptr;
          if (mosaic && parent.is_valid()) {
            g = &parent;
          } else if (scroll && scroll_parent.is_valid()) {
            g = &scroll_parent;
          } else if (!geoms.empty()) {
            const size_t i =
                static_cast<size_t>(apc.file_index) %
                geoms.size();
            g = &geoms[i];
          }
          if (g != nullptr) {
            const auto uv = apc_raster_uv(
                apc.ecef, *g, to_enu, origin);
            const auto en = apc_compass_enu(
                apc.ecef, *g, to_enu, origin);
            if (uv.has_value()) {
              d.apc_valid = true;
              d.raster_u = (*uv)[0];
              d.raster_v = (*uv)[1];
            }
            if (en.has_value()) {
              d.apc_valid = true;
              d.compass_east = (*en)[0];
              d.compass_north = (*en)[1];
            }
          }
        }
        return d;
      };
      auto hold = make_operator<HoldLastImageOp>(
          "hold_image",
          make_condition<holoscan::PeriodicCondition>(
              "hold_period", static_cast<int64_t>(100000000)));
      hold->hold_s = cfg.visualize_hold_s;

      HolovizLayout layout;
      layout.window_title = "SAR formation";
      if (strip_fit) {
        layout.width = viz_w;
        layout.height = viz_h;
      } else {
        const auto side =
            static_cast<uint32_t>(cfg.visualize_size);
        layout.width = side;
        layout.height = side;
      }
      layout.headless = cfg.headless;
      std::shared_ptr<holoscan::ops::HolovizOp> visualizer;
      if (strip_fit) {
        visualizer = make_image_holoviz_op(
            this, "holoviz", layout, "sar_image",
            make_grayscale_lut(),
            [title = layout.window_title](int, int) {
              spawn_place_window_top_left(title);
            });
      } else {
        visualizer = make_image_holoviz_op(
            this, "holoviz", layout, "sar_image",
            make_grayscale_lut());
      }

      auto timeline = make_operator<TimelineRasterOp>(
          "timeline",
          make_condition<holoscan::PeriodicCondition>(
              "timeline_period",
              static_cast<int64_t>(100000000)));
      timeline->scene_names = cfg.scene_names;
      timeline->spans = [reader]() {
        return reader->timeline_spans();
      };
      timeline->last_image_emit = [bp]() {
        return bp->last_image_timing().emit_at;
      };
      timeline->last_packet_at = [reader]() {
        return reader->last_packet_at();
      };
      if (show_lag) {
        timeline->playback = [reader]() {
          return reader->playback_status();
        };
      }
      HolovizLayout tl_layout;
      tl_layout.window_title = "SAR timeline";
      tl_layout.width = static_cast<uint32_t>(timeline->width);
      tl_layout.height = static_cast<uint32_t>(timeline->height);
      tl_layout.headless = cfg.headless;
      auto timeline_viz = make_color_holoviz_op(
          this, "timeline_holoviz", tl_layout, "timeline",
          [title = tl_layout.window_title](int, int) {
            spawn_place_window_bottom_left(title);
          });

      add_flow(formed, to_lut, {{"image", "image"}});
      add_flow(to_lut, hold, {{"sar_image", "image"}});
      add_flow(hold, visualizer, {{"image", "receivers"}});
      add_flow(timeline, timeline_viz,
               {{"timeline", "receivers"}});

      // One GLFW thread for both windows. EventBased
      // strict-pins a user pool; pin_operator false so Holoviz
      // ops share the single worker.
      auto viz_pool = make_thread_pool("viz", 1);
      viz_pool->add(visualizer, false);
      viz_pool->add(timeline_viz, false);
    }

    if (formed != bp && !cfg.dump && !cfg.visualize) {
      auto drain = make_operator<DiscardImageOp>("mosaic_drain");
      add_flow(formed, drain, {{"image", "image"}});
    }

    // Shared default pool, not a pinned thread per operator.
    // EventBasedScheduler strict-pins user ThreadPools: an idle
    // reader thread then cannot run BP. A pool of size N (N =
    // operator count) is enough that a ready op never waits on a
    // busy worker; extra threads sleep. Greedy (the SDK default)
    // cannot overlap reader.compute() with BP.compute().
    const auto n_ops = graph().get_nodes().size();
    const int64_t n_workers =
        n_ops > 0 ? static_cast<int64_t>(n_ops) : int64_t{2};
    scheduler(make_scheduler<holoscan::EventBasedScheduler>(
        "event_based",
        holoscan::Arg("worker_thread_number") = n_workers));
    HOLOSCAN_LOG_INFO(
        "sar_formation: EventBasedScheduler, {} workers "
        "(one per operator, shared pool)",
        n_workers);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<SarFormationApp>();
  try {
    const CliPaths cli = parse_cli(argc, argv, "mode01.yaml");
    app->cfg = load_config(cli.config, cli.cphd_override);
    if (app->cfg.visualize && !app->cfg.headless &&
        !app->cfg.skip_display_check && !gui_available()) {
      HOLOSCAN_LOG_WARN(
          "sar_formation: visualize is enabled but no usable "
          "display was detected (Holoviz could not initialize a "
          "window); running without the visualization window");
      app->cfg.visualize = false;
    }
    // Same file feeds from_config() for the operator blocks.
    app->config(cli.config.string());
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("sar_formation: {}", e.what());
    return 1;
  }
  return 0;
}
