// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FMCW sparse angle-then-cluster pipeline that wires all five sparse AoA
// estimators (Beamscan, MVDR, MUSIC, Root-MUSIC, ESPRIT) into one graph and
// uses Holoscan dynamic flow control so only the selected estimator ticks each
// frame:
//
//   SimAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
//     -> SnapshotGather -> EstimatorRouterOp
//        --(dynamic: one path)-->  { SnapshotK1Slice -> Beamscan | MVDR |
//                                    MUSIC | Root-MUSIC | ESPRIT }
//     -> FeatureAssembly [r,v,az,el] -> ClusterDBSCAN -> DetectionReduction
//     -> ClusterReportOp
//
// The router owns the selection policy (rotate / hold / file); its per-tick
// choice is read by set_dynamic_flows() to activate exactly one estimator's
// edges. All five share only (snapshot[/is_valid] in, aoa_estimates out,
// array_geometry + angle_convention params); a filled y-z URA lets one geometry
// serve all five (native 2-D az+el). App scaffolding (config parse, source,
// driver, Holoviz
// layout) is shared in radar_app/app; this file is the graph plus AppConfig.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_aoa_estimators/cpp/fmcw_aoa_estimators \
//            [config.yaml]

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>     // NOLINT(build/include_order)
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gxf/multimedia/video.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "radar_common/tensor/dlpack_buffer.hpp"

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_app/app/driver.hpp"
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/app/source.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/viz.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir): colocated app header
#include "custom_ops.hpp"   // NOLINT(build/include_subdir): colocated app header

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

RouterMode parse_router_mode(const std::string& mode) {
  if (mode == "rotate") {
    return RouterMode::kRotate;
  }
  if (mode == "hold") {
    return RouterMode::kHold;
  }
  if (mode == "file") {
    return RouterMode::kFile;
  }
  throw std::runtime_error(
      "estimator_router.mode must be one of rotate | hold | file, got '" +
      mode + "'");
}

// Live estimator-name caption for the Holoviz overlay. Ticks once per frame off
// rd_maps and re-emits a single TEXT InputSpec (on input_specs) whose string is
// the router's currently selected estimator, plus that layer's (1, 3) text
// position tensor (on label_pos -> receivers). HolovizOp appends dynamic
// input_specs to the compose-time layers and matches the position tensor to the
// spec by name, so this draws a top-left caption that updates as the router
// advances. Reads the router's active_index directly — a one-frame lag versus
// the estimator that actually ran is immaterial for a caption.
//
// HolovizOp has a single input_specs port, so this op is the TAIL of the shared
// spec chain: the RD-map chrome (ClusterOverlayOp's scenario clock + axes) and
// the primitive side panels arrive on specs_in, and this op appends its caption
// before emitting the combined list.
class EstimatorLabelOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(EstimatorLabelOp)
  EstimatorLabelOp() = default;

  const EstimatorRouterOp* router{nullptr};
  // On-screen pixel height of the (full-window) view. Drives pixel-locked text
  // sizing so the title stays crisp against Holoviz's fixed 25 px font atlas,
  // exactly like the rest of the RD-map chrome. Set by the app.
  float view_px_height{600.0F};
  // Window aspect (width / height in pixels) for glyph-advance centering.
  float window_aspect{1.0F};
  // Where the RD-map title goes, in full-window coordinates: the horizontal
  // centre of the inset image box and the top of the title line. The app
  // derives both from the layout (see rd_map_region / kBox*) and blanks
  // ClusterOverlayOp::rd_title, so this op owns the whole title line.
  float title_center_x{0.5F};
  float title_top_y{0.0F};
  std::string base_title{"Range-Doppler"};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
    // Upstream chrome/panel specs (overlay → rolling plot → boresight → here).
    // Optional so the op still works when the viz graph has no panels.
    spec.input<std::vector<holoscan::ops::HolovizOp::InputSpec>>("specs_in")
        .condition(holoscan::ConditionType::kNone);
    spec.output<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
        "input_specs");
    spec.output<holoscan::TensorMap>("label_pos");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    op_input.receive<holoscan::TensorMap>("rd_maps");  // frame tick only
    if (router == nullptr) {
      throw std::runtime_error("EstimatorLabelOp: router not set");
    }
    const int64_t idx = router->active_index();
    const std::string name =
        (idx >= 0 && idx < static_cast<int64_t>(router->display_names.size()))
            ? router->display_names[static_cast<size_t>(idx)]
            : std::string("?");

    using InputSpec = holoscan::ops::HolovizOp::InputSpec;
    // The estimator name rides in the RD-map title rather than a separate
    // caption, so it reads as the heading of the plot it describes. Styled
    // exactly like the title ClusterOverlayOp would have drawn (white,
    // pixel-locked to kRdTitlePx, centred on the image box); the app blanks
    // rd_title so this is the only title on screen.
    const float sz = kRdTitlePx / std::max(view_px_height, 1.0F);
    holoscan::TensorMap out;
    std::vector<InputSpec> specs;
    if (auto upstream = op_input.receive<std::vector<InputSpec>>("specs_in")) {
      specs = std::move(upstream.value());
    }
    push_abs_text(out, &specs, kLabelLayer,
                  {{title_center_x, title_top_y,
                    base_title + " - " + display_caption(name),
                    TextAnchor::kCenter}},
                  sz, window_aspect, {1.0F, 1.0F, 1.0F, 1.0F}, 8);
    op_output.emit(specs, "input_specs");
    op_output.emit(out, "label_pos");
  }

 private:
  static constexpr const char* kLabelLayer = "estimator_label";

  // "root_music" -> "ROOT-MUSIC": uppercase, '_' -> '-'.
  static std::string display_caption(const std::string& name) {
    std::string s;
    s.reserve(name.size());
    for (char c : name) {
      s.push_back(c == '_' ? '-'
                           : static_cast<char>(std::toupper(
                                 static_cast<unsigned char>(c))));
    }
    return s;
  }
};

// Diagnostic: save Holoviz's render_buffer_output (an RGBA VideoBuffer) to
// P6 PPM files, one per frame, into out_dir. Wired only when AOA_CAPTURE_DIR is
// set (headless required so the framebuffer is produced). PPM keeps the op
// dependency-free; convert to PNG afterward. Alpha is dropped.
class FrameCaptureOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameCaptureOp)
  FrameCaptureOp() = default;

  std::string out_dir;
  int64_t max_frames{24};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("render");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe = op_input.receive<holoscan::gxf::Entity>("render");
    if (!maybe || maybe.value().is_null()) {
      return;
    }
    if (frame_ >= max_frames) {
      ++frame_;
      return;
    }
    auto vb = holoscan::gxf::get_videobuffer(maybe.value(), nullptr);
    if (!vb) {
      HOLOSCAN_LOG_WARN("FrameCaptureOp: no VideoBuffer on render port");
      return;
    }
    const auto info = vb->video_frame_info();
    const int64_t w = static_cast<int64_t>(info.width);
    const int64_t h = static_cast<int64_t>(info.height);
    if (info.color_format !=
        nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
      HOLOSCAN_LOG_WARN("FrameCaptureOp: expected RGBA framebuffer");
      return;
    }
    // Rows are stride-aligned (stride >= width*4), so copy the padded buffer
    // and index each row by its true pitch to avoid a diagonal shear.
    const int64_t stride =
        info.color_planes.empty()
            ? w * 4
            : static_cast<int64_t>(info.color_planes[0].stride);
    const size_t total = static_cast<size_t>(stride * h);
    std::vector<uint8_t> host(total);
    if (cudaMemcpy(host.data(), vb->pointer(), total, cudaMemcpyDefault) !=
        cudaSuccess) {
      HOLOSCAN_LOG_WARN("FrameCaptureOp: framebuffer D2H copy failed");
      return;
    }
    std::vector<uint8_t> rgb(static_cast<size_t>(w * h * 3));
    for (int64_t y = 0; y < h; ++y) {
      const uint8_t* row = host.data() + static_cast<size_t>(y * stride);
      uint8_t* dst = rgb.data() + static_cast<size_t>(y * w * 3);
      for (int64_t x = 0; x < w; ++x) {
        dst[x * 3 + 0] = row[x * 4 + 0];
        dst[x * 3 + 1] = row[x * 4 + 1];
        dst[x * 3 + 2] = row[x * 4 + 2];
      }
    }
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03ld.ppm",
                  static_cast<long>(frame_));  // NOLINT(runtime/int)
    const std::filesystem::path path =
        std::filesystem::path(out_dir) / name;
    std::ofstream os(path, std::ios::binary);
    os << "P6\n" << w << " " << h << "\n255\n";
    os.write(reinterpret_cast<const char*>(rgb.data()),
             static_cast<std::streamsize>(rgb.size()));
    ++frame_;
  }

 private:
  int64_t frame_{0};
};

}  // namespace

class FmcwAoaEstimatorsApp : public holoscan::Application {
 public:
  AppConfig cfg;
  std::shared_ptr<CadenceMeter> cadence;

  void compose() override {
    auto source = make_operator<SimAdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count", cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    // K for the CFAR NCI fold derives from the shared array geometry, so it
    // never drifts from the channel count (nci_axes: [0] folds the RX axis).
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"),
        holoscan::Arg("nci_fold") = cfg.array_geometry.num_elements());
    auto gather = make_operator<radar::SnapshotGatherOp<float>>(
        "snapshot_gather", from_config("snapshot_gather"));

    // The five interchangeable estimators. They share array_geometry (the URA)
    // and their angle_convention/grids come from their own YAML blocks.
    auto router = make_operator<EstimatorRouterOp>("estimator_router");
    auto slice = make_operator<SnapshotK1SliceOp>("snapshot_k1_slice");
    auto beamscan = make_operator<radar::BeamscanEstimatorOp<float>>(
        "beamscan_estimator", from_config("beamscan_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);
    auto mvdr = make_operator<radar::MVDREstimatorOp<float>>(
        "mvdr_estimator", from_config("mvdr_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);
    auto music = make_operator<radar::MUSICEstimatorOp<float>>(
        "music_estimator", from_config("music_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);
    auto root_music = make_operator<radar::RootMUSICEstimatorOp<float>>(
        "root_music_estimator", from_config("root_music_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);
    auto esprit = make_operator<radar::ESPRITEstimatorOp<float>>(
        "esprit_estimator", from_config("esprit_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);

    router->mode = parse_router_mode(cfg.router.mode);
    router->rotate_period = cfg.router.rotate_period;
    router->log_every = cfg.router.log_every;
    router->control_path = cfg.router.control_path;
    router->display_names = {"beamscan", "mvdr", "music", "root_music",
                             "esprit"};
    router->target_op_names = {"snapshot_k1_slice", "mvdr_estimator",
                               "music_estimator", "root_music_estimator",
                               "esprit_estimator"};
    // Resolve hold-mode index against the display names.
    router->hold_index = 0;
    for (size_t i = 0; i < router->display_names.size(); ++i) {
      if (router->display_names[i] == cfg.router.hold) {
        router->hold_index = static_cast<int64_t>(i);
      }
    }

    auto det_report = make_operator<DetectionReportOp>("detection_report");
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");
    auto spectrum_sink =
        make_operator<TensorMapDiscardOp>("angle_spectrum_discard");

    if (cfg.meter_cadence) {
      cadence = std::make_shared<CadenceMeter>();
      cadence->frame_period_s = cfg.frame_period;
      cadence->log_every = cfg.meter_log_every;
      source->cadence = cadence;
      det_report->cadence = cadence;
    }
    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles_out", "range_profiles"}});
    add_flow(doppler, cfar, {{"rd_maps", "rd_maps"}});

    add_flow(doppler, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});

    // Router fans out to every estimator branch statically; the dynamic-flow
    // callback below activates exactly one per tick.
    add_flow(gather, router,
             {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});
    const std::vector<std::shared_ptr<holoscan::Operator>> branch_heads{
        slice, mvdr, music, root_music, esprit};
    for (const auto& head : branch_heads) {
      add_flow(router, head,
               {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});
    }
    add_flow(slice, beamscan,
             {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});

    // Beamscan's angle_spectrum is a required output; drain it.
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});
    // Every estimator branch, for the fan-ins below. Only the router-selected
    // one emits per tick, so each downstream aoa_estimates input sees exactly
    // one estimate per frame.
    const std::vector<std::shared_ptr<holoscan::Operator>> estimators{
        beamscan, mvdr, music, root_music, esprit};

    // Activate only the selected estimator's edges each tick. compute() has
    // already advanced the policy and set active_target().
    set_dynamic_flows(
        router, [](const std::shared_ptr<holoscan::Operator>& op) {
          auto r = std::static_pointer_cast<EstimatorRouterOp>(op);
          const std::string& target = r->active_target();
          auto flows = op->find_all_flow_info(
              [&target](
                  const std::shared_ptr<holoscan::Operator::FlowInfo>& f) {
                return f->next_operator->name() == target;
              });
          op->add_dynamic_flow(flows);
        });

    add_flow(cfar, det_report,
             {{"detection_idx", "detection_idx"},
              {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // Clustering subgraph — dropped when peak_only is set. In that mode CFAR
    // already emits one detection per target, so the per-detection angle is
    // reported through DetectionReportOp's optional aoa_estimates input instead
    // of via the (removed) cluster report.
    std::shared_ptr<radar::ClusterDBSCANOp<float>> cluster;
    std::shared_ptr<radar::DetectionReductionOp<float>> reduction;
    if (!cfg.peak_only) {
      const std::vector<radar::AssemblyFeature> feature_columns{
          radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
          radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
      auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
          "feature_assembly", holoscan::Arg("features") = feature_columns);
      cluster = make_operator<radar::ClusterDBSCANOp<float>>(
          "cluster_dbscan", from_config("cluster_dbscan"));
      auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
      reduction = make_operator<radar::DetectionReductionOp<float>>(
          "detection_reduction", from_config("reduction"));

      for (const auto& est : estimators) {
        add_flow(est, feature_assembly, {{"aoa_estimates", "aoa_estimates"}});
      }
      add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
      add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
      add_flow(doppler, feature_assembly, {{"velocity_bins", "velocity_bins"}});
      add_flow(feature_assembly, cluster, {{"points", "points"}});
      add_flow(feature_assembly, cluster_report, {{"points", "points"}});
      add_flow(cluster, cluster_report,
               {{"labels", "labels"}, {"sizes", "sizes"}});
      add_flow(cluster, reduction, {{"labels", "labels"}});
      add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
      add_flow(feature_assembly, reduction, {{"points", "points"}});
      add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
      add_flow(cfar, cluster_report, {{"detection_idx", "detection_idx"}});
      add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
      add_flow(doppler, cluster_report, {{"velocity_bins", "velocity_bins"}});
    } else {
      for (const auto& est : estimators) {
        add_flow(est, det_report, {{"aoa_estimates", "aoa_estimates"}});
      }
    }

    if (cfg.visualize) {
      // The RD display is CFAR's own thresholded surface (cfar_power:
      // post-NCI integrated power here), so the picture and the overlaid
      // detections are literally the same data — no re-derivation that
      // could drift from the detector's nci_axes / nci_fold.
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = "FMCW AoA estimators - RD (NCI) + clusters";
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
      layout.vsync = cfg.viz.vsync;
      layout.plots_enabled = cfg.plots.enable;
      // Side panels (range/velocity/boresight) are drawn with native Holoviz
      // primitives by the plot ops (dynamic InputSpecs); no static right-panel
      // images. Stacked right-column view rects:
      const PanelRect range_view{0.63F, 0.02F, 0.35F, 0.30F};
      const PanelRect velocity_view{0.63F, 0.35F, 0.35F, 0.30F};
      const PanelRect boresight_view{0.63F, 0.68F, 0.35F, 0.30F};
      const auto panel_aspect = [&](const PanelRect& v) {
        return (v.w * static_cast<float>(layout.width)) /
               (v.h * static_cast<float>(layout.height));
      };
      // Diagnostic headless frame capture: AOA_CAPTURE_DIR enables Holoviz's
      // render_buffer_output and dumps composited frames to PPM (off by
      // default; requires headless so the buffer is produced).
      const char* capture_dir = std::getenv("AOA_CAPTURE_DIR");
      std::shared_ptr<holoscan::ops::HolovizOp> visualizer;
      if (capture_dir != nullptr) {
        visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            holoscan::Arg("tensors", build_holoviz_input_specs(layout)),
            holoscan::Arg("color_lut", make_turbo_lut()),
            holoscan::Arg("window_title", layout.window_title),
            holoscan::Arg("width", layout.width),
            holoscan::Arg("height", layout.height),
            holoscan::Arg("vsync", layout.vsync),
            holoscan::Arg("headless", layout.headless),
            holoscan::Arg("enable_render_buffer_output", true),
            holoscan::Arg("allocator",
                          make_resource<holoscan::UnboundedAllocator>(
                              "viz_alloc")));
        auto capture = make_operator<FrameCaptureOp>("frame_capture");
        capture->out_dir = capture_dir;
        std::filesystem::create_directories(capture_dir);
        add_flow(visualizer, capture,
                 {{"render_buffer_output", "render"}});
      } else {
        visualizer = make_rd_holoviz_op(this, "holoviz", layout);
      }

      overlay->view_aspect = rd_map_view_aspect(layout);
      overlay->rd_region = rd_map_region(layout);  // axes/frame/title region
      overlay->rd_view_px_height = static_cast<float>(layout.height);
      // EstimatorLabelOp draws the title (it carries the live estimator
      // name), so suppress the overlay's own.
      overlay->rd_title = "";
      overlay->frame_period = cfg.frame_period;    // scenario clock
      overlay->marker_radius = cfg.viz.marker_radius;
      overlay->marker_outline = cfg.viz.marker_outline;
      overlay->marker_segments = cfg.viz.marker_segments;

      // Live caption of the active estimator, drawn over the top of the RD map
      // (a dynamic TEXT layer appended to Holoviz's compose-time specs). It is
      // the tail of the InputSpec chain — see the wiring below.
      auto label = make_operator<EstimatorLabelOp>("estimator_label");
      label->router = router.get();
      label->view_px_height = static_cast<float>(layout.height);
      label->window_aspect = static_cast<float>(layout.width) /
                             static_cast<float>(layout.height);
      // Same anchor emit_rd_axes would have used: centred on the inset image
      // box, one title height above its top edge.
      const PanelRect rd_region = rd_map_region(layout);
      const float title_sz = kRdTitlePx / static_cast<float>(layout.height);
      label->title_center_x =
          rd_region.x + 0.5F * (kBoxL + kBoxR) * rd_region.w;
      label->title_top_y =
          rd_region.y + kBoxT * rd_region.h - title_sz - 0.012F;

      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        plot->primitive = true;
        plot->range_view = range_view;
        plot->velocity_view = velocity_view;
        plot->range_aspect = panel_aspect(range_view);
        plot->velocity_aspect = panel_aspect(velocity_view);
        plot->marker_radius = cfg.viz.marker_radius;
        plot->marker_outline = cfg.viz.marker_outline;
        plot->marker_segments = cfg.viz.marker_segments;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
        boresight->primitive = true;
        boresight->view = boresight_view;
        boresight->aspect = panel_aspect(boresight_view);
        boresight->marker_radius = cfg.viz.marker_radius;
        boresight->marker_outline = cfg.viz.marker_outline;
        boresight->marker_segments = cfg.viz.marker_segments;
      }

      add_flow(cfar, to_image, {{"cfar_power", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // The active-estimator caption: its text position rides `receivers`; its
      // spec closes the input_specs chain assembled below.
      add_flow(doppler, label, {{"rd_maps", "rd_maps"}});
      add_flow(label, visualizer, {{"input_specs", "input_specs"}});
      add_flow(label, visualizer, {{"label_pos", "receivers"}});
      // Overlay/plot/boresight color by cluster when clustering runs (keyed on
      // each cluster's representative RD cell, via rep_idx, so the color is
      // stable across frames); in peak_only mode there are no labels, so all
      // three color each peak by its per-detection row index instead.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cfar, overlay, {{"cfar_power", "rd_maps"}});
      // Physical axis bins for the RD-map tick labels (range X, velocity Y).
      add_flow(range, overlay, {{"range_bins", "range_bins"}});
      add_flow(doppler, overlay, {{"velocity_bins", "velocity_bins"}});
      if (!cfg.peak_only) {
        add_flow(cluster, overlay, {{"labels", "labels"}});
        add_flow(reduction, overlay, {{"rep_idx", "rep_idx"}});
      }
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      // HolovizOp has one input_specs port, so every dynamic-spec producer is
      // chained into a single stream that terminates at the caption:
      //   overlay → rolling plot → boresight → estimator label → input_specs
      // (without panels: overlay → estimator label → input_specs).
      if (plot) {
        add_flow(overlay, plot, {{"time_specs", "specs_in"}});
      } else {
        add_flow(overlay, label, {{"time_specs", "specs_in"}});
      }
      if (plot) {
        if (!cfg.peak_only) {
          add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
          // labels flag clustered mode; the plot then colors each rep by its RD
          // cell (matching the RD-map / boresight markers).
          add_flow(cluster, plot, {{"labels", "labels"}});
        } else {
          add_flow(cfar, plot, {{"detection_idx", "rep_idx"}});
        }
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
      if (boresight) {
        // Boresight az/el is fed by whichever estimator ran this frame.
        for (const auto& est : estimators) {
          add_flow(est, boresight, {{"aoa_estimates", "aoa_estimates"}});
        }
        if (!cfg.peak_only) {
          add_flow(cluster, boresight, {{"labels", "labels"}});
          add_flow(reduction, boresight, {{"rep_idx", "rep_idx"}});
        }
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
        add_flow(plot, boresight, {{"plot_specs", "specs_in"}});
        add_flow(boresight, label, {{"specs_out", "specs_in"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwAoaEstimatorsApp>(argc, argv, "fmcw_aoa_estimators",
                                            "fmcw_aoa_estimators.yaml",
                                            load_config);
}
