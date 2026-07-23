// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
class EstimatorLabelOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(EstimatorLabelOp)
  EstimatorLabelOp() = default;

  const EstimatorRouterOp* router{nullptr};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("rd_maps");
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
    using InputType = holoscan::ops::HolovizOp::InputType;
    InputSpec label(kLabelLayer, InputType::TEXT);
    label.text_ = {"Estimator: " + display_caption(name)};
    label.color_ = {1.0F, 0.95F, 0.2F, 1.0F};  // amber, reads over turbo
    label.priority_ = 10;                       // draw above the RD image
    // No View: render in full-window [0, 1] space so glyph advance isn't
    // squished by an anisotropic sub-view (which clipped the estimator name).
    std::vector<InputSpec> specs{label};
    op_output.emit(specs, "input_specs");

    // Text (x, y, size) in full-window [0, 1] space; size is the glyph height
    // as a fraction of the window height.
    const std::array<float, 3> pos = {0.012F, 0.03F, 0.03F};
    auto tensor = radar::make_device_tensor({1, 3}, radar::real_dtype<float>());
    if (cudaMemcpy(tensor->data(), pos.data(), sizeof(pos),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("EstimatorLabelOp: label position H2D failed");
    }
    holoscan::TensorMap out;
    out.insert({kLabelLayer, tensor});
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
    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
        radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly", holoscan::Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
    auto reduction = make_operator<radar::DetectionReductionOp<float>>(
        "detection_reduction", from_config("reduction"));

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
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

    // Each estimator feeds the same aoa_estimates input; only the selected one
    // emits per tick, so FeatureAssembly sees exactly one estimate each frame.
    add_flow(beamscan, feature_assembly, {{"aoa_estimates", "aoa_estimates"}});
    add_flow(mvdr, feature_assembly, {{"aoa_estimates", "aoa_estimates"}});
    add_flow(music, feature_assembly, {{"aoa_estimates", "aoa_estimates"}});
    add_flow(root_music, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
    add_flow(esprit, feature_assembly, {{"aoa_estimates", "aoa_estimates"}});
    // Beamscan's angle_spectrum is a required output; drain it.
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

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

    if (cfg.visualize) {
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = "FMCW AoA estimators - RD (ch0) + clusters";
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
      layout.plots_enabled = cfg.plots.enable;
      if (cfg.plots.enable) {
        layout.right_panels = range_velocity_boresight_panels();
      }
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

      // Live caption of the active estimator, drawn over the top of the RD map
      // (a dynamic TEXT layer appended to Holoviz's compose-time specs).
      auto label = make_operator<EstimatorLabelOp>("estimator_label");
      label->router = router.get();

      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
      }

      add_flow(doppler, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // The active-estimator caption: its dynamic spec + text position.
      add_flow(doppler, label, {{"rd_maps", "rd_maps"}});
      add_flow(label, visualizer, {{"input_specs", "input_specs"}});
      add_flow(label, visualizer, {{"label_pos", "receivers"}});
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
      if (boresight) {
        // Boresight az/el is fed by whichever estimator ran this frame.
        add_flow(beamscan, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(mvdr, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(music, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(root_music, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(esprit, boresight, {{"aoa_estimates", "aoa_estimates"}});
        add_flow(cluster, boresight, {{"labels", "labels"}});
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwAoaEstimatorsApp>(argc, argv, "fmcw_aoa_estimators",
                                            "fmcw_aoa_estimators.yaml",
                                            load_config);
}
