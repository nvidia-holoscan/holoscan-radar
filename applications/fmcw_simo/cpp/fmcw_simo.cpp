// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIMO FMCW sparse angle-then-cluster pipeline:
//
//   SimAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
//     -> SnapshotGather -> BeamscanEstimator -> FeatureAssembly [r,v,az,el]
//     -> ClusterDBSCAN -> DetectionReduction -> ClusterReportOp
//
// The app scaffolding — config parse, the synthetic source, the run driver, and
// the Holoviz layout — is shared in radar_app/app; this file is the graph
// wiring plus its own AppConfig. Operator parameters flow through Holoscan's
// native from_config().
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_simo/cpp/fmcw_simo [config.yaml]

#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
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
#include "config.hpp"  // NOLINT(build/include_subdir): colocated app header

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)


class FmcwSimoApp : public holoscan::Application {
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
    // Derive the CFAR NCI fold (K) from the same array geometry beamscan
    // consumes, so it never drifts from the channel count. nci_axes: [0]
    // integrates the full channel axis, so K = num_elements; the operator
    // ignores this hint when nci_axes is empty (K = 1).
    // peak_only is a native cfar_detector param (from_config); the app reads it
    // (cfg.peak_only) to drop the clustering subgraph below.
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"),
        holoscan::Arg("nci_fold") = cfg.array_geometry.num_elements());
    auto gather = make_operator<radar::SnapshotGatherOp<float>>(
        "snapshot_gather", from_config("snapshot_gather"));
    auto beamscan = make_operator<radar::BeamscanEstimatorOp<float>>(
        "beamscan_estimator", from_config("beamscan_estimator"),
        holoscan::Arg("array_geometry") = cfg.array_geometry);

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
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, cfar, {{"rd_maps", "rd_maps"}});

    // Angle branch runs in both modes (it is the DOA estimator, not clustering).
    add_flow(doppler, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});
    add_flow(gather, beamscan,
             {{"snapshot", "snapshot"}, {"is_valid", "is_valid"}});
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

    add_flow(cfar, det_report,
             {{"detection_idx", "detection_idx"},
              {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // Clustering subgraph — dropped when peak_only is set. In that mode the
    // per-detection angle is reported through DetectionReportOp's optional
    // aoa_estimates input instead of via the (removed) cluster report.
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

      add_flow(beamscan, feature_assembly,
               {{"aoa_estimates", "aoa_estimates"}});
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
      add_flow(beamscan, det_report, {{"aoa_estimates", "aoa_estimates"}});
    }

    if (cfg.visualize) {
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = "FMCW SIMO - RD (ch0) + clusters";
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
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
      auto visualizer = make_rd_holoviz_op(this, "holoviz", layout);
      overlay->view_aspect = rd_map_view_aspect(layout);
      overlay->marker_radius = cfg.viz.marker_radius;
      overlay->marker_outline = cfg.viz.marker_outline;
      overlay->marker_segments = cfg.viz.marker_segments;

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

      add_flow(doppler, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // Overlay/plot/boresight color by cluster when clustering runs (keyed on
      // each cluster's representative RD cell, via rep_idx, so the color is
      // stable across frames); in peak_only mode there are no labels, so all
      // three color each peak by its per-detection row index instead.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      if (!cfg.peak_only) {
        add_flow(cluster, overlay, {{"labels", "labels"}});
        add_flow(reduction, overlay, {{"rep_idx", "rep_idx"}});
      }
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
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
        add_flow(beamscan, boresight, {{"aoa_estimates", "aoa_estimates"}});
        if (!cfg.peak_only) {
          add_flow(cluster, boresight, {{"labels", "labels"}});
          add_flow(reduction, boresight, {{"rep_idx", "rep_idx"}});
        }
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
        // Primitive-panel InputSpec chain: rolling plot specs → boresight →
        // HolovizOp.input_specs (one connection into the single input_specs port).
        add_flow(plot, boresight, {{"plot_specs", "specs_in"}});
        add_flow(boresight, visualizer, {{"specs_out", "input_specs"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwSimoApp>(argc, argv, "fmcw_simo", "fmcw_simo.yaml",
                                   load_config);
}
