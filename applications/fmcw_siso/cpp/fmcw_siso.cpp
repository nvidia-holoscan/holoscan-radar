// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example: a synthetic ADC source feeding the split
// signal-processing chain
//
//   SimAdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> CFARDetectorOp
//                                        -> DetectionReportOp
//                                        -> FeatureAssemblyOp
//                                             -> ClusterDBSCANOp
//                                                  -> DetectionReductionOp
//                                                       -> ClusterReportOp
//                                         [-> RdMapToImageOp   -> HolovizOp]
//                                         [-> ClusterOverlayOp -> HolovizOp]
//                                         [-> RollingPlotOp    -> HolovizOp]
//                                              (RD map + cluster crosses left;
//                                               range/velocity-vs-time right)
//
// The source synthesizes one static clutter return and one moving target with
// the shared analytic point-target simulator (radar_common/sim), emitting one
// cube (CPI) per tick and advancing the targets between ticks. CFAR detects on
// the range-Doppler map; the sinks report each detection / cluster in physical
// units via the operators' range_bins / velocity_bins axis outputs.
//
// The app scaffolding — config-path resolution, the app-owned config parse, the
// synthetic source, the display-probe downgrade, the run driver, and the
// Holoviz layout — is shared in radar_app/app and radar_app/viz; this file is
// the graph wiring plus its own AppConfig. The per-operator blocks
// (range_processor, clutter_filter, doppler_processor, cfar_detector,
// cluster_dbscan, reduction) flow through Holoscan's native from_config().
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_siso/cpp/fmcw_siso [config.yaml]

#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

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

// The shared app scaffolding (SimAdcSourceOp, run_fmcw_app, the Holoviz layout
// builder) and the reusable report / visualization operators live under
// holoscan::radar::app; this app pulls them in unqualified and only wires them
// into its graph. The app-owned AppConfig + load_config live in config.hpp.
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

class FmcwSisoApp : public holoscan::Application {
 public:
  AppConfig cfg;  // set before run()

  void compose() override {
    // Emit cfg.num_frames cubes, one every cfg.frame_period seconds, so the
    // targets are seen moving through space over the run.
    auto source = make_operator<SimAdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count", cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    // Operator parameters come straight from the same-named YAML blocks; the
    // yaml_converters.hpp specializations decode the custom types.
    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"));
    auto sink = make_operator<DetectionReportOp>("detection_report");
    // Every CFAR output needs a receiver or the operator never ticks.
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");

    // Clustering branch: assemble a [range, velocity] feature matrix from the
    // CFAR cells (physical units, via the range/Doppler axis outputs) and
    // density-cluster it. `features` is fixed in code — the parameter is
    // native-only by operator contract, and this app's downstream report /
    // overlays assume this exact 2-column layout. The DBSCAN parameters flow
    // natively from the cluster_dbscan YAML block.
    const std::vector<radar::AssemblyFeature> feature_columns{
        radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity};
    auto feature_assembly = make_operator<radar::FeatureAssemblyOp<float>>(
        "feature_assembly", holoscan::Arg("features") = feature_columns);
    auto cluster = make_operator<radar::ClusterDBSCANOp<float>>(
        "cluster_dbscan", from_config("cluster_dbscan"));
    auto cluster_report = make_operator<ClusterReportOp>("cluster_report");
    // Reduce each cluster to one representative cell. The reduction block flows
    // natively; set feature_weights ~ 1/epsilon in the YAML so the medoid
    // distance matches the DBSCAN metric.
    auto reduction = make_operator<radar::DetectionReductionOp<float>>(
        "detection_reduction", from_config("reduction"));

    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
    add_flow(doppler, cfar, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, sink, {{"detection_idx", "detection_idx"}});
    add_flow(cfar, sink, {{"detection_score", "detection_score"}});
    add_flow(range, sink, {{"range_bins", "range_bins"}});
    add_flow(doppler, sink, {{"velocity_bins", "velocity_bins"}});
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // CFAR cells → feature matrix → DBSCAN → report. The range/Doppler axis
    // outputs feed FeatureAssemblyOp so the columns land in physical units; its
    // `points` output fans out to both the clusterer and the reporter.
    add_flow(cfar, feature_assembly, {{"detection_idx", "detection_idx"}});
    add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
    add_flow(doppler, feature_assembly, {{"velocity_bins", "velocity_bins"}});
    add_flow(feature_assembly, cluster, {{"points", "points"}});
    add_flow(feature_assembly, cluster_report, {{"points", "points"}});
    add_flow(cluster, cluster_report, {{"labels", "labels"}});
    add_flow(cluster, cluster_report, {{"sizes", "sizes"}});

    // Reduction: one representative cell per cluster (medoid = most-central
    // detection). rep_idx feeds the report (logged in physical units, so the
    // report also needs the axis bins) and, when visualizing, the GUI.
    add_flow(cluster, reduction, {{"labels", "labels"}});
    add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
    add_flow(feature_assembly, reduction, {{"points", "points"}});
    add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
    add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
    add_flow(doppler, cluster_report, {{"velocity_bins", "velocity_bins"}});

    // Optional live RD-map window (cfg.visualize; the driver has already
    // downgraded it with a warning on GUI-less hosts). Turbo-mapped |RD| with
    // the CFAR detections overlaid as crosses colored by DBSCAN cluster (noise
    // in gray). With cfg.plots.enable, two rolling time-series plots (range and
    // velocity vs time) stack on the right; radar_app owns the layout, this app
    // wires the flows.
    if (cfg.visualize) {
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = "FMCW SISO - Range-Doppler + clusters";
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = 600U;
      layout.headless = cfg.headless;
      layout.plots_enabled = cfg.plots.enable;
      if (cfg.plots.enable) {
        layout.right_panels = range_velocity_panels();
      }
      auto visualizer = make_rd_holoviz_op(this, "holoviz", layout);

      std::shared_ptr<RollingPlotOp> plot;
      if (cfg.plots.enable) {
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
      }

      add_flow(doppler, to_image, {{"rd_maps", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // The overlay needs the detector's cells, the cluster labels, and the map
      // dimensions; its per-color layers fan into the same receiver.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(doppler, overlay, {{"rd_maps", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});

      // Rolling range/velocity plots: the per-cluster representative from
      // reduction, plus the axis bins that fix each plot's Y extent and resolve
      // the representative bins to physical units.
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
        add_flow(range, plot, {{"range_bins", "range_bins"}});
        add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        add_flow(plot, visualizer, {{"plots", "receivers"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwSisoApp>(argc, argv, "fmcw_siso", "fmcw_siso.yaml",
                                   load_config);
}
