// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIMO FMCW *dense* angle-then-cluster pipeline (headless-capable, with an
// optional Holoviz display):
//
//   SimAdcSourceOp -> Range -> Clutter -> Doppler
//     -> PhaseShiftBeamformerOp (rd_maps -> rda_cube, angle on every RD cell)
//     -> CFARDetectorOp (on the RDA cube; nci_axes=[] -> az/el ride as batch)
//     -> AnglePeakPickOp (keep the peak (az,el) beam per detected RD cell)
//     -> CubeAoaLookupOp (bridge: detection az/el bins -> aoa_estimates)
//     -> FeatureAssembly [range, velocity, az, el] -> ClusterDBSCAN
//     -> DetectionReduction -> ClusterReportOp (logs clusters)
//
// This is a lean demo exercising PhaseShiftBeamformerOp end-to-end; it is kept
// separate from fmcw_simo (the sparse demo) because the dense path diverges
// in the detect+angle middle: CFAR runs per-beam on the RDA cube, so the
// detection_idx / rep_idx carry (az_bin, el_bin) batch columns. The shared
// ClusterReportOp handles the 4-column rep_idx directly (az/el from the
// beam grids passed as az_grid/el_grid), so no forked report op is needed. See
// the operator's OPEN_ITEMS for the FeatureAssembly az/el contract gap that the
// CubeAoaLookupOp bridge works around.
//
// The display reuses the app-agnostic radar_app/viz operators unchanged: the
// RD-map image + rolling range/velocity plots + boresight az/el scatter. The
// Doppler op still emits per-channel rd_maps (N_rx, D, R) exactly like the
// sparse path, so those ops apply directly; the only mismatch is that the dense
// detection_idx / rep_idx carry the (az_bin, el_bin) batch columns ahead of
// (doppler_bin, range_bin). A small TrailingIdxColumnsOp slices the trailing
// two columns so ClusterOverlayOp / RollingPlotOp see the sparse 2-column form
// they expect — no changes to the shared ops.
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo [config.yaml]

#include <cstdint>
#include <memory>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
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
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir): colocated app header
#include "custom_ops.hpp"  // NOLINT(build/include_subdir): colocated app header

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

class FmcwDenseSimoApp : public holoscan::Application {
 public:
  AppConfig cfg;
  std::shared_ptr<CadenceMeter> cadence;

  void compose() override {
    using holoscan::Arg;
    auto source = make_operator<SimAdcSourceOp>(
        "adc_source",
        make_condition<holoscan::CountCondition>("frame_count",
                                                 cfg.num_frames),
        make_condition<holoscan::PeriodicCondition>(
            "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
    source->scene = cfg.scene;
    source->frame_period = cfg.frame_period;

    auto range = make_operator<radar::FMCWRangeProcessorOp<Precision>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<Precision>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<Precision>>(
        "doppler_processor", from_config("doppler_processor"));
    auto beamformer = make_operator<radar::PhaseShiftBeamformerOp<Precision>>(
        "phase_shift_beamformer", from_config("phase_shift_beamformer"),
        Arg("array_geometry") = cfg.array_geometry);
    // peak_only is a native cfar_detector param (from_config), applied per-beam
    // (nci_axes=[]); AnglePeakPickOp below still does the cross-beam collapse.
    // The app reads cfg.peak_only to drop the DBSCAN clustering + reduction.
    auto cfar = make_operator<radar::CFARDetectorOp<Precision>>(
        "cfar_detector", from_config("cfar_detector"));

    auto peak_pick = make_operator<AnglePeakPickOp>("angle_peak_pick");
    auto aoa_lookup = make_operator<CubeAoaLookupOp>("cube_aoa_lookup");
    aoa_lookup->az_grid = cfg.az_angles;
    aoa_lookup->el_grid = cfg.el_angles;

    if (cfg.meter_cadence) {
      cadence = std::make_shared<CadenceMeter>();
      cadence->frame_period_s = cfg.frame_period;
      cadence->log_every = cfg.meter_log_every;
      source->cadence = cadence;
    }

    // Front end.
    add_flow(source, range, {{"adc_cube", "adc_cube"}});
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});

    // Dense angle path: beamform every RD cell, detect on the cube.
    add_flow(doppler, beamformer, {{"rd_maps", "rd_maps"}});
    add_flow(beamformer, cfar, {{"rda_cube", "rd_maps"}});

    // Collapse the per-beam cube detections to one peak beam per RD cell (dense
    // angle NMS — distinct from CFAR peak_only, which is per-beam); the peaked
    // detection_idx drives the whole downstream. Drain CFAR's mask / noise.
    add_flow(cfar, peak_pick, {{"detection_idx", "detection_idx"},
                               {"detection_score", "detection_score"}});
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // Bridge the peak beam's (az_bin, el_bin) columns to angle values. The
    // per-detection row_index is unused on the detection path (only the display
    // rep uses it as a marker label), so drain it to keep the op ticking.
    add_flow(peak_pick, aoa_lookup, {{"detection_idx", "detection_idx"}});
    auto row_index_sink = make_operator<NullSinkOp>("aoa_row_index_sink");
    add_flow(aoa_lookup, row_index_sink, {{"row_index", "input"}});

    // Clustering subgraph — dropped when peak_only is set. In peak_only the
    // cube peaks (already one per RD cell after AnglePeakPickOp) are reported
    // directly via DetectionReportOp: CubeAoaLookupOp supplies az/el and the
    // 4-col detection_idx is sliced to (doppler, range). The dense score is a
    // rank-4 per-beam cube with no 2-D map, so it is omitted from the report.
    std::shared_ptr<radar::ClusterDBSCANOp<Precision>> cluster;
    std::shared_ptr<radar::DetectionReductionOp<Precision>> reduction;
    if (!cfg.peak_only) {
      const std::vector<radar::AssemblyFeature> feature_columns{
          radar::AssemblyFeature::kRange, radar::AssemblyFeature::kVelocity,
          radar::AssemblyFeature::kAz, radar::AssemblyFeature::kEl};
      auto feature_assembly =
          make_operator<radar::FeatureAssemblyOp<Precision>>(
              "feature_assembly", Arg("features") = feature_columns);
      cluster = make_operator<radar::ClusterDBSCANOp<Precision>>(
          "cluster_dbscan", from_config("cluster_dbscan"));
      reduction = make_operator<radar::DetectionReductionOp<Precision>>(
          "detection_reduction", from_config("reduction"));
      auto report = make_operator<ClusterReportOp>("cluster_report");
      report->az_grid = cfg.az_angles;
      report->el_grid = cfg.el_angles;
      if (cadence) {
        report->cadence = cadence;
      }

      // Assemble [range, velocity, az, el] and cluster.
      add_flow(aoa_lookup, feature_assembly,
               {{"aoa_estimates", "aoa_estimates"}});
      add_flow(peak_pick, feature_assembly,
               {{"detection_idx", "detection_idx"}});
      add_flow(range, feature_assembly, {{"range_bins", "range_bins"}});
      add_flow(doppler, feature_assembly,
               {{"velocity_bins", "velocity_bins"}});
      add_flow(feature_assembly, cluster, {{"points", "points"}});

      add_flow(cluster, reduction, {{"labels", "labels"}});
      add_flow(peak_pick, reduction, {{"detection_idx", "detection_idx"}});
      add_flow(feature_assembly, reduction, {{"points", "points"}});

      add_flow(feature_assembly, report, {{"points", "points"}});
      add_flow(cluster, report, {{"labels", "labels"}, {"sizes", "sizes"}});
      add_flow(reduction, report, {{"rep_idx", "rep_idx"}});
      add_flow(range, report, {{"range_bins", "range_bins"}});
      add_flow(doppler, report, {{"velocity_bins", "velocity_bins"}});
    } else {
      auto det_report = make_operator<DetectionReportOp>("detection_report");
      if (cadence) {
        det_report->cadence = cadence;
      }
      auto det_idx_rd =
          make_operator<TrailingIdxColumnsOp>("det_idx_rd_report_slice");
      add_flow(peak_pick, det_idx_rd, {{"detection_idx", "idx"}});
      add_flow(det_idx_rd, det_report, {{"idx", "detection_idx"}});
      add_flow(aoa_lookup, det_report, {{"aoa_estimates", "aoa_estimates"}});
      add_flow(range, det_report, {{"range_bins", "range_bins"}});
      add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    }

    if (cfg.visualize) {
      compose_display(doppler, peak_pick, cluster, reduction, range);
    }
  }

 private:
  // Optional Holoviz display, reusing the app-agnostic radar_app/viz ops. The
  // RD map (channel-0 slice) fills the left; the right column stacks range-,
  // velocity-vs-time rolling plots and the boresight az/el scatter. The dense
  // 4-column detection_idx / rep_idx are sliced to their trailing
  // (doppler_bin, range_bin) columns for the RD-plane overlay/plot ops.
  void compose_display(
      const std::shared_ptr<radar::FMCWDopplerProcessorOp<Precision>>& doppler,
      const std::shared_ptr<AnglePeakPickOp>& detections,
      const std::shared_ptr<radar::ClusterDBSCANOp<Precision>>& cluster,
      const std::shared_ptr<radar::DetectionReductionOp<Precision>>& reduction,
      const std::shared_ptr<radar::FMCWRangeProcessorOp<Precision>>& range) {
    auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
    ch_slice->channel = 0;
    auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
    auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");
    // Peaked detection_idx (M, 4) -> (M, 2) trailing (doppler_bin, range_bin).
    auto det_idx_rd = make_operator<TrailingIdxColumnsOp>("det_idx_rd_slice");

    // radar_app owns the RD-map + panel layout; this app wires the flows.
    HolovizLayout layout;
    layout.window_title = "FMCW SIMO dense - RD (ch0) + clusters";
    layout.width = cfg.plots.enable ? 1200U : 800U;
    layout.height = cfg.plots.enable ? 760U : 600U;
    layout.headless = cfg.headless;
    layout.plots_enabled = cfg.plots.enable;
    // Side panels drawn with native Holoviz primitives (dynamic InputSpecs);
    // stacked right-column view rects.
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
    std::shared_ptr<TrailingIdxColumnsOp> rep_idx_rd;
    std::shared_ptr<CubeAoaLookupOp> rep_aoa;
    if (cfg.plots.enable) {
      plot = make_operator<RollingPlotOp>("rolling_plot");
      plot->window = cfg.plots.window;
      plot->frame_period = cfg.frame_period;
      plot->velocity_ylim_lo = cfg.velocity_ylim_lo;
      plot->velocity_ylim_hi = cfg.velocity_ylim_hi;
      plot->primitive = true;
      plot->range_view = range_view;
      plot->velocity_view = velocity_view;
      plot->range_aspect = panel_aspect(range_view);
      plot->velocity_aspect = panel_aspect(velocity_view);
      plot->marker_radius = cfg.viz.marker_radius;
      plot->marker_outline = cfg.viz.marker_outline;
      plot->marker_segments = cfg.viz.marker_segments;
      rep_idx_rd = make_operator<TrailingIdxColumnsOp>("rep_idx_rd_slice");
      rep_aoa = make_operator<CubeAoaLookupOp>("cube_aoa_lookup_rep");
      rep_aoa->az_grid = cfg.az_angles;
      rep_aoa->el_grid = cfg.el_angles;
      boresight = make_operator<BoresightViewOp>("boresight_view");
      double az_lo = 0.0, az_hi = 0.0, el_lo = 0.0, el_hi = 0.0;
      grid_extent(cfg.az_angles, -0.35, 0.35, &az_lo, &az_hi);
      grid_extent(cfg.el_angles, -0.35, 0.35, &el_lo, &el_hi);
      boresight->az_lo = az_lo;
      boresight->az_hi = az_hi;
      boresight->el_lo = el_lo;
      boresight->el_hi = el_hi;
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
    add_flow(detections, det_idx_rd, {{"detection_idx", "idx"}});
    add_flow(det_idx_rd, overlay, {{"idx", "detection_idx"}});
    add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
    // Cluster labels color the dots when clustering runs; peak_only has no
    // labels, so the overlay colors each peak by its row index instead.
    if (!cfg.peak_only) {
      add_flow(cluster, overlay, {{"labels", "labels"}});
      // Color clusters by their representative RD cell (stable across frames)
      // via the same sliced rep the plot uses; without plots rep_idx_rd is null
      // and the overlay falls back to the DBSCAN label.
      if (rep_idx_rd) {
        add_flow(rep_idx_rd, overlay, {{"idx", "rep_idx"}});
      }
    }
    add_flow(overlay, visualizer, {{"overlay", "receivers"}});
    if (plot) {
      // Plot the per-cluster representative (clustered) or the raw cube peaks
      // (peak_only); both carry the 4-col idx sliced to (doppler, range).
      if (!cfg.peak_only) {
        add_flow(reduction, rep_idx_rd, {{"rep_idx", "idx"}});
        // labels flag clustered mode; the plot then colors each rep by its RD
        // cell (matching the RD-map / boresight markers).
        add_flow(cluster, plot, {{"labels", "labels"}});
      } else {
        add_flow(detections, rep_idx_rd, {{"detection_idx", "idx"}});
      }
      add_flow(rep_idx_rd, plot, {{"idx", "rep_idx"}});
      add_flow(range, plot, {{"range_bins", "range_bins"}});
      add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
      add_flow(plot, visualizer, {{"plots", "receivers"}});
    }
    if (boresight) {
      // Clustered: the per-cluster representative (medoid) angle, with the rep
      // lookup's row_index as the cluster id coloring each marker. peak_only:
      // the raw per-detection cube peaks (row_index is then a per-detection id).
      if (!cfg.peak_only) {
        add_flow(reduction, rep_aoa, {{"rep_idx", "detection_idx"}});
        // Match the overlay/plot: color each cluster by its representative RD
        // cell. row_index (passed as labels) indexes rep_idx_rd row-for-row.
        add_flow(rep_idx_rd, boresight, {{"idx", "rep_idx"}});
      } else {
        add_flow(detections, rep_aoa, {{"detection_idx", "detection_idx"}});
      }
      add_flow(rep_aoa, boresight, {{"aoa_estimates", "aoa_estimates"},
                                    {"row_index", "labels"}});
      add_flow(boresight, visualizer, {{"plot", "receivers"}});
      // Primitive-panel InputSpec chain: rolling plot specs → boresight →
      // HolovizOp.input_specs (one connection into the single input_specs port).
      add_flow(plot, boresight, {{"plot_specs", "specs_in"}});
      add_flow(boresight, visualizer, {{"specs_out", "input_specs"}});
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwDenseSimoApp>(argc, argv, "fmcw_dense_simo",
                                        "fmcw_dense_simo.yaml", load_config);
}
