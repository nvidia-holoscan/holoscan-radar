// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MIMO FMCW sparse angle-then-cluster pipeline (TDM or DDM via YAML):
//
// TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
//        -> Compensation -> VirtualArrayAssemblyOp -> ...
// DDM: MimoDdmAdcSourceOp -> Range -> Doppler -> Compensation -> Clutter
//        -> VirtualArrayAssemblyOp -> ...
//
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_mimo/cpp/fmcw_mimo [config.yaml]

#include <cstdint>
#include <memory>
#include <string>
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
#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_app/app/driver.hpp"
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/display.hpp"
#include "radar_app/viz/viz.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir): colocated app header
#include "custom_ops.hpp"  // NOLINT(build/include_subdir): colocated app header

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)


class FmcwMimoApp : public holoscan::Application {
 public:
  AppConfig cfg;
  std::shared_ptr<CadenceMeter> cadence;

  void compose() override {
    const bool ddm = cfg.scheme == radar::MultiplexingScheme::kDdm;
    const std::string window_title =
        ddm ? "FMCW MIMO DDM - virtual RD + clusters"
            : "FMCW MIMO TDM - virtual RD + clusters";

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range_processor", from_config("range_processor"));
    auto clutter = make_operator<radar::ClutterFilterOp<float>>(
        "clutter_filter", from_config("clutter_filter"));
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler_processor", from_config("doppler_processor"));
    auto compensation =
        make_operator<radar::MIMOCoherentCompensationOp<float>>(
            "mimo_coherent_compensation",
            from_config("mimo_coherent_compensation"),
            // Prefer the app-resolved descriptor (may expand
            // generate_phase_codes into explicit phase_codes).
            holoscan::Arg("ddm_code") = cfg.ddm_code);
    auto vaa = make_operator<radar::VirtualArrayAssemblyOp<float>>(
        "virtual_array_assembly", from_config("virtual_array_assembly"),
        holoscan::Arg("tx_positions") = cfg.tx_positions_flat,
        holoscan::Arg("rx_positions") = cfg.rx_positions_flat,
        holoscan::Arg("wavelength") = cfg.wavelength);

    if (cfg.meter_cadence) {
      cadence = std::make_shared<CadenceMeter>();
      cadence->frame_period_s = cfg.frame_period;
      cadence->log_every = cfg.meter_log_every;
    }

    if (ddm) {
      auto source = make_operator<MimoDdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->use_phase_codes = cfg.ddm_correlate;
      if (cfg.ddm_correlate) {
        source->phase_codes = cfg.ddm_code.phase_codes;
      } else {
        source->doppler_offsets = cfg.ddm_doppler_offsets;
      }
      source->frame_period = cfg.frame_period;
      if (cadence) {
        source->cadence = cadence;
      }

      add_flow(source, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, clutter, {{"rd_maps", "range_profiles"}});
      add_flow(clutter, vaa, {{"range_profiles", "rd_maps"}});
    } else {
      auto source = make_operator<MimoTdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->tx_schedule = cfg.tx_schedule;
      source->frame_period = cfg.frame_period;
      if (cadence) {
        source->cadence = cadence;
      }

      auto deinterleave = make_operator<radar::MIMODeinterleaveOp<float>>(
          "mimo_deinterleave", from_config("mimo_deinterleave"));

      add_flow(source, deinterleave, {{"adc_cube", "adc_cube"}});
      add_flow(deinterleave, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
      add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, vaa, {{"rd_maps", "rd_maps"}});
    }

    // Derive the CFAR NCI fold (K) from the same virtual array geometry
    // beamscan consumes (num_virtual = elements after the overlap policy), so
    // it never drifts from num_tx / rx_array. nci_axes: [0] integrates the full
    // virtual-channel axis, so K = num_elements; the operator ignores this hint
    // when nci_axes is empty (K = 1).
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
    if (cadence) {
      det_report->cadence = cadence;
    }
    auto mask_sink = make_operator<NullSinkOp>("detection_mask_sink");
    auto noise_sink = make_operator<NullSinkOp>("noise_estimate_sink");
    auto spectrum_sink =
        make_operator<TensorMapDiscardOp>("angle_spectrum_discard");

    std::shared_ptr<DdmReportVelocityBinsOp> ddm_velocity_bins;
    if (ddm) {
      ddm_velocity_bins =
          make_operator<DdmReportVelocityBinsOp>("ddm_velocity_bins");
      ddm_velocity_bins->waveform = cfg.report_waveform;
      ddm_velocity_bins->num_doppler = cfg.report_num_doppler;
      ddm_velocity_bins->bin_spacing_hz =
          (1.0 / cfg.report_waveform.chirp_repetition_interval()) /
          static_cast<double>(cfg.scene.num_chirps);
      add_flow(range, ddm_velocity_bins, {{"range_bins", "sync"}});
    }

    add_flow(vaa, cfar, {{"rd_maps", "rd_maps"}});
    add_flow(vaa, gather, {{"rd_maps", "rd_maps"}});
    add_flow(cfar, gather, {{"detection_idx", "detection_idx"}});
    add_flow(gather, beamscan, {{"snapshot", "snapshot"},
                                {"is_valid", "is_valid"}});
    add_flow(vaa, beamscan, {{"array_geometry", "array_geometry"}});
    add_flow(beamscan, spectrum_sink, {{"angle_spectrum", "input"}});

    add_flow(cfar, det_report, {{"detection_idx", "detection_idx"},
                                  {"detection_score", "detection_score"}});
    add_flow(range, det_report, {{"range_bins", "range_bins"}});
    if (ddm) {
      add_flow(ddm_velocity_bins, det_report,
               {{"velocity_bins", "velocity_bins"}});
    } else {
      add_flow(doppler, det_report, {{"velocity_bins", "velocity_bins"}});
    }
    add_flow(cfar, mask_sink, {{"detection_mask", "input"}});
    add_flow(cfar, noise_sink, {{"noise_estimate", "input"}});

    // Clustering subgraph — dropped when peak_only is set. The angle branch
    // (gather/beamscan) still runs; in
    // peak_only its aoa feeds DetectionReportOp
    // (optional aoa_estimates) instead of the removed feature_assembly.
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
      if (ddm) {
        add_flow(ddm_velocity_bins, feature_assembly,
                 {{"velocity_bins", "velocity_bins"}});
      } else {
        add_flow(doppler, feature_assembly,
                 {{"velocity_bins", "velocity_bins"}});
      }
      add_flow(feature_assembly, cluster, {{"points", "points"}});
      add_flow(feature_assembly, cluster_report, {{"points", "points"}});
      add_flow(cluster, cluster_report, {{"labels", "labels"},
                                        {"sizes", "sizes"}});
      add_flow(cluster, reduction, {{"labels", "labels"}});
      add_flow(cfar, reduction, {{"detection_idx", "detection_idx"}});
      add_flow(feature_assembly, reduction, {{"points", "points"}});
      add_flow(reduction, cluster_report, {{"rep_idx", "rep_idx"}});
      add_flow(cfar, cluster_report, {{"detection_idx", "detection_idx"}});
      add_flow(range, cluster_report, {{"range_bins", "range_bins"}});
      if (ddm) {
        add_flow(ddm_velocity_bins, cluster_report,
                 {{"velocity_bins", "velocity_bins"}});
      } else {
        add_flow(doppler, cluster_report,
                 {{"velocity_bins", "velocity_bins"}});
      }
    } else {
      add_flow(beamscan, det_report, {{"aoa_estimates", "aoa_estimates"}});
    }

    if (cfg.visualize) {
      // NCI over the virtual-channel axis: display the same surface CFAR
      // thresholds (nci_axes: [0]), so the picture and the overlaid
      // detections come from the same data.
      auto rd_nci = make_operator<RdNciOp>("rd_nci");
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = window_title;
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
      layout.vsync = cfg.viz.vsync;
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
      overlay->rd_region = rd_map_region(layout);  // axes/frame/title region
      overlay->rd_view_px_height = static_cast<float>(layout.height);
      overlay->frame_period = cfg.frame_period;    // scenario clock
      overlay->marker_radius = cfg.viz.marker_radius;
      overlay->marker_outline = cfg.viz.marker_outline;
      overlay->marker_segments = cfg.viz.marker_segments;

      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
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

      add_flow(vaa, rd_nci, {{"rd_maps", "rd_maps"}});
      add_flow(rd_nci, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      // Overlay/plot/boresight color by cluster when clustering runs (keyed on
      // each cluster's representative RD cell, via rep_idx, so the color is
      // stable across frames); in peak_only mode there are no labels, so all
      // three color each peak by its per-detection row index instead.
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(rd_nci, overlay, {{"rd_map", "rd_maps"}});
      // Physical axis bins for the RD-map
      // tick labels (range X, velocity Y); the
      // velocity axis comes from the DDM report bins in DDM mode, matching the
      // rolling plot's source.
      add_flow(range, overlay, {{"range_bins", "range_bins"}});
      if (ddm) {
        add_flow(ddm_velocity_bins, overlay,
                 {{"velocity_bins", "velocity_bins"}});
      } else {
        add_flow(doppler, overlay, {{"velocity_bins", "velocity_bins"}});
      }
      if (!cfg.peak_only) {
        add_flow(cluster, overlay, {{"labels", "labels"}});
        add_flow(reduction, overlay, {{"rep_idx", "rep_idx"}});
      }
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      // Scenario clock overlay: its spec heads the primitive-panel InputSpec
      // chain when panels are on (overlay → rolling plot → boresight →
      // input_specs), else feeds input_specs directly.
      if (plot) {
        add_flow(overlay, plot, {{"time_specs", "specs_in"}});
      } else {
        add_flow(overlay, visualizer, {{"time_specs", "input_specs"}});
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
        if (ddm) {
          add_flow(ddm_velocity_bins, plot,
                   {{"velocity_bins", "velocity_bins"}});
        } else {
          add_flow(doppler, plot, {{"velocity_bins", "velocity_bins"}});
        }
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
        // HolovizOp.input_specs (one connection
        // into the single input_specs port).
        add_flow(plot, boresight, {{"plot_specs", "specs_in"}});
        add_flow(boresight, visualizer, {{"specs_out", "input_specs"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwMimoApp>(argc, argv, "fmcw_mimo", "fmcw_mimo.yaml",
                                   load_config);
}
