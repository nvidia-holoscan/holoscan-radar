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
#include "config.hpp"
#include "custom_ops.hpp"
#include "mimo_sim.hpp"

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)


class FmcwMimoApp : public holoscan::Application {
 public:
  AppConfig cfg;

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
            from_config("mimo_coherent_compensation"));
    auto vaa = make_operator<radar::VirtualArrayAssemblyOp<float>>(
        "virtual_array_assembly", from_config("virtual_array_assembly"),
        holoscan::Arg("tx_positions") = cfg.tx_positions_flat,
        holoscan::Arg("rx_positions") = cfg.rx_positions_flat,
        holoscan::Arg("wavelength") = cfg.wavelength);

    if (ddm) {
      auto source = make_operator<MimoDdmAdcSourceOp>(
          "adc_source",
          make_condition<holoscan::CountCondition>("frame_count",
                                                   cfg.num_frames),
          make_condition<holoscan::PeriodicCondition>(
              "frame_rate", static_cast<int64_t>(cfg.frame_period * 1e9)));
      source->scene = cfg.scene;
      source->doppler_offsets = cfg.ddm_doppler_offsets;
      source->frame_period = cfg.frame_period;

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

      auto deinterleave = make_operator<radar::MIMODeinterleaveOp<float>>(
          "mimo_deinterleave", from_config("mimo_deinterleave"));

      add_flow(source, deinterleave, {{"adc_cube", "adc_cube"}});
      add_flow(deinterleave, range, {{"adc_cube", "adc_cube"}});
      add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
      add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
      add_flow(doppler, compensation, {{"rd_maps", "rd_maps"}});
      add_flow(compensation, vaa, {{"rd_maps", "rd_maps"}});
    }

    auto cfar = make_operator<radar::CFARDetectorOp<float>>(
        "cfar_detector", from_config("cfar_detector"));
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
    add_flow(beamscan, feature_assembly,
             {{"aoa_estimates", "aoa_estimates"}});
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

    if (cfg.visualize) {
      auto ch_slice = make_operator<RdChannelSliceOp>("rd_channel_slice");
      ch_slice->channel = 0;
      auto to_image = make_operator<RdMapToImageOp>("rd_map_to_image");
      auto overlay = make_operator<ClusterOverlayOp>("cluster_overlay");

      HolovizLayout layout;
      layout.window_title = window_title;
      layout.width = cfg.plots.enable ? 1200U : 800U;
      layout.height = cfg.plots.enable ? 760U : 600U;
      layout.headless = cfg.headless;
      layout.plots_enabled = cfg.plots.enable;
      if (cfg.plots.enable) {
        layout.right_panels = range_velocity_boresight_panels();
      }
      auto visualizer = make_rd_holoviz_op(this, "holoviz", layout);

      std::shared_ptr<RollingPlotOp> plot;
      std::shared_ptr<BoresightViewOp> boresight;
      if (cfg.plots.enable) {
        plot = make_operator<RollingPlotOp>("rolling_plot");
        plot->window = cfg.plots.window;
        plot->frame_period = cfg.frame_period;
        plot->velocity_ylim_lo = cfg.velocity_ylim_lo;
        plot->velocity_ylim_hi = cfg.velocity_ylim_hi;
        boresight = make_operator<BoresightViewOp>("boresight_view");
        boresight->az_lo = cfg.az_lo;
        boresight->az_hi = cfg.az_hi;
        boresight->el_lo = cfg.el_lo;
        boresight->el_hi = cfg.el_hi;
      }

      add_flow(vaa, ch_slice, {{"rd_maps", "rd_maps"}});
      add_flow(ch_slice, to_image, {{"rd_map", "rd_maps"}});
      add_flow(to_image, visualizer, {{"rd_image", "receivers"}});
      add_flow(cfar, overlay, {{"detection_idx", "detection_idx"}});
      add_flow(cluster, overlay, {{"labels", "labels"}});
      add_flow(ch_slice, overlay, {{"rd_map", "rd_maps"}});
      add_flow(overlay, visualizer, {{"overlay", "receivers"}});
      if (plot) {
        add_flow(reduction, plot, {{"rep_idx", "rep_idx"}});
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
        add_flow(cluster, boresight, {{"labels", "labels"}});
        add_flow(boresight, visualizer, {{"plot", "receivers"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  return run_fmcw_app<FmcwMimoApp>(argc, argv, "fmcw_mimo", "fmcw_mimo.yaml",
                                   load_config);
}
