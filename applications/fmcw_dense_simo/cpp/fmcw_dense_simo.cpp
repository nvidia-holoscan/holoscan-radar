// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIMO FMCW *dense* angle-then-cluster pipeline (HSDK 5 Graph):
//
//   SimAdcSourceOp -> Range -> Clutter -> Doppler
//     -> PhaseShiftBeamformerOp (rd_maps -> rda_cube)
//     -> CFARDetectorOp (nci_axes=[]; az/el ride as batch)
//     -> AnglePeakPickOp -> CubeAoaLookupOp
//     -> [PhysicalRangeVelocityAzElPointsOp -> DBSCAN -> Reduction]
//     -> DetectionReportOp / ClusterReportOp
//     [-> RdAngleMaxOp(cfar_power) -> RdMapToImageOp -> X11RdViewerOp]
//
// Dense detection_idx / rep_idx carry (az_bin, el_bin) ahead of
// (doppler, range); TrailingIdxColumnsOp slices the trailing two columns for
// PhysicalPoints and the X11 viewer. FeatureAssembly optional-port gap →
// PhysicalRangeVelocityAzElPointsOp.
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo [config.yaml]

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/physical_points_op.hpp"
#include "radar_app/app/source.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/viz/rd_angle_max_op.hpp"
#include "config.hpp"      // NOLINT(build/include_subdir): colocated app header
#include "custom_ops.hpp"  // NOLINT(build/include_subdir): colocated app header

#if defined(HOLOSCAN_RADAR_HAS_X11)
#include "radar_app/viz/rd_map_to_image_op.hpp"
#include "radar_app/viz/x11_rd_viewer_op.hpp"
#endif

namespace radar = holoscan::radar;
using namespace holoscan::radar::app;  // NOLINT(build/namespaces)

namespace {

void bind_out(holoscan::CompileOptions& options, const char* op,
              const char* port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = op,
          .output_port = port,
          .device = holoscan::DeviceId{0},
      });
}

}  // namespace

int main(int argc, char** argv) {
  return run_fmcw_graph_app<AppConfig>(
      argc, argv, "fmcw_dense_simo", "fmcw_dense_simo.yaml", load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         AppConfig& cfg, std::shared_ptr<CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        SimAdcSourceConfig src_cfg;
        src_cfg.scene = cfg.scene;
        src_cfg.frame_period = cfg.frame_period;
        src_cfg.num_frames = cfg.num_frames;

        std::shared_ptr<CadenceMeter> cadence;
        if (cfg.meter_cadence) {
          cadence = std::make_shared<CadenceMeter>();
          cadence->frame_period_s = cfg.frame_period;
          cadence->log_every = cfg.meter_log_every;
          src_cfg.cadence = cadence;
          *cadence_out = cadence;
        }

        auto source =
            graph.op<SimAdcSourceOp>("adc_source", extents, std::move(src_cfg));

        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);

        auto clutter = graph.op<radar::ClutterFilterOp<float>>(
            "clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);

        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);

        auto beamformer = graph.op<radar::PhaseShiftBeamformerOp<float>>(
            "phase_shift_beamformer", extents);
        beamformer.set_params(cfg.beamformer_params);

        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", cfg.cfar_extents);
        cfar.set_params(cfg.cfar_params);

        auto peak_pick =
            graph.op<AnglePeakPickOp>("angle_peak_pick", extents);
        auto aoa_lookup = graph.op<CubeAoaLookupOp>(
            "cube_aoa_lookup", extents,
            CubeAoaLookupOp::Config{cfg.beamformer_params.az_angles,
                                    cfg.beamformer_params.el_angles});

        // (N,4) → (N,2) trailing (doppler, range) for PhysicalPoints / viewer
        // / peak_only DetectionReport.
        auto det_idx_rd =
            graph.op<TrailingIdxColumnsOp>("det_idx_rd_slice", extents);

        auto mask_sink = graph.op<NullSinkOp>("detection_mask_sink");
        auto noise_sink = graph.op<NullSinkOp>("noise_estimate_sink");
        auto power_sink = graph.op<NullSinkOp>("cfar_power_sink");

        graph.add_flow(source->adc_cube, range->adc_cube, q);
        graph.add_flow(range->range_profiles, clutter->range_profiles, q);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
        graph.add_flow(doppler->rd_maps, beamformer->rd_maps, q);
        graph.add_flow(beamformer->rda_cube, cfar->rd_maps, q);

        graph.add_flow(cfar->detection_idx, peak_pick->detection_idx, q);
        graph.add_flow(cfar->detection_score, peak_pick->detection_score, q);
        graph.add_flow(peak_pick->detection_idx_out, aoa_lookup->detection_idx,
                       q);
        graph.add_flow(peak_pick->detection_idx_out, det_idx_rd->idx, q);
        graph.add_flow(cfar->detection_mask, mask_sink->input, q);
        graph.add_flow(cfar->noise_estimate, noise_sink->input, q);

#if defined(HOLOSCAN_RADAR_HAS_X11)
        std::optional<holoscan::OperatorHandle<X11RdViewerOp>> viewer;
        std::optional<holoscan::OperatorHandle<TrailingIdxColumnsOp>>
            rep_idx_rd;
        if (cfg.visualize) {
          X11RdViewerConfig viz_cfg;
          viz_cfg.title = "fmcw_dense_simo - Range-Doppler (max-beam)";
          viz_cfg.width = static_cast<int>(cfg.viz.window_width);
          viz_cfg.height = static_cast<int>(cfg.viz.window_height);
          viz_cfg.overlay_detections = true;
          viz_cfg.overlay_clusters = !cfg.peak_only;
          viz_cfg.plots_enabled = cfg.plots.enable;
          viz_cfg.plot_window = cfg.plots.window;
          viz_cfg.frame_period = cfg.frame_period;
          viz_cfg.marker_radius = cfg.viz.marker_radius;
          viz_cfg.marker_outline = cfg.viz.marker_outline;
          // Peak-beam angles over the beamformer scan grid extents.
          viz_cfg.overlay_boresight = true;
          viz_cfg.az_lo = cfg.az_lo;
          viz_cfg.az_hi = cfg.az_hi;
          viz_cfg.el_lo = cfg.el_lo;
          viz_cfg.el_hi = cfg.el_hi;

          auto angle_max =
              graph.op<RdAngleMaxOp>("rd_angle_max", extents);
          auto to_image =
              graph.op<RdMapToImageOp>("rd_map_to_image", extents);
          viewer = graph.op<X11RdViewerOp>("x11_rd_viewer", viz_cfg);
          // Max over beams so overlays match AnglePeakPick's peak projection.
          graph.add_flow(cfar->cfar_power, angle_max->rd_maps, q);
          graph.add_flow(angle_max->rd_map, to_image->rd_maps, q);
          graph.add_flow(to_image->rd_image, (*viewer)->rd_image, q);
          graph.add_flow(det_idx_rd->idx_out, (*viewer)->detection_idx, q);
          graph.add_flow(range->range_bins, (*viewer)->range_bins, q);
          graph.add_flow(doppler->velocity_bins, (*viewer)->velocity_bins, q);
          graph.add_flow(aoa_lookup->aoa_estimates, (*viewer)->aoa_estimates,
                         q);
          bind_out(options, "rd_angle_max", "rd_map");
          bind_out(options, "rd_map_to_image", "rd_image");
          HOLOSCAN_LOG_INFO(
              "fmcw_dense_simo: X11 RD live viewer enabled ({}x{})",
              viz_cfg.width, viz_cfg.height);
        }
        // [EA HSDK5 ONLY] Drain cfar_power even when the viewer's RdAngleMax
        // also consumes it: an operator left with no wired port is an isolated
        // vertex, which auto-partitioning rejects
        // (PARTITION_MANUAL_AUTHORING_REQUIRED — HSDK5_PORT §6).
        graph.add_flow(cfar->cfar_power, power_sink->input, q);
#else
        if (cfg.visualize) {
          HOLOSCAN_LOG_WARN(
              "fmcw_dense_simo: visualize requested but X11/Holoviz "
              "unavailable; running log-only");
          cfg.visualize = false;
        }
        graph.add_flow(cfar->cfar_power, power_sink->input, q);
#endif

        if (cfg.peak_only) {
          // Dense detection_score is rank-4; collapse to (D,R) for the report.
          auto score_max =
              graph.op<RdAngleMaxOp>("detection_score_max", extents);
          auto sink = graph.op<DetectionReportOp>(
              "detection_report",
              DetectionReportOp::Config{cadence, /*report_aoa=*/true});
          graph.add_flow(cfar->detection_score, score_max->rd_maps, q);
          graph.add_flow(score_max->rd_map, sink->detection_score, q);
          graph.add_flow(det_idx_rd->idx_out, sink->detection_idx, q);
          graph.add_flow(aoa_lookup->aoa_estimates, sink->aoa_estimates, q);
          graph.add_flow(range->range_bins, sink->range_bins, q);
          graph.add_flow(doppler->velocity_bins, sink->velocity_bins, q);
          bind_out(options, "detection_score_max", "rd_map");
        } else {
          auto feature_assembly =
              graph.op<PhysicalRangeVelocityAzElPointsOp>("feature_assembly",
                                                          extents);

          auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
              "cluster_dbscan", extents);
          cluster.set_params(cfg.dbscan_params);

          auto reduction = graph.op<radar::DetectionReductionOp<float>>(
              "detection_reduction", extents);
          reduction.set_params(cfg.reduction_params);

          // 4-col rep_idx from reduction → az_grid/el_grid for bin→rad.
          auto cluster_report = graph.op<ClusterReportOp>(
              "cluster_report",
              ClusterReportOp::Config{
                  /*report_detection_idx=*/false,
                  cfg.beamformer_params.az_angles,
                  cfg.beamformer_params.el_angles,
                  cadence});

          graph.add_flow(det_idx_rd->idx_out, feature_assembly->detection_idx,
                         q);
          graph.add_flow(range->range_bins, feature_assembly->range_bins, q);
          graph.add_flow(doppler->velocity_bins,
                         feature_assembly->velocity_bins, q);
          graph.add_flow(aoa_lookup->aoa_estimates,
                         feature_assembly->aoa_estimates, q);
          graph.add_flow(feature_assembly->points, cluster->points, q);
          graph.add_flow(feature_assembly->points, cluster_report->points, q);
          graph.add_flow(cluster->labels, cluster_report->labels, q);
          graph.add_flow(cluster->sizes, cluster_report->sizes, q);
          graph.add_flow(cluster->labels, reduction->labels, q);
          // Keep 4-col detection_idx into reduction → 4-col rep_idx.
          graph.add_flow(peak_pick->detection_idx_out, reduction->detection_idx,
                         q);
          graph.add_flow(feature_assembly->points, reduction->points, q);
          graph.add_flow(reduction->rep_idx, cluster_report->rep_idx, q);
          graph.add_flow(range->range_bins, cluster_report->range_bins, q);
          graph.add_flow(doppler->velocity_bins, cluster_report->velocity_bins,
                         q);

#if defined(HOLOSCAN_RADAR_HAS_X11)
          if (viewer) {
            rep_idx_rd = graph.op<TrailingIdxColumnsOp>("rep_idx_rd_slice",
                                                        extents);
            graph.add_flow(reduction->rep_idx, (*rep_idx_rd)->idx, q);
            graph.add_flow(cluster->labels, (*viewer)->labels, q);
            graph.add_flow((*rep_idx_rd)->idx_out, (*viewer)->rep_idx, q);
            bind_out(options, "rep_idx_rd_slice", "idx_out");
          }
#endif

          bind_out(options, "feature_assembly", "points");
          bind_out(options, "cluster_dbscan", "labels");
          bind_out(options, "cluster_dbscan", "sizes");
          bind_out(options, "detection_reduction", "rep_idx");
        }

        bind_out(options, "adc_source", "adc_cube");
        bind_out(options, "range_processor", "range_profiles");
        bind_out(options, "range_processor", "range_bins");
        bind_out(options, "clutter_filter", "range_profiles_out");
        bind_out(options, "doppler_processor", "rd_maps");
        bind_out(options, "doppler_processor", "velocity_bins");
        bind_out(options, "phase_shift_beamformer", "rda_cube");
        bind_out(options, "cfar_detector", "detection_mask");
        bind_out(options, "cfar_detector", "detection_score");
        bind_out(options, "cfar_detector", "noise_estimate");
        bind_out(options, "cfar_detector", "detection_idx");
        bind_out(options, "cfar_detector", "cfar_power");
        bind_out(options, "angle_peak_pick", "detection_idx_out");
        bind_out(options, "cube_aoa_lookup", "aoa_estimates");
        bind_out(options, "det_idx_rd_slice", "idx_out");
      });
}
