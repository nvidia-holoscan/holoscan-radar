// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// MIMO FMCW sparse angle-then-cluster pipeline (HSDK 5 Graph; TDM or DDM):
//
// TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
//        -> TDMDenseCompensationOp -> VirtualArrayAssemblyOp -> ...
// DDM: MimoDdmAdcSourceOp -> Range -> Doppler
//        -> DDMDenseDemultiplexOp -> Clutter -> VirtualArrayAssemblyOp -> ...
// Then: CFAR -> SnapshotGather -> Beamscan
//        -> [FeatureAssemblyOp -> DBSCAN -> Reduction]
//        -> DetectionReportOp / ClusterReportOp
//        [-> RdMapToImageOp(cfar_power) -> X11RdViewerOp]
//
// SnapshotGather's is_valid mask aligns into Beamscan; VAA geometry directly
// drives Beamscan's Exact live override.
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_mimo/cpp/fmcw_mimo [config.yaml]

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "angle_estimation/snapshot_gather_op/snapshot_gather_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "config.hpp"       // NOLINT(build/include_subdir): colocated app header
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
      argc, argv, "fmcw_mimo", "fmcw_mimo.yaml", load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         AppConfig& cfg, std::shared_ptr<CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions q{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;
        const bool ddm = cfg.scheme == radar::MultiplexingScheme::kDdm;

        std::shared_ptr<CadenceMeter> cadence;
        if (cfg.meter_cadence) {
          cadence = std::make_shared<CadenceMeter>();
          cadence->frame_period_s = cfg.frame_period;
          cadence->log_every = cfg.meter_log_every;
          *cadence_out = cadence;
        }

        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);

        auto clutter = graph.op<radar::ClutterFilterOp<float>>(
            "clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);

        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);

        auto vaa = graph.op<radar::VirtualArrayAssemblyOp<float>>(
            "virtual_array_assembly", extents);
        vaa.set_params(cfg.vaa_params);

        // Velocity-bin producer for reports / assembly / viewer.
        // TDM: DopplerProcessor; DDM: DdmReportVelocityBins (demux spacing).
        std::optional<holoscan::OperatorHandle<DdmReportVelocityBinsOp>>
            ddm_velocity;
        if (ddm) {
          DdmReportVelocityBinsConfig vcfg;
          vcfg.waveform = cfg.report_waveform;
          vcfg.num_doppler = cfg.report_num_doppler;
          vcfg.bin_spacing_hz =
              (1.0 / cfg.report_waveform.chirp_repetition_interval()) /
              static_cast<double>(cfg.scene.num_chirps);
          ddm_velocity = graph.op<DdmReportVelocityBinsOp>(
              "ddm_velocity_bins", extents, std::move(vcfg));
          graph.add_flow(range->range_bins, (*ddm_velocity)->sync, q);
        }

        if (ddm) {
          auto demultiplex = graph.op<radar::DDMDenseDemultiplexOp<float>>(
              "ddm_dense_demultiplex", extents);
          demultiplex.set_params(cfg.ddm_demultiplex_params);
          MimoDdmAdcSourceConfig src_cfg;
          src_cfg.scene = cfg.scene;
          src_cfg.use_phase_codes = cfg.ddm_correlate;
          if (cfg.ddm_correlate) {
            src_cfg.phase_codes = cfg.ddm_code.phase_codes;
          } else {
            src_cfg.doppler_offsets = cfg.ddm_doppler_offsets;
          }
          src_cfg.frame_period = cfg.frame_period;
          src_cfg.num_frames = cfg.num_frames;
          src_cfg.cadence = cadence;

          auto source = graph.op<MimoDdmAdcSourceOp>(
              "adc_source", extents, std::move(src_cfg));
          // Reports/viewer use DdmReportVelocityBins; drain Doppler's own
          // velocity_bins (always emitted when waveform is set).
          auto doppler_vel_sink =
              graph.op<NullSinkOp>("doppler_velocity_bins_sink");
          graph.add_flow(source->adc_cube, range->adc_cube, q);
          graph.add_flow(range->range_profiles, doppler->range_profiles, q);
          graph.add_flow(doppler->rd_maps, demultiplex->rd_maps, q);
          graph.add_flow(doppler->velocity_bins, doppler_vel_sink->input, q);
          graph.add_flow(demultiplex->rd_maps_out, clutter->range_profiles, q);
          graph.add_flow(clutter->range_profiles_out, vaa->rd_maps, q);
        } else {
          auto compensation = graph.op<radar::TDMDenseCompensationOp<float>>(
              "tdm_dense_compensation", extents);
          compensation.set_params(cfg.tdm_compensation_params);
          MimoTdmAdcSourceConfig src_cfg;
          src_cfg.scene = cfg.scene;
          src_cfg.tx_schedule = cfg.tx_schedule;
          src_cfg.frame_period = cfg.frame_period;
          src_cfg.num_frames = cfg.num_frames;
          src_cfg.cadence = cadence;

          auto source = graph.op<MimoTdmAdcSourceOp>(
              "adc_source", extents, std::move(src_cfg));
          auto deinterleave = graph.op<radar::MIMODeinterleaveOp<float>>(
              "mimo_deinterleave", extents);
          deinterleave.set_params(cfg.deinterleave_params);

          graph.add_flow(source->adc_cube, deinterleave->adc_cube, q);
          graph.add_flow(deinterleave->adc_cube_out, range->adc_cube, q);
          graph.add_flow(range->range_profiles, clutter->range_profiles, q);
          graph.add_flow(clutter->range_profiles_out, doppler->range_profiles,
                         q);
          graph.add_flow(doppler->rd_maps, compensation->rd_maps, q);
          graph.add_flow(compensation->rd_maps_out, vaa->rd_maps, q);

          bind_out(options, "mimo_deinterleave", "adc_cube_out");
        }

        auto cfar =
            graph.op<radar::CFARDetectorOp<float>>("cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);

        auto gather = graph.op<radar::SnapshotGatherOp<float>>(
            "snapshot_gather", extents);
        gather.set_params(cfg.gather_params);

        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);

        auto sink = graph.op<DetectionReportOp>(
            "detection_report",
            DetectionReportOp::Config{
                cadence, /*report_aoa=*/cfg.use_cfar_peaks});
        auto mask_sink = graph.op<NullSinkOp>("detection_mask_sink");
        auto noise_sink = graph.op<NullSinkOp>("noise_estimate_sink");
        auto power_sink = graph.op<NullSinkOp>("cfar_power_sink");
        auto spectrum_sink = graph.op<NullSinkOp>("angle_spectrum_sink");

        graph.add_flow(vaa->rd_maps_out, cfar->rd_maps, q);
        graph.add_flow(vaa->rd_maps_out, gather->rd_maps, q);
        graph.add_flow(cfar->detection_idx, gather->detection_idx, q);
        graph.add_flow(gather->snapshot, beamscan->snapshot, q);
        graph.add_flow(vaa->array_geometry, beamscan->array_geometry, q);
        graph.add_flow(gather->is_valid, beamscan->is_valid, q);
        graph.add_flow(beamscan->angle_spectrum, spectrum_sink->input, q);

        graph.add_flow(cfar->detection_idx, sink->detection_idx, q);
        graph.add_flow(cfar->detection_score, sink->detection_score, q);
        graph.add_flow(range->range_bins, sink->range_bins, q);
        if (ddm) {
          graph.add_flow((*ddm_velocity)->velocity_bins, sink->velocity_bins,
                         q);
        } else {
          graph.add_flow(doppler->velocity_bins, sink->velocity_bins, q);
        }
        graph.add_flow(cfar->detection_mask, mask_sink->input, q);
        graph.add_flow(cfar->noise_estimate, noise_sink->input, q);
        // [EA HSDK5 ONLY] Drain cfar_power even when the viewer also consumes
        // it: an operator left with no wired port is an isolated vertex, which
        // auto-partitioning rejects (PARTITION_MANUAL_AUTHORING_REQUIRED —
        // HSDK5_PORT §6).
        graph.add_flow(cfar->cfar_power, power_sink->input, q);

#if defined(HOLOSCAN_RADAR_HAS_X11)
        std::optional<holoscan::OperatorHandle<X11RdViewerOp>> viewer;
        if (cfg.visualize) {
          X11RdViewerConfig viz_cfg;
          viz_cfg.title = ddm ? "fmcw_mimo - Range-Doppler (DDM NCI)"
                              : "fmcw_mimo - Range-Doppler (TDM NCI)";
          viz_cfg.width = static_cast<int>(cfg.viz.window_width);
          viz_cfg.height = static_cast<int>(cfg.viz.window_height);
          viz_cfg.overlay_detections = true;
          viz_cfg.overlay_clusters = !cfg.use_cfar_peaks;
          viz_cfg.plots_enabled = cfg.plots.enable;
          viz_cfg.plot_window = cfg.plots.window;
          viz_cfg.frame_period = cfg.frame_period;
          viz_cfg.marker_radius = cfg.viz.marker_radius;
          viz_cfg.marker_outline = cfg.viz.marker_outline;
          // Virtual-array beamscan angles over the scan grid extents.
          viz_cfg.overlay_boresight = true;
          viz_cfg.az_lo = cfg.az_lo;
          viz_cfg.az_hi = cfg.az_hi;
          viz_cfg.el_lo = cfg.el_lo;
          viz_cfg.el_hi = cfg.el_hi;

          auto to_image =
              graph.op<RdMapToImageOp>("rd_map_to_image", extents);
          viewer = graph.op<X11RdViewerOp>("x11_rd_viewer", viz_cfg);
          graph.add_flow(cfar->cfar_power, to_image->rd_maps, q);
          graph.add_flow(to_image->rd_image, (*viewer)->rd_image, q);
          graph.add_flow(cfar->detection_idx, (*viewer)->detection_idx, q);
          graph.add_flow(range->range_bins, (*viewer)->range_bins, q);
          graph.add_flow(beamscan->aoa_estimates, (*viewer)->aoa_estimates, q);
          if (ddm) {
            graph.add_flow((*ddm_velocity)->velocity_bins,
                           (*viewer)->velocity_bins, q);
          } else {
            graph.add_flow(doppler->velocity_bins, (*viewer)->velocity_bins, q);
          }
          bind_out(options, "rd_map_to_image", "rd_image");
          HOLOSCAN_LOG_INFO("fmcw_mimo: X11 RD live viewer enabled ({}x{})",
                            viz_cfg.width, viz_cfg.height);
        }
#else
        if (cfg.visualize) {
          HOLOSCAN_LOG_WARN(
              "fmcw_mimo: visualize requested but X11/Holoviz unavailable; "
              "running log-only");
          cfg.visualize = false;
        }
#endif

        if (cfg.use_cfar_peaks) {
          graph.add_flow(beamscan->aoa_estimates, sink->aoa_estimates, q);
        } else {
          std::vector<radar::AssemblyFeature> features{
              radar::AssemblyFeature::kRange,
              radar::AssemblyFeature::kVelocity,
              radar::AssemblyFeature::kAzEstimate,
              radar::AssemblyFeature::kElEstimate};
          auto feature_assembly =
              graph.op<radar::FeatureAssemblyOp<float>>(
                  "feature_assembly", extents, std::move(features));

          auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
              "cluster_dbscan", extents);
          cluster.set_params(cfg.dbscan_params);

          auto reduction = graph.op<radar::DetectionReductionOp<float>>(
              "detection_reduction", extents);
          reduction.set_params(cfg.reduction_params);

          auto cluster_report = graph.op<ClusterReportOp>(
              "cluster_report",
              ClusterReportOp::Config{/*report_detection_idx=*/true});

          graph.add_flow(cfar->detection_idx, feature_assembly->detection_idx,
                         q);
          graph.add_flow(range->range_bins, feature_assembly->range_bins, q);
          if (ddm) {
            graph.add_flow((*ddm_velocity)->velocity_bins,
                           feature_assembly->velocity_bins, q);
          } else {
            graph.add_flow(doppler->velocity_bins,
                           feature_assembly->velocity_bins, q);
          }
          graph.add_flow(beamscan->aoa_estimates,
                         feature_assembly->aoa_estimates, q);
          graph.add_flow(feature_assembly->points, cluster->points, q);
          graph.add_flow(feature_assembly->points, cluster_report->points, q);
          graph.add_flow(cluster->labels, cluster_report->labels, q);
          graph.add_flow(cluster->sizes, cluster_report->sizes, q);
          graph.add_flow(cluster->labels, reduction->labels, q);
          graph.add_flow(cfar->detection_idx, reduction->detection_idx, q);
          graph.add_flow(feature_assembly->points, reduction->points, q);
          graph.add_flow(reduction->rep_idx, cluster_report->rep_idx, q);
          graph.add_flow(cfar->detection_idx, cluster_report->detection_idx, q);
          graph.add_flow(range->range_bins, cluster_report->range_bins, q);
          if (ddm) {
            graph.add_flow((*ddm_velocity)->velocity_bins,
                           cluster_report->velocity_bins, q);
          } else {
            graph.add_flow(doppler->velocity_bins,
                           cluster_report->velocity_bins, q);
          }

#if defined(HOLOSCAN_RADAR_HAS_X11)
          if (viewer) {
            graph.add_flow(cluster->labels, (*viewer)->labels, q);
            graph.add_flow(reduction->rep_idx, (*viewer)->rep_idx, q);
          }
#endif

          bind_out(options, "feature_assembly", "points");
          bind_out(options, "cluster_dbscan", "labels");
          bind_out(options, "cluster_dbscan", "sizes");
          bind_out(options, "detection_reduction", "rep_idx");
          bind_out(options, "detection_reduction", "reduced_points");
        }

        bind_out(options, "adc_source", "adc_cube");
        bind_out(options, "range_processor", "range_profiles");
        bind_out(options, "range_processor", "range_bins");
        bind_out(options, "clutter_filter", "range_profiles_out");
        bind_out(options, "doppler_processor", "rd_maps");
        bind_out(options, "doppler_processor", "velocity_bins");
        if (ddm) {
          bind_out(options, "ddm_dense_demultiplex", "rd_maps_out");
        } else {
          bind_out(options, "tdm_dense_compensation", "rd_maps_out");
        }
        bind_out(options, "virtual_array_assembly", "rd_maps_out");
        bind_out(options, "cfar_detector", "detection_mask");
        bind_out(options, "cfar_detector", "detection_score");
        bind_out(options, "cfar_detector", "noise_estimate");
        bind_out(options, "cfar_detector", "detection_idx");
        bind_out(options, "cfar_detector", "cfar_power");
        bind_out(options, "snapshot_gather", "snapshot");
        bind_out(options, "snapshot_gather", "is_valid");
        bind_out(options, "beamscan_estimator", "aoa_estimates");
        bind_out(options, "beamscan_estimator", "angle_spectrum");
        if (ddm) {
          bind_out(options, "ddm_velocity_bins", "velocity_bins");
        }
      });
}
