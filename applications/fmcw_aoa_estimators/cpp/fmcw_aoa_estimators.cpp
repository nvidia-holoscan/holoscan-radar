// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Interchangeable AoA estimator demo (HSDK 5 Graph):
//
//   SimAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
//     -> SnapshotGather -> EstimatorRouterOp (5 snapshot outs; emit one)
//       -> SnapshotK1Slice -> Beamscan
//       -> MVDR / MUSIC / Root-MUSIC / ESPRIT
//     -> AoaMergeOp (OnAnyInput) -> [FeatureAssembly -> DBSCAN -> Reduction]
//     -> DetectionReportOp / ClusterReportOp
//     [-> RdMapToImageOp(cfar_power) -> X11RdViewerOp]
//
// Graph-native stand-in for Application set_dynamic_flows(): idle estimators
// never receive a snapshot; merge fires on whichever aoa port arrives.
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_aoa_estimators/cpp/fmcw_aoa_estimators

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

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
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/source.hpp"
#include "radar_app/report/report_ops.hpp"
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
      argc, argv, "fmcw_aoa_estimators", "fmcw_aoa_estimators.yaml",
      load_config,
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

        auto cfar =
            graph.op<radar::CFARDetectorOp<float>>("cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);

        auto gather = graph.op<radar::SnapshotGatherOp<float>>(
            "snapshot_gather", extents);
        gather.set_params(cfg.gather_params);

        auto router = graph.op<EstimatorRouterOp>(
            "estimator_router", extents, cfg.router);

        auto k1_slice =
            graph.op<SnapshotK1SliceOp>("snapshot_k1_slice", extents);

        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);

        auto mvdr = graph.op<radar::MVDREstimatorOp<float>>("mvdr_estimator",
                                                            extents);
        mvdr.set_params(cfg.mvdr_params);

        auto music = graph.op<radar::MUSICEstimatorOp<float>>(
            "music_estimator", extents);
        music.set_params(cfg.music_params);

        auto root_music = graph.op<radar::RootMUSICEstimatorOp<float>>(
            "root_music_estimator", extents);
        root_music.set_params(cfg.root_music_params);

        auto esprit = graph.op<radar::ESPRITEstimatorOp<float>>(
            "esprit_estimator", extents);
        esprit.set_params(cfg.esprit_params);

        auto aoa_merge = graph.op<AoaMergeOp>("aoa_merge", extents);

        auto sink = graph.op<DetectionReportOp>(
            "detection_report",
            DetectionReportOp::Config{cadence, /*report_aoa=*/cfg.peak_only});
        auto mask_sink = graph.op<NullSinkOp>("detection_mask_sink");
        auto noise_sink = graph.op<NullSinkOp>("noise_estimate_sink");
        auto power_sink = graph.op<NullSinkOp>("cfar_power_sink");
        auto is_valid_sink = graph.op<NullSinkOp>("is_valid_sink");
        auto estimator_idx_sink = graph.op<NullSinkOp>("estimator_idx_sink");

        // Drain always-emitted estimator side outputs (OnEach sinks idle when
        // that estimator does not run this tick).
        auto beam_spec_sink = graph.op<NullSinkOp>("beamscan_spectrum_sink");
        auto mvdr_spec_sink = graph.op<NullSinkOp>("mvdr_spectrum_sink");
        auto mvdr_doa_sink = graph.op<NullSinkOp>("mvdr_doa_multi_sink");
        auto mvdr_count_sink = graph.op<NullSinkOp>("mvdr_source_count_sink");
        auto mvdr_eig_sink = graph.op<NullSinkOp>("mvdr_eigenvalues_sink");
        auto music_spec_sink = graph.op<NullSinkOp>("music_spectrum_sink");
        auto music_doa_sink = graph.op<NullSinkOp>("music_doa_multi_sink");
        auto music_count_sink = graph.op<NullSinkOp>("music_source_count_sink");
        auto music_eig_sink = graph.op<NullSinkOp>("music_eigenvalues_sink");
        auto rm_doa_sink = graph.op<NullSinkOp>("root_music_doa_multi_sink");
        auto rm_count_sink =
            graph.op<NullSinkOp>("root_music_source_count_sink");
        auto rm_eig_sink = graph.op<NullSinkOp>("root_music_eigenvalues_sink");
        auto esp_doa_sink = graph.op<NullSinkOp>("esprit_doa_multi_sink");
        auto esp_count_sink = graph.op<NullSinkOp>("esprit_source_count_sink");
        auto esp_eig_sink = graph.op<NullSinkOp>("esprit_eigenvalues_sink");

        graph.add_flow(source->adc_cube, range->adc_cube, q);
        graph.add_flow(range->range_profiles, clutter->range_profiles, q);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, q);
        graph.add_flow(doppler->rd_maps, gather->rd_maps, q);
        graph.add_flow(cfar->detection_idx, gather->detection_idx, q);
        graph.add_flow(gather->snapshot, router->snapshot_in, q);
        graph.add_flow(gather->is_valid, is_valid_sink->input, q);

        graph.add_flow(router->snapshot_beamscan, k1_slice->snapshot_in, q);
        graph.add_flow(k1_slice->snapshot_out, beamscan->snapshot, q);
        graph.add_flow(router->snapshot_mvdr, mvdr->snapshot, q);
        graph.add_flow(router->snapshot_music, music->snapshot, q);
        graph.add_flow(router->snapshot_root_music, root_music->snapshot, q);
        graph.add_flow(router->snapshot_esprit, esprit->snapshot, q);

        graph.add_flow(beamscan->aoa_estimates, aoa_merge->aoa_beamscan, q);
        graph.add_flow(mvdr->aoa_estimates, aoa_merge->aoa_mvdr, q);
        graph.add_flow(music->aoa_estimates, aoa_merge->aoa_music, q);
        graph.add_flow(root_music->aoa_estimates, aoa_merge->aoa_root_music, q);
        graph.add_flow(esprit->aoa_estimates, aoa_merge->aoa_esprit, q);

        graph.add_flow(beamscan->angle_spectrum, beam_spec_sink->input, q);
        graph.add_flow(mvdr->angle_spectrum, mvdr_spec_sink->input, q);
        graph.add_flow(mvdr->doa_multi, mvdr_doa_sink->input, q);
        graph.add_flow(mvdr->source_count, mvdr_count_sink->input, q);
        graph.add_flow(mvdr->eigenvalues, mvdr_eig_sink->input, q);
        graph.add_flow(music->angle_spectrum, music_spec_sink->input, q);
        graph.add_flow(music->doa_multi, music_doa_sink->input, q);
        graph.add_flow(music->source_count, music_count_sink->input, q);
        graph.add_flow(music->eigenvalues, music_eig_sink->input, q);
        graph.add_flow(root_music->doa_multi, rm_doa_sink->input, q);
        graph.add_flow(root_music->source_count, rm_count_sink->input, q);
        graph.add_flow(root_music->eigenvalues, rm_eig_sink->input, q);
        graph.add_flow(esprit->doa_multi, esp_doa_sink->input, q);
        graph.add_flow(esprit->source_count, esp_count_sink->input, q);
        graph.add_flow(esprit->eigenvalues, esp_eig_sink->input, q);

        graph.add_flow(cfar->detection_idx, sink->detection_idx, q);
        graph.add_flow(cfar->detection_score, sink->detection_score, q);
        graph.add_flow(range->range_bins, sink->range_bins, q);
        graph.add_flow(doppler->velocity_bins, sink->velocity_bins, q);
        graph.add_flow(cfar->detection_mask, mask_sink->input, q);
        graph.add_flow(cfar->noise_estimate, noise_sink->input, q);
        // Always drain cfar_power (EA isolated-vertex partition trap); same for
        // the merge's estimator index, which only the viewer displays.
        graph.add_flow(cfar->cfar_power, power_sink->input, q);
        graph.add_flow(aoa_merge->estimator_idx, estimator_idx_sink->input, q);

#if defined(HOLOSCAN_RADAR_HAS_X11)
        std::optional<holoscan::OperatorHandle<X11RdViewerOp>> viewer;
        if (cfg.visualize) {
          X11RdViewerConfig viz_cfg;
          viz_cfg.title = "fmcw_aoa_estimators - Range-Doppler (NCI)";
          viz_cfg.width = static_cast<int>(cfg.viz.window_width);
          viz_cfg.height = static_cast<int>(cfg.viz.window_height);
          viz_cfg.overlay_detections = true;
          viz_cfg.overlay_clusters = !cfg.peak_only;
          viz_cfg.plots_enabled = cfg.plots.enable;
          viz_cfg.plot_window = cfg.plots.window;
          viz_cfg.frame_period = cfg.frame_period;
          viz_cfg.marker_radius = cfg.viz.marker_radius;
          viz_cfg.marker_outline = cfg.viz.marker_outline;
          viz_cfg.overlay_boresight = true;
          viz_cfg.overlay_label = true;
          viz_cfg.label_caption = "estimator";
          viz_cfg.label_names = cfg.router.display_names;
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
          graph.add_flow(doppler->velocity_bins, (*viewer)->velocity_bins, q);
          graph.add_flow(aoa_merge->aoa_estimates, (*viewer)->aoa_estimates, q);
          graph.add_flow(aoa_merge->estimator_idx, (*viewer)->label_idx, q);
          bind_out(options, "rd_map_to_image", "rd_image");
          HOLOSCAN_LOG_INFO(
              "fmcw_aoa_estimators: X11 RD live viewer enabled ({}x{})",
              viz_cfg.width, viz_cfg.height);
        }
#else
        if (cfg.visualize) {
          HOLOSCAN_LOG_WARN(
              "fmcw_aoa_estimators: visualize requested but X11 unavailable; "
              "running log-only");
          cfg.visualize = false;
        }
#endif

        if (cfg.peak_only) {
          graph.add_flow(aoa_merge->aoa_estimates, sink->aoa_estimates, q);
        } else {
          auto feature_assembly =
              graph.op<radar::FeatureAssemblyOp<float>>("feature_assembly",
                                                        extents);
          radar::FeatureAssemblyParams feature_params;
          feature_params.features = {
              radar::AssemblyFeature::kRange,
              radar::AssemblyFeature::kVelocity,
              radar::AssemblyFeature::kAz,
              radar::AssemblyFeature::kEl};
          feature_assembly.set_params(std::move(feature_params));
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
          graph.add_flow(doppler->velocity_bins,
                         feature_assembly->velocity_bins, q);
          graph.add_flow(aoa_merge->aoa_estimates,
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
          graph.add_flow(doppler->velocity_bins, cluster_report->velocity_bins,
                         q);

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
        }

        bind_out(options, "adc_source", "adc_cube");
        bind_out(options, "range_processor", "range_profiles");
        bind_out(options, "range_processor", "range_bins");
        bind_out(options, "clutter_filter", "range_profiles_out");
        bind_out(options, "doppler_processor", "rd_maps");
        bind_out(options, "doppler_processor", "velocity_bins");
        bind_out(options, "cfar_detector", "detection_idx");
        bind_out(options, "cfar_detector", "detection_score");
        bind_out(options, "cfar_detector", "detection_mask");
        bind_out(options, "cfar_detector", "noise_estimate");
        bind_out(options, "cfar_detector", "cfar_power");
        bind_out(options, "snapshot_gather", "snapshot");
        bind_out(options, "snapshot_gather", "is_valid");
        bind_out(options, "estimator_router", "snapshot_beamscan");
        bind_out(options, "estimator_router", "snapshot_mvdr");
        bind_out(options, "estimator_router", "snapshot_music");
        bind_out(options, "estimator_router", "snapshot_root_music");
        bind_out(options, "estimator_router", "snapshot_esprit");
        bind_out(options, "snapshot_k1_slice", "snapshot_out");
        bind_out(options, "beamscan_estimator", "aoa_estimates");
        bind_out(options, "beamscan_estimator", "angle_spectrum");
        bind_out(options, "mvdr_estimator", "aoa_estimates");
        bind_out(options, "mvdr_estimator", "angle_spectrum");
        bind_out(options, "mvdr_estimator", "doa_multi");
        bind_out(options, "mvdr_estimator", "source_count");
        bind_out(options, "mvdr_estimator", "eigenvalues");
        bind_out(options, "music_estimator", "aoa_estimates");
        bind_out(options, "music_estimator", "angle_spectrum");
        bind_out(options, "music_estimator", "doa_multi");
        bind_out(options, "music_estimator", "source_count");
        bind_out(options, "music_estimator", "eigenvalues");
        bind_out(options, "root_music_estimator", "aoa_estimates");
        bind_out(options, "root_music_estimator", "doa_multi");
        bind_out(options, "root_music_estimator", "source_count");
        bind_out(options, "root_music_estimator", "eigenvalues");
        bind_out(options, "esprit_estimator", "aoa_estimates");
        bind_out(options, "esprit_estimator", "doa_multi");
        bind_out(options, "esprit_estimator", "source_count");
        bind_out(options, "esprit_estimator", "eigenvalues");
        bind_out(options, "aoa_merge", "aoa_estimates");
        bind_out(options, "aoa_merge", "estimator_idx");
      });
}
