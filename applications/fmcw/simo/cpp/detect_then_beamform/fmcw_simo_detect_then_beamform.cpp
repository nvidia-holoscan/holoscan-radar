// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
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
#include "config.hpp"  // NOLINT(build/include_subdir): app-private header

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include "viz/simo_viz_graph.hpp"  // NOLINT(build/include_subdir)
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;
namespace simo = holoscan::radar::app::simo;

namespace {

void bind_output(holoscan::CompileOptions& options, const char* op,
                 const char* port) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = op,
          .output_port = port,
          .device = holoscan::DeviceId{0},
      });
}

void bind_outputs(holoscan::CompileOptions& options, const char* op,
                  std::initializer_list<const char*> ports) {
  for (const char* port : ports) {
    bind_output(options, op, port);
  }
}

}  // namespace

int main(int argc, char** argv) {
  return app::run_fmcw_graph_app<simo::DetectThenBeamformConfig>(
      argc, argv, "fmcw_simo_detect_then_beamform", "default.yaml",
      simo::load_detect_then_beamform_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         simo::DetectThenBeamformConfig& cfg,
         std::shared_ptr<app::CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        app::SimAdcSourceConfig source_config;
        source_config.scene = cfg.scene;
        source_config.frame_period = cfg.frame_period;
        source_config.num_frames = cfg.num_frames;
        std::shared_ptr<app::CadenceMeter> cadence;
        if (cfg.meter_cadence) {
          cadence = std::make_shared<app::CadenceMeter>();
          cadence->frame_period_s = cfg.frame_period;
          cadence->log_every = cfg.meter_log_every;
          source_config.cadence = cadence;
          *cadence_out = cadence;
        }

        // Simulated ADC source.
        auto source = graph.op<app::SimAdcSourceOp>(
            "adc_source", extents, std::move(source_config));
        bind_outputs(options, "adc_source", {"adc_cube"});

        // Range processing.
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);
        bind_outputs(options, "range_processor",
                     {"range_profiles", "range_bins"});

        // Static-clutter suppression.
        auto clutter = graph.op<radar::ClutterFilterOp<float>>(
            "clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);
        bind_outputs(options, "clutter_filter", {"range_profiles_out"});

        // Doppler processing.
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        bind_outputs(options, "doppler_processor",
                     {"rd_maps", "velocity_bins"});

        // Receiver-integrated CFAR detection.
        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(options, "cfar_detector",
                     {"detection_mask", "detection_score", "noise_estimate",
                      "detection_idx", "cfar_power"});

        // Sparse channel-snapshot gathering.
        auto gather = graph.op<radar::SnapshotGatherOp<float>>(
            "snapshot_gather", extents);
        gather.set_params(cfg.gather_params);
        bind_outputs(options, "snapshot_gather", {"snapshot", "is_valid"});

        // Per-detection angle estimation.
        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);
        bind_outputs(options, "beamscan_estimator",
                     {"aoa_estimates", "angle_spectrum"});

        // Physical feature assembly.
        std::vector<radar::AssemblyFeature> feature_config{
            radar::AssemblyFeature::kRange,
            radar::AssemblyFeature::kVelocity,
            radar::AssemblyFeature::kAzEstimate,
            radar::AssemblyFeature::kElEstimate};
        auto features = graph.op<radar::FeatureAssemblyOp<float>>(
            "feature_assembly", extents, std::move(feature_config));
        bind_outputs(options, "feature_assembly", {"points"});

        // Row-aligned reduction features add CFAR score without changing the
        // four-dimensional DBSCAN metric.
        std::vector<radar::AssemblyFeature> reduction_feature_config{
            radar::AssemblyFeature::kRange,
            radar::AssemblyFeature::kVelocity,
            radar::AssemblyFeature::kAzEstimate,
            radar::AssemblyFeature::kElEstimate,
            radar::AssemblyFeature::kScore};
        auto reduction_features = graph.op<radar::FeatureAssemblyOp<float>>(
            "reduction_feature_assembly", extents,
            std::move(reduction_feature_config));
        bind_outputs(options, "reduction_feature_assembly", {"points"});

        // Detection clustering.
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
            "cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        bind_outputs(options, "cluster_dbscan", {"labels", "sizes"});

        // Cluster representative selection.
        auto reduction = graph.op<radar::DetectionReductionOp<float>>(
            "detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        bind_outputs(options, "detection_reduction",
                     {"rep_idx", "reduced_points"});

        // Final log report.
        auto report = graph.op<app::ClusterReportOp>(
            "cluster_report",
            app::ClusterReportOp::Config{
                /*report_detection_idx=*/true, {}, {}, cadence});

        // Processing and reporting share one source-driven partition.
        auto application = graph.partition("application");
        application.add(source)
            .add(range)
            .add(clutter)
            .add(doppler)
            .add(cfar)
            .add(gather)
            .add(beamscan)
            .add(features)
            .add(reduction_features)
            .add(cluster)
            .add(reduction)
            .add(report);

        // FMCW front end: ADC cube -> per-channel range-Doppler maps.
        graph.add_flow(source->adc_cube, range->adc_cube, queue);
        graph.add_flow(range->range_profiles, clutter->range_profiles, queue);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles,
                       queue);

        // Detection-first branch: detect after folding receiver power, while
        // retaining the coherent channel maps for sparse snapshot gathering.
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, queue);
        graph.add_flow(doppler->rd_maps, gather->rd_maps, queue);
        graph.add_flow(cfar->detection_idx, gather->detection_idx, queue);
        graph.add_flow(gather->snapshot, beamscan->snapshot, queue);
        graph.add_flow(gather->is_valid, beamscan->is_valid, queue);

        // Assemble physical range, velocity, azimuth, and elevation features
        // for each detected cell.
        graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, features->velocity_bins, queue);
        graph.add_flow(beamscan->aoa_estimates, features->aoa_estimates, queue);
        graph.add_flow(cfar->detection_idx,
                       reduction_features->detection_idx, queue);
        graph.add_flow(range->range_bins, reduction_features->range_bins,
                       queue);
        graph.add_flow(doppler->velocity_bins,
                       reduction_features->velocity_bins, queue);
        graph.add_flow(beamscan->aoa_estimates,
                       reduction_features->aoa_estimates, queue);
        graph.add_flow(cfar->detection_score,
                       reduction_features->detection_score, queue);

        // Cluster detections and choose one representative cell per cluster.
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(reduction_features->points, reduction->points, queue);

        // Report the clustered detections with their physical coordinate axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->rep_idx, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);
        graph.add_flow(cfar->detection_idx, report->detection_idx, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          simo::viz::add_simo_visualization(
              graph, application, options, cfg.visualization,
              cfg.frame_period, cfg.beamscan_params->az_angles,
              cfg.beamscan_params->el_angles, cfg.scene.targets,
              "Detect then beamform",
              simo::viz::SimoVizInputs{
                  cfar->cfar_power, cfar->detection_idx, features->points,
                  cluster->labels, cluster->sizes, range->range_bins,
                  doppler->velocity_bins, reduction->reduced_points});
        }
#endif
      });
}
