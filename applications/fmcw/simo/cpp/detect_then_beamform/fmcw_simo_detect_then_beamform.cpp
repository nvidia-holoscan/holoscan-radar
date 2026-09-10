// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <utility>

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

        auto source = graph.op<app::SimAdcSourceOp>(
            "adc_source", extents, std::move(source_config));
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);
        auto clutter = graph.op<radar::ClutterFilterOp<float>>(
            "clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        auto gather = graph.op<radar::SnapshotGatherOp<float>>(
            "snapshot_gather", extents);
        gather.set_params(cfg.gather_params);
        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);
        auto features = graph.op<radar::FeatureAssemblyOp<float>>(
            "feature_assembly", extents);
        radar::FeatureAssemblyParams feature_params;
        feature_params.features = {
            radar::AssemblyFeature::kRange,
            radar::AssemblyFeature::kVelocity,
            radar::AssemblyFeature::kAzEstimate,
            radar::AssemblyFeature::kElEstimate};
        features.set_params(std::move(feature_params));
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
            "cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        auto reduction = graph.op<radar::DetectionReductionOp<float>>(
            "detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        auto report = graph.op<app::ClusterReportOp>(
            "cluster_report",
            app::ClusterReportOp::Config{
                /*report_detection_idx=*/true, {}, {}, cadence});

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

        // Cluster detections and choose one representative cell per cluster.
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(features->points, reduction->points, queue);

        // Report the clustered detections with their physical coordinate axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->rep_idx, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);
        graph.add_flow(cfar->detection_idx, report->detection_idx, queue);

        bind_output(options, "adc_source", "adc_cube");
        bind_output(options, "range_processor", "range_profiles");
        bind_output(options, "range_processor", "range_bins");
        bind_output(options, "clutter_filter", "range_profiles_out");
        bind_output(options, "doppler_processor", "rd_maps");
        bind_output(options, "doppler_processor", "velocity_bins");
        bind_output(options, "cfar_detector", "detection_mask");
        bind_output(options, "cfar_detector", "detection_score");
        bind_output(options, "cfar_detector", "noise_estimate");
        bind_output(options, "cfar_detector", "detection_idx");
        bind_output(options, "cfar_detector", "cfar_power");
        bind_output(options, "snapshot_gather", "snapshot");
        bind_output(options, "snapshot_gather", "is_valid");
        bind_output(options, "beamscan_estimator", "aoa_estimates");
        bind_output(options, "beamscan_estimator", "angle_spectrum");
        bind_output(options, "feature_assembly", "points");
        bind_output(options, "cluster_dbscan", "labels");
        bind_output(options, "cluster_dbscan", "sizes");
        bind_output(options, "detection_reduction", "rep_idx");
      });
}
