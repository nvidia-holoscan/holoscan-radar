// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <utility>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"
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
  return app::run_fmcw_graph_app<simo::BeamformThenDetectConfig>(
      argc, argv, "fmcw_simo_beamform_then_detect", "default.yaml",
      simo::load_beamform_then_detect_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         simo::BeamformThenDetectConfig& cfg,
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

        // Dense angle beamforming.
        auto beamformer = graph.op<radar::PhaseShiftBeamformerOp<float>>(
            "phase_shift_beamformer", extents);
        beamformer.set_params(cfg.beamformer_params);
        bind_outputs(options, "phase_shift_beamformer",
                     {"rda_cube", "az_grid", "el_grid"});

        // Per-angle-plane CFAR detection.
        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", cfg.cfar_extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(options, "cfar_detector",
                     {"detection_mask", "detection_score", "noise_estimate",
                      "detection_idx", "cfar_power"});

        // Physical feature assembly.
        auto features = graph.op<radar::FeatureAssemblyOp<float>>(
            "feature_assembly", extents);
        radar::FeatureAssemblyParams feature_params;
        feature_params.features = {
            radar::AssemblyFeature::kRange,
            radar::AssemblyFeature::kVelocity,
            radar::AssemblyFeature::kAz,
            radar::AssemblyFeature::kEl};
        features.set_params(std::move(feature_params));
        bind_outputs(options, "feature_assembly", {"points"});

        // Detection clustering.
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
            "cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        bind_outputs(options, "cluster_dbscan", {"labels", "sizes"});

        // Cluster representative selection.
        auto reduction = graph.op<radar::DetectionReductionOp<float>>(
            "detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        bind_outputs(options, "detection_reduction", {"rep_idx"});

        // Final log report.
        auto report = graph.op<app::ClusterReportOp>(
            "cluster_report",
            app::ClusterReportOp::Config{
                /*report_detection_idx=*/false,
                cfg.beamformer_params->az_angles,
                cfg.beamformer_params->el_angles,
                cadence});

        // Processing and reporting share one source-driven partition.
        auto application = graph.partition("application");
        application.add(source)
            .add(range)
            .add(clutter)
            .add(doppler)
            .add(beamformer)
            .add(cfar)
            .add(features)
            .add(cluster)
            .add(reduction)
            .add(report);

        // FMCW front end: ADC cube -> per-channel range-Doppler maps.
        graph.add_flow(source->adc_cube, range->adc_cube, queue);
        graph.add_flow(range->range_profiles, clutter->range_profiles, queue);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles,
                       queue);

        // Beamforming-first branch: form the complete range-Doppler-angle
        // cube, then detect independently across its angle planes.
        graph.add_flow(doppler->rd_maps, beamformer->rd_maps, queue);
        graph.add_flow(beamformer->rda_cube, cfar->rd_maps, queue);

        // Preserve every detected angle/range/Doppler coordinate and use the
        // beamformer's effective grids to resolve physical angle features.
        graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, features->velocity_bins, queue);
        graph.add_flow(beamformer->az_grid, features->az_angles, queue);
        graph.add_flow(beamformer->el_grid, features->el_angles, queue);

        // Cluster physical range, velocity, azimuth, and elevation features,
        // then choose one real detection cell per cluster.
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(features->points, reduction->points, queue);

        // Report clustered detections with physical range/velocity axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->rep_idx, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          simo::viz::add_simo_visualization(
              graph, application, options, extents, cfg.visualization,
              cfg.frame_period, "Beamform then detect",
              simo::viz::SimoVizInputs{
                  cfar->cfar_power, cfar->detection_idx, cluster->labels,
                  reduction->rep_idx, range->range_bins,
                  doppler->velocity_bins});
        }
#endif
      });
}
