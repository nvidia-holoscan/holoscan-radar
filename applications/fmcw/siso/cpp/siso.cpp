// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/app/source.hpp"
#include "radar_app/report/report_ops.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include "radar_app/viz/fmcw_dashboard_graph.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;
namespace workflow = holoscan::radar::app::siso;

namespace {

void bind_output(holoscan::CompileOptions& options, const char* op, const char* port) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = op,
      .output_port = port,
      .device = app::application_device(),
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
  return app::run_fmcw_graph_app<workflow::Config>(
      argc, argv, "siso", "siso.yaml", workflow::load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options, workflow::Config& cfg,
         std::shared_ptr<app::CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        app::SimAdcSourceConfig source_config;
        source_config.scene = cfg.scene;
        source_config.scene_visibility = cfg.scene_visibility;
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

        // Simulated single-channel ADC source.
        auto source =
            graph.op<app::SimAdcSourceOp>("adc_source", extents, std::move(source_config));
        bind_outputs(options, "adc_source", {"adc_cube"});

        // Range processing.
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>("range_processor", extents);
        range.set_params(cfg.range_params);
        bind_outputs(options, "range_processor", {"range_profiles", "range_bins"});

        // Static-clutter suppression.
        auto clutter = graph.op<radar::ClutterFilterOp<float>>("clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);
        bind_outputs(options, "clutter_filter", {"range_profiles_out"});

        // Doppler processing.
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>("doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        bind_outputs(options, "doppler_processor", {"rd_maps", "velocity_bins"});

        // CFAR detection on the rank-2 range-Doppler map.
        auto cfar = graph.op<radar::CFARDetectorOp<float>>("cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(
            options, "cfar_detector",
            {"detection_mask", "detection_score", "noise_estimate", "detection_idx", "cfar_power"});

        // Assemble physical range/velocity plus RD-bin clustering columns.
        auto features = graph.op<radar::FeatureAssemblyOp<float>>("feature_assembly", extents,
                                                                  cfg.assembly_params->features);
        bind_outputs(options, "feature_assembly", {"points"});

        // Group neighboring range-Doppler detections.
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>("cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        bind_outputs(options, "cluster_dbscan", {"labels", "sizes"});

        // Reduce each cluster to its configured representative feature row.
        auto reduction =
            graph.op<radar::DetectionReductionOp<float>>("detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        bind_outputs(options, "detection_reduction", {"rep_idx", "reduced_points"});

        // Final clustered detection report.
        auto report = graph.op<app::DetectionReportOp>(
            "cluster_report",
            app::DetectionReportOp::Config{.mode = app::DetectionReportMode::kClusters,
                                           .use_reduced_points = true,
                                           .cadence = cadence});

        // Processing and reporting share one source-driven partition.
        auto application = graph.partition("application");
        application.add(source)
            .add(range)
            .add(clutter)
            .add(doppler)
            .add(cfar)
            .add(features)
            .add(cluster)
            .add(reduction)
            .add(report);

        // FMCW front end: ADC cube -> range-Doppler map -> detections.
        graph.add_flow(source->adc_cube, range->adc_cube, queue);
        graph.add_flow(range->range_profiles, clutter->range_profiles, queue);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, queue);
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, queue);

        // Convert detected bin coordinates to physical range and velocity.
        graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, features->velocity_bins, queue);

        // Cluster the complete feature rows and select one result per cluster.
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(features->points, reduction->points, queue);

        // Report clusters against the same physical coordinate axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->detection_idx, queue);
        graph.add_flow(reduction->reduced_points, report->reduced_points, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          const std::vector<double> no_angle_grid;
          app::viz::add_fmcw_dashboard(
              graph, application, options, cfg.visualization, cfg.frame_period, false,
              no_angle_grid, no_angle_grid, radar::AngleConvention{}, cfg.scene.targets,
              cfg.scene_visibility, "SISO",
              app::viz::ClusterDashboardInputs{cfar->cfar_power, cfar->detection_idx,
                                               features->points, cluster->labels, cluster->sizes,
                                               range->range_bins, doppler->velocity_bins,
                                               reduction->reduced_points},
              app::application_description("siso"));
        }
#endif
      });
}
