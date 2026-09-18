// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
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
#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/simple/mimo/mimo_ddm_config.hpp"
#include "radar_app/simple/mimo/custom_ops.hpp"

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include "radar_app/viz/fmcw_dashboard_graph.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;
namespace mimo = holoscan::radar::app::mimo;
namespace workflow = holoscan::radar::app::mimo_ddm;

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
  return app::run_fmcw_graph_app<workflow::Config>(
      argc, argv, "mimo_ddm", "mimo_ddm_linear.yaml", workflow::load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         workflow::Config& cfg,
         std::shared_ptr<app::CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        // Simulated coded DDM ADC source.
        mimo::MimoDdmAdcSourceConfig source_config;
        source_config.scene = cfg.scene;
        source_config.scene_visibility = cfg.scene_visibility;
        source_config.doppler_offsets = cfg.doppler_offsets;
        if (cfg.correlate) {
          source_config.phase_codes = cfg.ddm_code.phase_codes;
        }
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

        auto source = graph.op<mimo::MimoDdmAdcSourceOp>(
            "adc_source", extents, std::move(source_config));
        bind_outputs(options, "adc_source", {"adc_cube"});

        // Shared-channel range processing before TX separation.
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);
        bind_outputs(options, "range_processor",
                     {"range_profiles", "range_bins"});

        // Form the coded range-Doppler maps.
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        bind_outputs(options, "doppler_processor",
                     {"rd_maps", "velocity_bins"});

        // Separate TX channels by linear band gather or phase-code correlation.
        auto demultiplex = graph.op<radar::DDMDenseDemultiplexOp<float>>(
            "ddm_dense_demultiplex", extents);
        demultiplex.set_params(cfg.demultiplex_params);
        bind_outputs(options, "ddm_dense_demultiplex",
                     {"velocity_bins_out", "rd_maps_out"});

        // Suppress a physical-velocity band after the DDM codes are removed.
        auto notch = graph.op<radar::DopplerNotchOp<float>>(
            "doppler_notch", extents);
        notch.set_params(cfg.notch_params);
        bind_outputs(options, "doppler_notch", {"rd_maps_out"});

        // Flatten TX/RX channels into the physical virtual array.
        auto vaa = graph.op<radar::VirtualArrayAssemblyOp<float>>(
            "virtual_array_assembly", extents);
        vaa.set_params(cfg.vaa_params);
        bind_outputs(options, "virtual_array_assembly", {"rd_maps_out"});

        // Detect after non-coherently integrating the virtual channels.
        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(options, "cfar_detector",
                     {"detection_mask", "detection_score", "noise_estimate",
                      "detection_idx", "cfar_power"});

        // Gather coherent virtual-array snapshots only at detected RD cells.
        auto gather = graph.op<radar::SnapshotGatherOp<float>>(
            "snapshot_gather", extents);
        gather.set_params(cfg.gather_params);
        bind_outputs(options, "snapshot_gather", {"snapshot", "is_valid"});

        // Estimate one sparse angle pair per gathered detection.
        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);
        bind_outputs(options, "beamscan_estimator",
                     {"aoa_estimates", "angle_spectrum"});

        // Assemble physical features, score, and RD-bin clustering columns.
        auto features = graph.op<radar::FeatureAssemblyOp<float>>(
            "feature_assembly", extents, cfg.assembly_params->features);
        bind_outputs(options, "feature_assembly", {"points"});

        // Cluster neighboring detections in range-Doppler bin space.
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
            "cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        bind_outputs(options, "cluster_dbscan", {"labels", "sizes"});

        // Reduce each cluster to its score-weighted physical feature row.
        auto reduction = graph.op<radar::DetectionReductionOp<float>>(
            "detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        bind_outputs(options, "detection_reduction",
                     {"rep_idx", "reduced_points"});

        // Final clustered detection report.
        auto report = graph.op<app::DetectionReportOp>(
            "cluster_report",
            app::DetectionReportOp::Config{
                .mode = app::DetectionReportMode::kClusters,
                .point_index_columns = cfg.report_index_columns,
                .use_reduced_points = true,
                .angle_convention =
                    cfg.beamscan_params->angle_convention,
                .cadence = cadence});

        // Processing and reporting share one source-driven partition.
        auto application = graph.partition("application");
        application.add(source)
            .add(range)
            .add(doppler)
            .add(demultiplex)
            .add(notch)
            .add(vaa)
            .add(cfar)
            .add(gather)
            .add(beamscan)
            .add(features)
            .add(cluster)
            .add(reduction)
            .add(report);

        // DDM front end: coded ADC -> separated virtual-array RD maps. The
        // demultiplexer transforms the same-tick physical velocity axis.
        graph.add_flow(source->adc_cube, range->adc_cube, queue);
        graph.add_flow(range->range_profiles, doppler->range_profiles, queue);
        graph.add_flow(doppler->velocity_bins, demultiplex->velocity_bins,
                       queue);
        graph.add_flow(doppler->rd_maps, demultiplex->rd_maps, queue);
        graph.add_flow(demultiplex->rd_maps_out, notch->rd_maps, queue);
        graph.add_flow(demultiplex->velocity_bins_out,
                       notch->velocity_bins, queue);
        graph.add_flow(notch->rd_maps_out, vaa->rd_maps, queue);

        // Detection-first angle branch. Runtime VAA geometry remains coherent
        // with the snapshots consumed by Beamscan on the same tick.
        graph.add_flow(vaa->rd_maps_out, cfar->rd_maps, queue);
        graph.add_flow(vaa->rd_maps_out, gather->rd_maps, queue);
        graph.add_flow(cfar->detection_idx, gather->detection_idx, queue);
        graph.add_flow(gather->snapshot, beamscan->snapshot, queue);
        graph.add_flow(gather->is_valid, beamscan->is_valid, queue);
        graph.add_flow(vaa->array_geometry, beamscan->array_geometry, queue);

        // Resolve each detection against the post-demultiplex velocity axis
        // while retaining its score and original range-Doppler indices.
        graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(demultiplex->velocity_bins_out,
                       features->velocity_bins, queue);
        graph.add_flow(beamscan->aoa_estimates, features->aoa_estimates, queue);
        graph.add_flow(cfar->detection_score, features->detection_score, queue);

        // Cluster on RD-bin columns and reduce using the score column.
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(features->points, reduction->points, queue);

        // Report clustered detections against their physical coordinate axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->detection_idx, queue);
        graph.add_flow(reduction->reduced_points, report->reduced_points,
                       queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(demultiplex->velocity_bins_out,
                       report->velocity_bins, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          app::viz::add_fmcw_dashboard(
              graph, application, options, cfg.visualization,
              cfg.frame_period, true, cfg.beamscan_params->az_coordinates,
              cfg.beamscan_params->el_coordinates,
              cfg.beamscan_params->angle_convention, cfg.scene.targets,
              cfg.scene_visibility,
              "MIMO DDM",
              app::viz::ClusterDashboardInputs{
                  cfar->cfar_power, cfar->detection_idx, features->points,
                  cluster->labels, cluster->sizes, range->range_bins,
                  demultiplex->velocity_bins_out,
                  reduction->reduced_points});
        }
#endif
      });
}
