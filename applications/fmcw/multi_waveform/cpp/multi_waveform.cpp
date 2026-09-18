// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <utility>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)
#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "detection/detection_reduction_op/detection_reduction_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/multi_profile_source.hpp"
#include "radar_app/report/report_ops.hpp"

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include "radar_app/viz/fmcw_dashboard_graph.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;
namespace workflow = holoscan::radar::app::multi_waveform;

namespace {

void bind_outputs(holoscan::CompileOptions& options, const char* op,
                  std::initializer_list<const char*> ports) {
  for (const char* port : ports) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{
            .operator_path = op, .output_port = port,
            .device = holoscan::DeviceId{0}});
  }
}

}  // namespace

int main(int argc, char** argv) {
  return app::run_fmcw_graph_app<workflow::Config>(
      argc, argv, "multi_waveform", "multi_waveform.yaml",
      workflow::load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options,
         workflow::Config& cfg,
         std::shared_ptr<app::CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        const holoscan::ConnectionOptions waveform_queue{
            .queue_depth = cfg.max_frames_in_flight + 1U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        auto coordinator = std::make_shared<app::TransmitFrameCoordinator>(
            cfg.max_frames_in_flight);
        auto link = std::make_shared<app::SimulatedTdmLink>(
            app::SimulatedTdmLinkConfig{
                .profiles = cfg.profiles,
                .scene = cfg.scene,
                .scene_visibility = cfg.scene_visibility,
                .tx_schedule = cfg.tx_schedule,
                .frame_period = cfg.frame_period,
                .polls_before_ready = cfg.simulated_receive_polls});
        app::FMCWTransmitSourceConfig tx_config{
            .profiles = cfg.profiles,
            .profile_sequence = cfg.profile_sequence,
            .frame_period = cfg.frame_period,
            .num_frames = cfg.num_frames,
            .driver = link,
            .coordinator = coordinator,
        };
        auto tx = graph.op<app::FMCWTransmitSourceOp>(
            "waveform_source", std::move(tx_config));

        std::shared_ptr<app::CadenceMeter> cadence;
        if (cfg.meter_cadence) {
          cadence = std::make_shared<app::CadenceMeter>();
          cadence->frame_period_s = cfg.frame_period;
          cadence->log_every = cfg.meter_log_every;
          *cadence_out = cadence;
        }
        app::CorrelatedMimoTdmAdcSourceConfig rx_config{
            .poll_period = cfg.rx_poll_period,
            .frame_period = cfg.frame_period,
            .num_frames = cfg.num_frames,
            .pending_capacity = cfg.max_frames_in_flight,
            .driver = link,
            .coordinator = coordinator,
            .cadence = cadence,
        };
        auto source = graph.op<app::CorrelatedMimoTdmAdcSourceOp>(
            "adc_source", extents, std::move(rx_config));
        bind_outputs(options, "adc_source", {"adc_cube"});

        auto deinterleave = graph.op<radar::MIMODeinterleaveOp<float>>(
            "mimo_deinterleave", extents);
        deinterleave.set_params(cfg.deinterleave_params);
        bind_outputs(options, "mimo_deinterleave", {"adc_cube_out"});
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>(
            "range_processor", extents);
        range.set_params(cfg.range_params);
        bind_outputs(options, "range_processor", {"range_profiles", "range_bins"});
        auto clutter = graph.op<radar::ClutterFilterOp<float>>(
            "clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);
        bind_outputs(options, "clutter_filter", {"range_profiles_out"});
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>(
            "doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        bind_outputs(options, "doppler_processor", {"rd_maps", "velocity_bins"});
        auto cfar = graph.op<radar::CFARDetectorOp<float>>(
            "cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(options, "cfar_detector",
                     {"detection_mask", "detection_score", "noise_estimate",
                      "detection_idx", "cfar_power"});
        auto compensation = graph.op<radar::TDMSparseCompensationOp<float>>(
            "tdm_sparse_compensation", extents);
        compensation.set_params(cfg.compensation_params);
        bind_outputs(options, "tdm_sparse_compensation", {"snapshot", "is_valid"});
        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>(
            "beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);
        bind_outputs(options, "beamscan_estimator", {"aoa_estimates", "angle_spectrum"});
        auto features = graph.op<radar::FeatureAssemblyOp<float>>(
            "feature_assembly", extents, cfg.assembly_params->features);
        bind_outputs(options, "feature_assembly", {"points"});
        auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
            "cluster_dbscan", extents);
        cluster.set_params(cfg.dbscan_params);
        bind_outputs(options, "cluster_dbscan", {"labels", "sizes"});
        auto reduction = graph.op<radar::DetectionReductionOp<float>>(
            "detection_reduction", extents);
        reduction.set_params(cfg.reduction_params);
        bind_outputs(options, "detection_reduction", {"rep_idx", "reduced_points"});
        auto report = graph.op<app::DetectionReportOp>(
            "cluster_report", app::DetectionReportOp::Config{
                .mode = app::DetectionReportMode::kClusters,
                .point_index_columns = cfg.report_index_columns,
                .use_reduced_points = true,
                .angle_convention = cfg.beamscan_params->angle_convention,
                .cadence = cadence});

        graph.partition("transmit").add(tx);
        graph.partition("receive").add(source);
        auto processing = graph.partition("processing");
        processing.add(deinterleave).add(range).add(clutter).add(doppler)
            .add(cfar).add(compensation).add(beamscan).add(features)
            .add(cluster).add(reduction).add(report);

        graph.add_flow(source->adc_cube, deinterleave->adc_cube, queue);
        graph.add_flow(deinterleave->adc_cube_out, range->adc_cube, queue);
        graph.add_flow(tx->waveform, range->waveform_in, waveform_queue);
        graph.add_flow(range->range_profiles, clutter->range_profiles, queue);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, queue);
        graph.add_flow(tx->waveform, doppler->waveform_in, waveform_queue);
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, queue);
        graph.add_flow(doppler->rd_maps, compensation->rd_maps, queue);
        graph.add_flow(cfar->detection_idx, compensation->detection_idx, queue);
        graph.add_flow(tx->waveform, compensation->waveform_in, waveform_queue);
        graph.add_flow(compensation->snapshot, beamscan->snapshot, queue);
        graph.add_flow(compensation->is_valid, beamscan->is_valid, queue);
        graph.add_flow(compensation->array_geometry, beamscan->array_geometry, queue);
        graph.add_flow(cfar->detection_idx, features->detection_idx, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, features->velocity_bins, queue);
        graph.add_flow(beamscan->aoa_estimates, features->aoa_estimates, queue);
        graph.add_flow(cfar->detection_score, features->detection_score, queue);
        graph.add_flow(features->points, cluster->points, queue);
        graph.add_flow(cluster->labels, reduction->labels, queue);
        graph.add_flow(cfar->detection_idx, reduction->detection_idx, queue);
        graph.add_flow(features->points, reduction->points, queue);
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(cluster->labels, report->labels, queue);
        graph.add_flow(cluster->sizes, report->sizes, queue);
        graph.add_flow(reduction->rep_idx, report->detection_idx, queue);
        graph.add_flow(reduction->reduced_points, report->reduced_points, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          app::viz::add_fmcw_dashboard(
              graph, processing, options, cfg.visualization, cfg.frame_period,
              true, cfg.beamscan_params->az_coordinates,
              cfg.beamscan_params->el_coordinates,
              cfg.beamscan_params->angle_convention, cfg.scene.targets,
              cfg.scene_visibility,
              "Multi-Waveform",
              app::viz::ProfiledClusterDashboardInputs{
                  cfar->cfar_power, cfar->detection_idx, features->points,
                  cluster->labels, cluster->sizes, range->range_bins,
                  doppler->velocity_bins, reduction->reduced_points, tx->waveform,
                  cfg.profiles[cfg.visualization_profile_index].waveform,
                  cfg.visualization_profile_max_gap_frames});
        }
#endif
      });
}
