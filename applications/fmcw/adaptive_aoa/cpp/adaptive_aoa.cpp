// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include <holoscan/radar/angle_estimation/beamscan_estimator_op.hpp>
#include <holoscan/radar/angle_estimation/esprit_estimator_op.hpp>
#include <holoscan/radar/angle_estimation/rank_one_residual_screen_op.hpp>
#include <holoscan/radar/angle_estimation/music_estimator_op.hpp>
#include <holoscan/radar/angle_estimation/mvdr_estimator_op.hpp>
#include <holoscan/radar/angle_estimation/root_music_estimator_op.hpp>
#include "config.hpp"  // NOLINT(build/include_subdir)
#include <holoscan/radar/detection/cfar_detector_op.hpp>
#include <holoscan/radar/detection/feature_assembly_op.hpp>
#include <holoscan/radar/detection/multi_source_expansion_op.hpp>
#include <holoscan/radar/fmcw/clutter_filter_op.hpp>
#include <holoscan/radar/fmcw/fmcw_doppler_processor_op.hpp>
#include <holoscan/radar/fmcw/fmcw_range_processor_op.hpp>
#include <holoscan/radar/fmcw/mimo_deinterleave_op.hpp>
#include <holoscan/radar/fmcw/tdm_sparse_compensation_op.hpp>
#include "radar_app/app/graph_driver.hpp"
#include "radar_app/app/gpu_placement.hpp"
#include "radar_app/report/report_ops.hpp"
#include "radar_app/fmcw/mimo_sources.hpp"

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
#include "radar_app/viz/fmcw_dashboard_graph.hpp"
#endif

namespace radar = holoscan::radar;
namespace app = holoscan::radar::app;
namespace adaptive = holoscan::radar::app::adaptive_aoa;
namespace mimo = holoscan::radar::app::mimo;

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
  return app::run_fmcw_graph_app<adaptive::Config>(
      argc, argv, "adaptive_aoa", "adaptive_aoa.yaml", adaptive::load_config,
      [](holoscan::Graph& graph, holoscan::CompileOptions& options, adaptive::Config& cfg,
         std::shared_ptr<app::CadenceMeter>* cadence_out) {
        const holoscan::ConnectionOptions queue{.queue_depth = 4U};
        const radar::MaxTensorExtents& extents = cfg.extents;

        // Simulated interleaved TDM ADC source.
        mimo::MimoTdmAdcSourceConfig source_config;
        source_config.scene = cfg.scene;
        source_config.scene_visibility = cfg.scene_visibility;
        source_config.tx_schedule = cfg.tx_schedule;
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
        auto source =
            graph.op<mimo::MimoTdmAdcSourceOp>("adc_source", extents, std::move(source_config));
        bind_outputs(options, "adc_source", {"adc_cube"});

        // Restore the explicit TX axis from the interleaved slow-time stream.
        auto deinterleave =
            graph.op<radar::MIMODeinterleaveOp<float>>("mimo_deinterleave", extents);
        deinterleave.set_params(cfg.deinterleave_params);
        bind_outputs(options, "mimo_deinterleave", {"adc_cube_out"});
        // Per-TX range processing.
        auto range = graph.op<radar::FMCWRangeProcessorOp<float>>("range_processor", extents);
        range.set_params(cfg.range_params);
        bind_outputs(options, "range_processor", {"range_profiles", "range_bins"});
        // Static-clutter suppression before Doppler processing.
        auto clutter = graph.op<radar::ClutterFilterOp<float>>("clutter_filter", extents);
        clutter.set_params(cfg.clutter_params);
        bind_outputs(options, "clutter_filter", {"range_profiles_out"});
        // Per-TX Doppler processing on the effective TDM PRI.
        auto doppler = graph.op<radar::FMCWDopplerProcessorOp<float>>("doppler_processor", extents);
        doppler.set_params(cfg.doppler_params);
        bind_outputs(options, "doppler_processor", {"rd_maps", "velocity_bins"});
        // Peak-only CFAR after non-coherent TX/RX power integration.
        auto cfar = graph.op<radar::CFARDetectorOp<float>>("cfar_detector", extents);
        cfar.set_params(cfg.cfar_params);
        bind_outputs(
            options, "cfar_detector",
            {"detection_mask", "detection_score", "noise_estimate", "detection_idx", "cfar_power"});

        // Gather and compensate a 3x3 multi-look virtual-array snapshot for
        // each detected range-Doppler peak.
        auto compensation =
            graph.op<radar::TDMSparseCompensationOp<float>>("tdm_sparse_compensation", extents);
        compensation.set_params(cfg.compensation_params);
        bind_outputs(options, "tdm_sparse_compensation", {"snapshot", "is_valid"});
        // Conventional angle baseline used for every detection and by the
        // rank-one residual screen.
        auto beamscan = graph.op<radar::BeamscanEstimatorOp<float>>("beamscan_estimator", extents);
        beamscan.set_params(cfg.beamscan_params);
        bind_outputs(options, "beamscan_estimator", {"aoa_estimates", "angle_spectrum"});
        // Compact only cells whose snapshots are poorly explained by the
        // conventional rank-one steering model.
        auto screen =
            graph.op<radar::RankOneResidualScreenOp<float>>("rank_one_residual_screen", extents);
        screen.set_params(cfg.screen_params);
        bind_outputs(
            options, "rank_one_residual_screen",
            {"snapshot_out", "selected_idx", "is_valid_out", "residual_score", "selection_mask"});
        // Merge selected 1..N estimator outputs with unselected Beamscan rows.
        auto expansion =
            graph.op<radar::MultiSourceExpansionOp<float>>("multi_source_expansion", extents);
        bind_outputs(options, "multi_source_expansion", {"detection_idx_out", "aoa_estimates_out"});
        // Gather physical coordinates and score for the expanded row set.
        auto features = graph.op<radar::FeatureAssemblyOp<float>>("feature_assembly", extents,
                                                                  cfg.assembly_params->features);
        bind_outputs(options, "feature_assembly", {"points"});
        // Report raw angular detections; this workflow intentionally has no
        // clustering or detection-reduction stage.
        auto report = graph.op<app::DetectionReportOp>(
            "detection_report", app::DetectionReportOp::Config{
                                    .mode = app::DetectionReportMode::kRawDetections,
                                    .angle_convention = cfg.beamscan_params->angle_convention,
                                    .cadence = cadence});

        // Processing and reporting share one source-driven partition. The
        // selected super-resolution estimator is added below.
        auto application = graph.partition("application");
        application.add(source)
            .add(deinterleave)
            .add(range)
            .add(clutter)
            .add(doppler)
            .add(cfar)
            .add(compensation)
            .add(beamscan)
            .add(screen)
            .add(expansion)
            .add(features)
            .add(report);

        // TDM front end: interleaved ADC -> uncompensated per-TX RD maps.
        graph.add_flow(source->adc_cube, deinterleave->adc_cube, queue);
        graph.add_flow(deinterleave->adc_cube_out, range->adc_cube, queue);
        graph.add_flow(range->range_profiles, clutter->range_profiles, queue);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, queue);
        // Detection-first branch: fold TX/RX power, then gather coherent
        // multi-look snapshots only at peak-CFAR cells.
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, queue);
        graph.add_flow(doppler->rd_maps, compensation->rd_maps, queue);
        graph.add_flow(cfar->detection_idx, compensation->detection_idx, queue);

        // Beamscan and the residual screen consume the same compensated looks,
        // validity mask, and tick-coherent virtual geometry.
        graph.add_flow(compensation->snapshot, beamscan->snapshot, queue);
        graph.add_flow(compensation->is_valid, beamscan->is_valid, queue);
        graph.add_flow(compensation->array_geometry, beamscan->array_geometry, queue);
        graph.add_flow(compensation->snapshot, screen->snapshot, queue);
        graph.add_flow(compensation->is_valid, screen->is_valid, queue);
        graph.add_flow(beamscan->aoa_estimates, screen->baseline_aoa, queue);
        graph.add_flow(compensation->array_geometry, screen->array_geometry, queue);

        // Preserve the complete CFAR row set while selected rows take the
        // super-resolution detour.
        graph.add_flow(cfar->detection_idx, expansion->detection_idx, queue);
        graph.add_flow(beamscan->aoa_estimates, expansion->baseline_aoa, queue);
        graph.add_flow(screen->selected_idx, expansion->selected_idx, queue);

        // Every supported estimator fills the same sparse multi-source slot.
        // Only the startup-selected implementation is instantiated.
        const auto connect_estimator = [&](auto estimator) {
          application.add(estimator);
          graph.add_flow(screen->snapshot_out, estimator->snapshot, queue);
          graph.add_flow(screen->is_valid_out, estimator->is_valid, queue);
          graph.add_flow(compensation->array_geometry, estimator->array_geometry, queue);
          graph.add_flow(estimator->aoa_estimates, expansion->resolved_aoa, queue);
          graph.add_flow(estimator->doa_multi, expansion->doa_multi, queue);
          graph.add_flow(estimator->source_count, expansion->source_count, queue);
        };
        switch (cfg.algorithm) {
          case adaptive::Algorithm::kMusic: {
            auto estimator = graph.op<radar::MUSICEstimatorOp<float>>("music_estimator", extents);
            estimator.set_params(*cfg.music_params);
            bind_outputs(
                options, "music_estimator",
                {"aoa_estimates", "angle_spectrum", "doa_multi", "source_count", "eigenvalues"});
            connect_estimator(estimator);
            break;
          }
          case adaptive::Algorithm::kRootMusic: {
            auto estimator =
                graph.op<radar::RootMUSICEstimatorOp<float>>("root_music_estimator", extents);
            estimator.set_params(*cfg.root_music_params);
            bind_outputs(options, "root_music_estimator",
                         {"aoa_estimates", "doa_multi", "source_count", "eigenvalues"});
            connect_estimator(estimator);
            break;
          }
          case adaptive::Algorithm::kEsprit: {
            auto estimator = graph.op<radar::ESPRITEstimatorOp<float>>("esprit_estimator", extents);
            estimator.set_params(*cfg.esprit_params);
            bind_outputs(options, "esprit_estimator",
                         {"aoa_estimates", "doa_multi", "source_count", "eigenvalues"});
            connect_estimator(estimator);
            break;
          }
          case adaptive::Algorithm::kMvdr: {
            auto estimator = graph.op<radar::MVDREstimatorOp<float>>("mvdr_estimator", extents);
            estimator.set_params(*cfg.mvdr_params);
            bind_outputs(
                options, "mvdr_estimator",
                {"aoa_estimates", "angle_spectrum", "doa_multi", "source_count", "eigenvalues"});
            connect_estimator(estimator);
            break;
          }
        }

        // Duplicate indices deliberately gather identical range, velocity,
        // and score values for multiple angles resolved in one RD cell.
        graph.add_flow(expansion->detection_idx_out, features->detection_idx, queue);
        graph.add_flow(expansion->aoa_estimates_out, features->aoa_estimates, queue);
        graph.add_flow(range->range_bins, features->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, features->velocity_bins, queue);
        graph.add_flow(cfar->detection_score, features->detection_score, queue);

        // Report expanded raw detections against the physical coordinate axes.
        graph.add_flow(features->points, report->points, queue);
        graph.add_flow(expansion->detection_idx_out, report->detection_idx, queue);
        graph.add_flow(range->range_bins, report->range_bins, queue);
        graph.add_flow(doppler->velocity_bins, report->velocity_bins, queue);

#if HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ
        if (cfg.visualize) {
          app::viz::add_fmcw_dashboard(
              graph, application, options, cfg.visualization, cfg.frame_period, true,
              cfg.beamscan_params->az_coordinates, cfg.beamscan_params->el_coordinates,
              cfg.beamscan_params->angle_convention, cfg.scene.targets, cfg.scene_visibility,
              std::string("Adaptive AoA | ") + adaptive::algorithm_name(cfg.algorithm),
              app::viz::RawDashboardInputs{cfar->cfar_power, expansion->detection_idx_out,
                                           features->points, range->range_bins,
                                           doppler->velocity_bins},
              app::application_description("adaptive_aoa",
                                           adaptive::algorithm_mode(cfg.algorithm)));
        }
#endif
      });
}
