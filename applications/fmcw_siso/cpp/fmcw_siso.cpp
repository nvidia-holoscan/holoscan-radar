// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SISO FMCW pipeline example (HSDK 5 Graph):
//
//   SimAdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
//               -> FMCWDopplerProcessorOp -> CFARDetectorOp
//                                        -> DetectionReportOp
//                                        -> FeatureAssemblyOp
//                                             -> ClusterDBSCANOp
//                                                  -> DetectionReductionOp
//                                                       -> ClusterReportOp
//                                         [-> RdMapToImageOp -> X11RdViewerOp]
//
// With visualize on, the X11 viewer also takes the detection cells, the range /
// velocity bins, and (outside CFAR peak modes) cluster labels + representatives,
// so it draws the RD-map overlays the Holoviz demo drew: physical axes, cluster
// -colored detections, a legend, and the rolling range/velocity plots.
// Build: cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
// Run:   ./build/applications/fmcw_siso/cpp/fmcw_siso [config.yaml]

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

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
#include "config.hpp"  // NOLINT(build/include_subdir): colocated app header

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
      argc, argv, "fmcw_siso", "fmcw_siso.yaml", load_config,
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

        auto sink = graph.op<DetectionReportOp>(
            "detection_report", DetectionReportOp::Config{cadence});
        auto mask_sink = graph.op<NullSinkOp>("detection_mask_sink");
        auto noise_sink = graph.op<NullSinkOp>("noise_estimate_sink");
        auto power_sink = graph.op<NullSinkOp>("cfar_power_sink");

        graph.add_flow(source->adc_cube, range->adc_cube, q);
        graph.add_flow(range->range_profiles, clutter->range_profiles, q);
        graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
        graph.add_flow(doppler->rd_maps, cfar->rd_maps, q);
        graph.add_flow(cfar->detection_idx, sink->detection_idx, q);
        graph.add_flow(cfar->detection_score, sink->detection_score, q);
        graph.add_flow(range->range_bins, sink->range_bins, q);
        graph.add_flow(doppler->velocity_bins, sink->velocity_bins, q);
        graph.add_flow(cfar->detection_mask, mask_sink->input, q);
        graph.add_flow(cfar->noise_estimate, noise_sink->input, q);
        graph.add_flow(cfar->cfar_power, power_sink->input, q);

#if defined(HOLOSCAN_RADAR_HAS_X11)
        // Created before the clustering stages so their outputs can be wired
        // into the overlay ports below; the viewer's contract covers exactly
        // the ports this app wires (see X11RdViewerConfig).
        std::optional<holoscan::OperatorHandle<X11RdViewerOp>> viewer;
        if (cfg.visualize) {
          X11RdViewerConfig viz_cfg;
          viz_cfg.title = "fmcw_siso - Range-Doppler";  // ASCII: Latin-1 draw
          viz_cfg.width = static_cast<int>(cfg.viz.window_width);
          viz_cfg.height = static_cast<int>(cfg.viz.window_height);
          viz_cfg.overlay_detections = true;
          viz_cfg.overlay_clusters = !cfg.use_cfar_peaks;
          viz_cfg.plots_enabled = cfg.plots.enable;
          viz_cfg.plot_window = cfg.plots.window;
          viz_cfg.frame_period = cfg.frame_period;
          viz_cfg.marker_radius = cfg.viz.marker_radius;
          viz_cfg.marker_outline = cfg.viz.marker_outline;

          auto to_image =
              graph.op<RdMapToImageOp>("rd_map_to_image", extents);
          viewer = graph.op<X11RdViewerOp>("x11_rd_viewer", viz_cfg);
          graph.add_flow(doppler->rd_maps, to_image->rd_maps, q);
          graph.add_flow(to_image->rd_image, (*viewer)->rd_image, q);
          graph.add_flow(cfar->detection_idx, (*viewer)->detection_idx, q);
          graph.add_flow(range->range_bins, (*viewer)->range_bins, q);
          graph.add_flow(doppler->velocity_bins, (*viewer)->velocity_bins, q);
          bind_out(options, "rd_map_to_image", "rd_image");
          HOLOSCAN_LOG_INFO("fmcw_siso: X11 RD live viewer enabled ({}x{})",
                            viz_cfg.width, viz_cfg.height);
        }
#else
        if (cfg.visualize) {
          HOLOSCAN_LOG_WARN(
              "fmcw_siso: visualize requested but X11/Holoviz unavailable; "
              "running log-only");
          cfg.visualize = false;
        }
#endif

        if (!cfg.use_cfar_peaks) {
          std::vector<radar::AssemblyFeature> features{
              radar::AssemblyFeature::kRange,
              radar::AssemblyFeature::kVelocity};
          auto feature_assembly =
              graph.op<radar::FeatureAssemblyOp<float>>(
                  "feature_assembly", extents, std::move(features));

          auto cluster = graph.op<radar::ClusterDBSCANOp<float>>(
              "cluster_dbscan", extents);
          cluster.set_params(cfg.dbscan_params);

          auto reduction = graph.op<radar::DetectionReductionOp<float>>(
              "detection_reduction", extents);
          reduction.set_params(cfg.reduction_params);

          auto cluster_report =
              graph.op<ClusterReportOp>("cluster_report");

          graph.add_flow(cfar->detection_idx, feature_assembly->detection_idx,
                         q);
          graph.add_flow(range->range_bins, feature_assembly->range_bins, q);
          graph.add_flow(doppler->velocity_bins,
                         feature_assembly->velocity_bins, q);
          graph.add_flow(feature_assembly->points, cluster->points, q);
          graph.add_flow(feature_assembly->points, cluster_report->points, q);
          graph.add_flow(cluster->labels, cluster_report->labels, q);
          graph.add_flow(cluster->sizes, cluster_report->sizes, q);
          graph.add_flow(cluster->labels, reduction->labels, q);
          graph.add_flow(cfar->detection_idx, reduction->detection_idx, q);
          graph.add_flow(feature_assembly->points, reduction->points, q);
          graph.add_flow(reduction->rep_idx, cluster_report->rep_idx, q);
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
          bind_out(options, "detection_reduction", "reduced_points");
        }

        bind_out(options, "adc_source", "adc_cube");
        bind_out(options, "range_processor", "range_profiles");
        bind_out(options, "range_processor", "range_bins");
        bind_out(options, "clutter_filter", "range_profiles_out");
        bind_out(options, "doppler_processor", "rd_maps");
        bind_out(options, "doppler_processor", "velocity_bins");
        bind_out(options, "cfar_detector", "detection_mask");
        bind_out(options, "cfar_detector", "detection_score");
        bind_out(options, "cfar_detector", "noise_estimate");
        bind_out(options, "cfar_detector", "detection_idx");
        bind_out(options, "cfar_detector", "cfar_power");
      });
}
