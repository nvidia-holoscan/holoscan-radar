// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <holoscan/core/graph.hpp>
#include "radar_app/sar/graph_driver.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/sar/coordinate_ops.hpp"
#include "collection_stager.hpp"  // NOLINT(build/include_subdir)

namespace app = holoscan::radar::app;

int main(int argc, char** argv) {
  return run_sar_graph_app(
      argc, argv, "independent_images", load_config,
      [](auto& graph, auto& options, const auto& queue, const auto& cfg, auto& reader, auto& range,
         auto& manager, auto& bp, auto& source, auto& application, auto& /*formation*/,
         const auto& extents) -> const holoscan::Output<holoscan::Tensor>* {
        auto stager = graph.template op<app::sar::CollectionBoundaryStagerOp>(
            "collection_stager", cfg.collection_geometries, cfg.collection_spacings,
            cfg.collection_frequencies);
        application.add(stager);
        graph.add_flow(range->range_profiles, stager->range_profiles, queue);
        graph.add_flow(reader->stream_boundary, stager->stream_boundary, queue);
        graph.add_flow(stager->range_profiles_out, manager->range_profiles, queue);
        graph.add_flow(stager->sample_geometry, manager->sample_geometry, queue);
        graph.add_flow(stager->range_bin_spacing, bp->range_bin_spacing, queue);
        graph.add_flow(stager->reference_frequency, bp->reference_frequency, queue);
        graph.add_flow(manager->sample_geometry_out, bp->sample_geometry, queue);
        graph.add_flow(reader->stream_boundary, manager->stream_boundary, queue);
        if (cfg.transform_to_scene_enu) {
          auto enu = graph.template op<app::sar::EcefToSceneEnuOp>(
              "ecef_to_enu", cfg.enu_origin_ecef, cfg.quantize_phase_center_f32,
              cfg.reader_pulses_per_tick);
          source.add(enu);
          bind_outputs(options, "ecef_to_enu", {"positions_out", "reference_position_out"});
          graph.add_flow(reader->positions, enu->positions, queue);
          graph.add_flow(reader->reference_position, enu->reference_position, queue);
          graph.add_flow(enu->positions_out, manager->positions, queue);
          graph.add_flow(enu->reference_position_out, manager->reference_position, queue);
        } else {
          graph.add_flow(reader->positions, manager->positions, queue);
          graph.add_flow(reader->reference_position, manager->reference_position, queue);
        }
        graph.add_flow(manager->range_profiles_out, bp->range_profiles, queue);
        graph.add_flow(manager->positions_out, bp->positions, queue);
        graph.add_flow(manager->reference_position_out, bp->reference_position, queue);
        graph.add_flow(manager->control_out, bp->control, queue);
        return &bp->image;
      });
}
