// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <holoscan/core/graph.hpp>
#include "radar_app/sar/graph_driver.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/sar/coordinate_ops.hpp"
#include "radar_app/sar/mosaic_op.hpp"
#include "tile_stager.hpp"  // NOLINT(build/include_subdir)

namespace app = holoscan::radar::app;

int main(int argc, char** argv) {
  return run_sar_graph_app(
      argc, argv, "tiled_strip", load_config,
      [](auto& graph, auto& options, const auto& queue, const auto& cfg, auto& reader, auto& range,
         auto& manager, auto& bp, auto& application,
         const auto& extents) -> const holoscan::Output<holoscan::Tensor>* {
        auto stager = graph.template op<app::sar::TileGeometryStagerOp>(
            "tile_stager", cfg.tile_geometries, cfg.tile_pulses);
        application.add(stager);
        bind_outputs(options, "tile_stager", {"range_profiles_out"});
        graph.add_flow(range->range_profiles, stager->range_profiles, queue);
        graph.add_flow(reader->stream_boundary, stager->stream_boundary, queue);
        graph.add_flow(stager->range_profiles_out, manager->range_profiles, queue);
        graph.add_flow(stager->sample_geometry, manager->sample_geometry, queue);
        graph.add_flow(manager->sample_geometry_out, bp->sample_geometry, queue);
        graph.add_flow(reader->stream_boundary, manager->stream_boundary, queue);
        if (cfg.transform_to_scene_enu) {
          auto enu = graph.template op<app::sar::EcefToSceneEnuOp>(
              "ecef_to_enu", cfg.enu_origin_ecef, cfg.quantize_phase_center_f32,
              cfg.reader_pulses_per_tick);
          application.add(enu);
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
        if (cfg.wire_weight)
          graph.add_flow(manager->weight, bp->weight, queue);
        auto mosaic = graph.template op<app::sar::SarMosaicOp>("sar_mosaic", cfg.parent_geometry,
                                                               cfg.reader_loop);
        application.add(mosaic);
        bind_outputs(options, "sar_mosaic", {"image_out"});
        graph.add_flow(bp->image, mosaic->image, queue);
        graph.add_flow(bp->sample_geometry_out, mosaic->sample_geometry, queue);
        return &mosaic->image_out;
      });
}
