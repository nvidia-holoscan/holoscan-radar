// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <holoscan/core/graph.hpp>
#include "radar_app/sar/graph_driver.hpp"
#include "config.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/sar/mosaic_op.hpp"
#include "scroll_op.hpp"  // NOLINT(build/include_subdir)

namespace app = holoscan::radar::app;

int main(int argc, char** argv) {
  return run_sar_graph_app(
      argc, argv, "continuous_strip", load_config,
      [](auto& graph, auto& options, const auto& queue, const auto& cfg, auto& reader, auto& range,
         auto& manager, auto& bp, auto& application,
         const auto& extents) -> const holoscan::Output<holoscan::Tensor>* {
        const int64_t slice =
            cfg.emit_interval_pulses > 0 && cfg.reader_pulses_per_tick > cfg.emit_interval_pulses
                ? cfg.emit_interval_pulses
                : cfg.reader_pulses_per_tick;
        auto scroll = graph.template op<app::sar::ScrollGeometryStagerOp>(
            "scroll_stager", cfg.scroll_parent, cfg.sample_geometry, cfg.scroll_anchor,
            cfg.reader_loop, slice, cfg.reader_pulses_per_tick, *extents.max_range_bins);
        application.add(scroll);
        bind_outputs(options, "scroll_stager",
                     {"range_profiles_out", "positions_out", "reference_position_out"});
        graph.add_flow(range->range_profiles, scroll->range_profiles, queue);
        graph.add_flow(reader->positions, scroll->positions, queue);
        graph.add_flow(reader->reference_position, scroll->reference_position, queue);
        graph.add_flow(reader->stream_boundary, scroll->stream_boundary, queue);
        graph.add_flow(scroll->range_profiles_out, manager->range_profiles, queue);
        graph.add_flow(scroll->positions_out, manager->positions, queue);
        graph.add_flow(scroll->reference_position_out, manager->reference_position, queue);
        graph.add_flow(scroll->stream_boundary_out, manager->stream_boundary, queue);
        graph.add_flow(scroll->sample_geometry, bp->sample_geometry, queue);
        graph.add_flow(manager->range_profiles_out, bp->range_profiles, queue);
        graph.add_flow(manager->positions_out, bp->positions, queue);
        graph.add_flow(manager->reference_position_out, bp->reference_position, queue);
        graph.add_flow(manager->control_out, bp->control, queue);
        auto mosaic = graph.template op<app::sar::SarMosaicOp>("sar_mosaic", cfg.scroll_parent,
                                                               cfg.reader_loop);
        application.add(mosaic);
        bind_outputs(options, "sar_mosaic", {"image_out"});
        graph.add_flow(bp->image, mosaic->image, queue);
        graph.add_flow(bp->sample_geometry_out, mosaic->sample_geometry, queue);
        return &mosaic->image_out;
      });
}
