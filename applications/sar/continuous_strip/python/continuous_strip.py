# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python graph for the continuous strip SAR workflow."""

from pathlib import Path

from holoscan_radar.applications.sar_graph import BuiltSarGraph, prepare


def build_graph(
    config: str | Path,
    *,
    cphd_override: str | Path | None = None,
    presenter: str | None = None,
    choose_presenter: bool = False,
) -> BuiltSarGraph:
    """Compose the C++-equivalent processing graph using native SAR helpers."""
    ctx = prepare(
        "continuous_strip",
        config,
        cphd_override,
        presenter=presenter,
        choose_presenter=choose_presenter,
    )
    reader = ctx.op("cphd_reader")
    range_op = ctx.op("range_compression")
    manager = ctx.op("aperture_manager")
    bp = ctx.op("backprojection")
    ctx.add_flow(reader.phase_history, range_op.phase_history)
    scroll = ctx.op("scroll_stager")
    ctx.partition("application", scroll)
    ctx.add_flow(range_op.range_profiles, scroll.range_profiles)
    ctx.add_flow(reader.positions, scroll.positions)
    ctx.add_flow(reader.reference_position, scroll.reference_position)
    ctx.add_flow(reader.stream_boundary, scroll.stream_boundary)
    # A reader batch may split into four formation ticks; preserve C++ slack.
    ctx.add_flow(scroll.range_profiles_out, manager.range_profiles, queue_depth=16)
    ctx.add_flow(scroll.positions_out, manager.positions, queue_depth=16)
    ctx.add_flow(scroll.reference_position_out, manager.reference_position, queue_depth=16)
    ctx.add_flow(scroll.stream_boundary_out, manager.stream_boundary, queue_depth=16)
    ctx.add_flow(scroll.sample_geometry, bp.sample_geometry, queue_depth=16)
    ctx.add_flow(manager.range_profiles_out, bp.range_profiles, queue_depth=16)
    ctx.add_flow(manager.positions_out, bp.positions, queue_depth=16)
    ctx.add_flow(manager.reference_position_out, bp.reference_position, queue_depth=16)
    ctx.add_flow(manager.control_out, bp.control, queue_depth=16)
    mosaic = ctx.op("sar_mosaic")
    ctx.partition("formation", mosaic)
    ctx.add_flow(bp.image, mosaic.image)
    ctx.add_flow(bp.sample_geometry_out, mosaic.sample_geometry)
    ctx.finish()
    return BuiltSarGraph(ctx)
