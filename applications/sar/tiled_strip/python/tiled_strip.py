# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python graph for the tiled strip SAR workflow."""

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
        "tiled_strip", config, cphd_override, presenter=presenter, choose_presenter=choose_presenter
    )
    reader = ctx.op("cphd_reader")
    range_op = ctx.op("range_compression")
    manager = ctx.op("aperture_manager")
    bp = ctx.op("backprojection")
    ctx.add_flow(reader.phase_history, range_op.phase_history)
    stager = ctx.op("tile_stager")
    ctx.partition("application", stager)
    ctx.add_flow(range_op.range_profiles, stager.range_profiles)
    ctx.add_flow(reader.stream_boundary, stager.stream_boundary)
    ctx.add_flow(stager.range_profiles_out, manager.range_profiles)
    ctx.add_flow(stager.sample_geometry, manager.sample_geometry)
    ctx.add_flow(manager.sample_geometry_out, bp.sample_geometry)
    ctx.add_flow(reader.stream_boundary, manager.stream_boundary)
    if ctx.flags["transform_to_scene_enu"]:
        enu = ctx.op("ecef_to_enu")
        ctx.partition("source", enu)
        ctx.add_flow(reader.positions, enu.positions)
        ctx.add_flow(reader.reference_position, enu.reference_position)
        positions = enu.positions_out
        reference = enu.reference_position_out
    else:
        positions = reader.positions
        reference = reader.reference_position
    ctx.add_flow(positions, manager.positions)
    ctx.add_flow(reference, manager.reference_position)
    ctx.add_flow(manager.range_profiles_out, bp.range_profiles)
    ctx.add_flow(manager.positions_out, bp.positions)
    ctx.add_flow(manager.reference_position_out, bp.reference_position)
    ctx.add_flow(manager.control_out, bp.control)
    if ctx.flags["wire_weight"]:
        ctx.add_flow(manager.weight, bp.weight)
    mosaic = ctx.op("sar_mosaic")
    ctx.partition("formation", mosaic)
    ctx.add_flow(bp.image, mosaic.image)
    ctx.add_flow(bp.sample_geometry_out, mosaic.sample_geometry)
    ctx.finish()
    return BuiltSarGraph(ctx)
