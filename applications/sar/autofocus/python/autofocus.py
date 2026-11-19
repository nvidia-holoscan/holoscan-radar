# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python graph for the autofocus SAR workflow."""

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
        "autofocus", config, cphd_override, presenter=presenter, choose_presenter=choose_presenter
    )
    reader = ctx.op("cphd_reader")
    range_op = ctx.op("range_compression")
    manager = ctx.op("aperture_manager")
    bp = ctx.op("backprojection")
    ctx.add_flow(reader.phase_history, range_op.phase_history)
    pc = ctx.op("phase_correction")
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
    if ctx.flags["position_error"]:
        lie = ctx.op("perturb_positions")
        ctx.partition("application", lie)
        ctx.add_flow(range_op.range_profiles, lie.range_profiles)
        ctx.add_flow(positions, lie.positions)
        ctx.add_flow(reference, lie.reference_position)
        ctx.add_flow(lie.positions_out, manager.positions)
        ctx.add_flow(lie.reference_position_out, manager.reference_position)
        ingest = lie.range_profiles_out
    else:
        ctx.add_flow(positions, manager.positions)
        ctx.add_flow(reference, manager.reference_position)
        ingest = range_op.range_profiles

    # The retained replay uses one queue slot to bound aperture-sized pools.
    if ctx.flags["oracle"]:
        oracle = ctx.op("oracle_phase")
        ctx.partition("estimate", oracle)
        ctx.add_flow(manager.range_profiles_out, oracle.range_profiles, queue_depth=1)
        ctx.add_flow(oracle.phase_correction, pc.phase_correction)
    else:
        auxiliary_stage = ctx.op("af_aux_stage")
        ctx.partition("estimate", auxiliary_stage)
        ctx.add_flow(manager.range_profiles_out, auxiliary_stage.input, queue_depth=1)
        estimator_stage = ctx.op("af_estimator_stage")
        aux = ctx.op("aux_bp")
        if ctx.flags["aux_polar"]:
            points = ctx.op("aux_points")
            ctx.partition("estimate", points)
            ctx.add_flow(auxiliary_stage.output, points.range_profiles, queue_depth=1)
            ctx.add_flow(points.range_profiles_out, aux.range_profiles, queue_depth=1)
            ctx.add_flow(points.sample_points, aux.sample_points, queue_depth=1)
        else:
            ctx.add_flow(auxiliary_stage.output, aux.range_profiles, queue_depth=1)
        af = ctx.op("autofocus")
        resample = ctx.op("phase_resample")
        drain = ctx.op("af_image_drain")
        sharp = ctx.op("autofocus_sharpness")
        for member in (aux, estimator_stage, af, resample, drain):
            ctx.partition("estimate", member)
        ctx.partition("sharpness", sharp)
        ctx.add_flow(manager.positions_out, aux.positions)
        ctx.add_flow(manager.reference_position_out, aux.reference_position)
        ctx.add_flow(manager.control_out, aux.control)
        if ctx.flags["wire_weight"]:
            ctx.add_flow(manager.weight, aux.weight)
        ctx.add_flow(aux.image, estimator_stage.input)
        ctx.add_flow(estimator_stage.output, af.image)
        ctx.add_flow(af.image_out, drain.input)
        ctx.add_flow(af.sharpness, sharp.sharpness)
        ctx.add_flow(af.termination, sharp.termination)
        ctx.add_flow(af.iterations, sharp.iterations)
        ctx.add_flow(af.phase_correction, resample.phase_correction)
        ctx.add_flow(resample.phase_correction_out, pc.phase_correction)
        if ctx.flags["position_error"]:
            score = ctx.op("autofocus_score")
            ctx.partition("score", score)
            ctx.add_flow(resample.phase_correction_out, score.phase_correction)

    ctx.add_flow(ingest, manager.range_profiles)
    ctx.add_flow(manager.range_profiles_out, pc.range_profiles, queue_depth=1)
    final_stage = ctx.op("af_final_stage")
    ctx.add_flow(pc.range_profiles_out, final_stage.input, queue_depth=1)
    ctx.add_flow(final_stage.output, bp.range_profiles, queue_depth=1)
    ctx.add_flow(manager.positions_out, bp.positions)
    ctx.add_flow(manager.reference_position_out, bp.reference_position)
    ctx.add_flow(manager.control_out, bp.control)
    if ctx.flags["wire_weight"]:
        ctx.add_flow(manager.weight, bp.weight)
    ctx.partition("correction", pc)
    ctx.partition("correction", final_stage)
    ctx.finish()
    return BuiltSarGraph(ctx)
