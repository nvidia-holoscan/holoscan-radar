# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense beamform-then-detect SIMO Graph."""

from pathlib import Path

from holoscan_radar import (
    CFARDetectorOpF32,
    ClutterFilterOpF32,
    FeatureAssemblyOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    PhaseShiftBeamformerOpF32,
    _radar,
)
from holoscan_radar.applications.fmcw_dashboard import add_dashboard
from holoscan_radar.applications.siso import BuiltGraph

import holoscan as hs


def build_graph(config: str | Path) -> BuiltGraph:
    """Compose the dense SIMO Graph using the C++ config loader."""

    path = str(Path(config).resolve())
    cfg = _radar._load_dense(path)
    device = _radar._select_application_device(cfg["visualize"], cfg["visualize"])
    graph = hs.Graph("dense_beamforming")
    options = hs.CompileOptions()
    extents = cfg["extents"]
    source = graph.op("holoscan.radar._app.DenseSource", "adc_source", path=path)
    range_op = FMCWRangeProcessorOpF32(
        graph, "range_processor", extents=extents, **cfg["range_processor"]
    )
    clutter = ClutterFilterOpF32(graph, "clutter_filter", extents=extents, **cfg["clutter_filter"])
    doppler = FMCWDopplerProcessorOpF32(
        graph, "doppler_processor", extents=extents, **cfg["doppler_processor"]
    )
    beamformer = PhaseShiftBeamformerOpF32(
        graph, "phase_shift_beamformer", extents=extents, **cfg["phase_shift_beamformer"]
    )
    cfar = CFARDetectorOpF32(
        graph, "cfar_detector", extents=cfg["cfar_extents"], **cfg["cfar_detector"]
    )
    features = FeatureAssemblyOpF32(
        graph, "feature_assembly", extents=extents, features=cfg["feature_assembly"]["features"]
    )
    report = graph.op("holoscan.radar._app.DenseReport", "detection_report", path=path)
    application = graph.partition("application")
    application.add(source, range_op, clutter, doppler, beamformer, cfar, features, report)
    # FMCW front end: ADC cube to per-channel range-Doppler maps.
    graph.add_flow(source.adc_cube, range_op.adc_cube, queue_depth=4)
    graph.add_flow(range_op.range_profiles, clutter.range_profiles, queue_depth=4)
    graph.add_flow(clutter.range_profiles_out, doppler.range_profiles, queue_depth=4)

    # Beamform the full angle cube before detecting across angle planes.
    graph.add_flow(doppler.rd_maps, beamformer.rd_maps, queue_depth=4)
    graph.add_flow(beamformer.rda_cube, cfar.rd_maps, queue_depth=4)

    # Resolve each peak with the effective angle, range, and velocity grids.
    graph.add_flow(cfar.detection_idx, features.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, features.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, features.velocity_bins, queue_depth=4)
    graph.add_flow(beamformer.az_grid, features.az_grid, queue_depth=4)
    graph.add_flow(beamformer.el_grid, features.el_grid, queue_depth=4)
    graph.add_flow(cfar.detection_score, features.detection_score, queue_depth=4)

    # Report every retained multidimensional peak directly.
    graph.add_flow(features.points, report.points, queue_depth=4)
    graph.add_flow(cfar.detection_idx, report.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, report.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, report.velocity_bins, queue_depth=4)
    outputs = {
        "adc_source": ("adc_cube",),
        "range_processor": ("range_profiles", "range_bins"),
        "clutter_filter": ("range_profiles_out",),
        "doppler_processor": ("rd_maps", "velocity_bins"),
        "phase_shift_beamformer": ("rda_cube", "az_grid", "el_grid"),
        "cfar_detector": (
            "detection_mask",
            "detection_score",
            "noise_estimate",
            "detection_idx",
            "cfar_power",
        ),
        "feature_assembly": ("points",),
    }
    for operator_name, port_names in outputs.items():
        for port_name in port_names:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_name,
                output_port=port_name,
                device=hs.execution.DeviceId(device),
            )
    if cfg["visualize"]:
        add_dashboard(
            graph,
            application,
            options,
            cfg,
            path,
            "Dense",
            "SIMO Dense Beamforming",
            {
                "cfar_power": cfar.cfar_power,
                "detection_idx": cfar.detection_idx,
                "points": features.points,
                "range_bins": range_op.range_bins,
                "velocity_bins": doppler.velocity_bins,
            },
            device=device,
            clustered=False,
        )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return BuiltGraph(graph, options, cfg["num_frames"])
