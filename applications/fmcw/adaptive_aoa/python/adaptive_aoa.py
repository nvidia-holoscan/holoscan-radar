# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-source screening with a selected sparse angle estimator."""

from pathlib import Path

from holoscan_radar import (
    BeamscanEstimatorOpF32,
    CFARDetectorOpF32,
    ClutterFilterOpF32,
    ESPRITEstimatorOpF32,
    FeatureAssemblyOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    MIMODeinterleaveOpF32,
    MultiSourceExpansionOpF32,
    MUSICEstimatorOpF32,
    MVDREstimatorOpF32,
    RankOneResidualScreenOpF32,
    RootMUSICEstimatorOpF32,
    TDMSparseCompensationOpF32,
    _radar,
)
from holoscan_radar.applications.fmcw_dashboard import add_dashboard
from holoscan_radar.applications.siso import BuiltGraph

import holoscan as hs

_ESTIMATORS = {
    "music_estimator": MUSICEstimatorOpF32,
    "root_music_estimator": RootMUSICEstimatorOpF32,
    "esprit_estimator": ESPRITEstimatorOpF32,
    "mvdr_estimator": MVDREstimatorOpF32,
}


def build_graph(config: str | Path) -> BuiltGraph:
    """Compose the selected adaptive AoA graph from the C++ config loader."""

    path = str(Path(config).resolve())
    cfg = _radar._load_adaptive(path)
    graph = hs.Graph("adaptive_aoa")
    options = hs.CompileOptions()
    extents = cfg["extents"]
    source = graph.op("holoscan.radar._app.AdaptiveSource", "adc_source", path=path)
    deinterleave = MIMODeinterleaveOpF32(
        graph, "mimo_deinterleave", extents=extents, **cfg["mimo_deinterleave"]
    )
    range_op = FMCWRangeProcessorOpF32(
        graph, "range_processor", extents=extents, **cfg["range_processor"]
    )
    clutter = ClutterFilterOpF32(graph, "clutter_filter", extents=extents, **cfg["clutter_filter"])
    doppler = FMCWDopplerProcessorOpF32(
        graph, "doppler_processor", extents=extents, **cfg["doppler_processor"]
    )
    cfar = CFARDetectorOpF32(graph, "cfar_detector", extents=extents, **cfg["cfar_detector"])
    compensation = TDMSparseCompensationOpF32(
        graph, "tdm_sparse_compensation", extents=extents, **cfg["tdm_sparse_compensation"]
    )
    beamscan = BeamscanEstimatorOpF32(
        graph, "beamscan_estimator", extents=extents, **cfg["beamscan_estimator"]
    )
    screen = RankOneResidualScreenOpF32(
        graph, "rank_one_residual_screen", extents=extents, **cfg["rank_one_residual_screen"]
    )
    expansion = MultiSourceExpansionOpF32(graph, "multi_source_expansion", extents=extents)
    features = FeatureAssemblyOpF32(
        graph, "feature_assembly", extents=extents, features=cfg["feature_assembly"]["features"]
    )
    report = graph.op("holoscan.radar._app.AdaptiveReport", "detection_report", path=path)
    estimator_name = cfg["estimator_name"]
    estimator = _ESTIMATORS[estimator_name](
        graph, estimator_name, extents=extents, **cfg["estimator_params"]
    )
    application = graph.partition("application")
    application.add(
        source,
        deinterleave,
        range_op,
        clutter,
        doppler,
        cfar,
        compensation,
        beamscan,
        screen,
        expansion,
        features,
        report,
        estimator,
    )
    # Deinterleave TDM ADC samples and form per-channel Doppler maps.
    graph.add_flow(source.adc_cube, deinterleave.adc_cube, queue_depth=4)
    graph.add_flow(deinterleave.adc_cube_out, range_op.adc_cube, queue_depth=4)
    graph.add_flow(range_op.range_profiles, clutter.range_profiles, queue_depth=4)
    graph.add_flow(clutter.range_profiles_out, doppler.range_profiles, queue_depth=4)

    # Detect on folded power and compensate selected multi-look snapshots.
    graph.add_flow(doppler.rd_maps, cfar.rd_maps, queue_depth=4)
    graph.add_flow(doppler.rd_maps, compensation.rd_maps, queue_depth=4)
    graph.add_flow(cfar.detection_idx, compensation.detection_idx, queue_depth=4)

    # Share snapshots, validity, and tick-coherent geometry with the screen.
    graph.add_flow(compensation.snapshot, beamscan.snapshot, queue_depth=4)
    graph.add_flow(compensation.is_valid, beamscan.is_valid, queue_depth=4)
    graph.add_flow(compensation.array_geometry, beamscan.array_geometry, queue_depth=4)
    graph.add_flow(compensation.snapshot, screen.snapshot, queue_depth=4)
    graph.add_flow(compensation.is_valid, screen.is_valid, queue_depth=4)
    graph.add_flow(beamscan.aoa_estimates, screen.baseline_aoa, queue_depth=4)
    graph.add_flow(compensation.array_geometry, screen.array_geometry, queue_depth=4)

    # Keep every CFAR row while selected rows take the high-resolution path.
    graph.add_flow(cfar.detection_idx, expansion.detection_idx, queue_depth=4)
    graph.add_flow(beamscan.aoa_estimates, expansion.baseline_aoa, queue_depth=4)
    graph.add_flow(screen.selected_idx, expansion.selected_idx, queue_depth=4)

    # Connect the startup-selected estimator to the sparse multi-source slot.
    graph.add_flow(screen.snapshot_out, estimator.snapshot, queue_depth=4)
    graph.add_flow(screen.is_valid_out, estimator.is_valid, queue_depth=4)
    graph.add_flow(compensation.array_geometry, estimator.array_geometry, queue_depth=4)
    graph.add_flow(estimator.aoa_estimates, expansion.resolved_aoa, queue_depth=4)
    graph.add_flow(estimator.doa_multi, expansion.doa_multi, queue_depth=4)
    graph.add_flow(estimator.source_count, expansion.source_count, queue_depth=4)

    # Expand resolved angles while retaining source bins and scores.
    graph.add_flow(expansion.detection_idx_out, features.detection_idx, queue_depth=4)
    graph.add_flow(expansion.aoa_estimates_out, features.aoa_estimates, queue_depth=4)
    graph.add_flow(range_op.range_bins, features.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, features.velocity_bins, queue_depth=4)
    graph.add_flow(cfar.detection_score, features.detection_score, queue_depth=4)

    # Report expanded detections against the physical coordinate axes.
    graph.add_flow(features.points, report.points, queue_depth=4)
    graph.add_flow(expansion.detection_idx_out, report.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, report.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, report.velocity_bins, queue_depth=4)
    outputs = {
        "adc_source": ("adc_cube",),
        "mimo_deinterleave": ("adc_cube_out",),
        "range_processor": ("range_profiles", "range_bins"),
        "clutter_filter": ("range_profiles_out",),
        "doppler_processor": ("rd_maps", "velocity_bins"),
        "cfar_detector": (
            "detection_mask",
            "detection_score",
            "noise_estimate",
            "detection_idx",
            "cfar_power",
        ),
        "tdm_sparse_compensation": ("snapshot", "is_valid"),
        "beamscan_estimator": ("aoa_estimates", "angle_spectrum"),
        "rank_one_residual_screen": (
            "snapshot_out",
            "selected_idx",
            "is_valid_out",
            "residual_score",
            "selection_mask",
        ),
        "multi_source_expansion": ("detection_idx_out", "aoa_estimates_out"),
        "feature_assembly": ("points",),
        estimator_name: ("aoa_estimates", "doa_multi", "source_count", "eigenvalues")
        + (("angle_spectrum",) if estimator_name in ("music_estimator", "mvdr_estimator") else ()),
    }
    for operator_name, port_names in outputs.items():
        for port_name in port_names:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_name, output_port=port_name, device=hs.execution.DeviceId(0)
            )
    if cfg["visualize"]:
        add_dashboard(
            graph,
            application,
            options,
            cfg,
            path,
            "Adaptive",
            "Adaptive AoA",
            {
                "cfar_power": cfar.cfar_power,
                "detection_idx": expansion.detection_idx_out,
                "points": features.points,
                "range_bins": range_op.range_bins,
                "velocity_bins": doppler.velocity_bins,
            },
            clustered=False,
        )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return BuiltGraph(graph, options, cfg["num_frames"])
