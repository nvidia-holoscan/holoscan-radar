# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinated multi-waveform MIMO TDM Graph."""

from pathlib import Path

from holoscan_radar import (
    BeamscanEstimatorOpF32,
    CFARDetectorOpF32,
    ClusterDBSCANOpF32,
    ClutterFilterOpF32,
    DetectionReductionOpF32,
    FeatureAssemblyOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    MIMODeinterleaveOpF32,
    TDMSparseCompensationOpF32,
    _radar,
)
from holoscan_radar.applications.fmcw_dashboard import add_dashboard
from holoscan_radar.applications.siso import BuiltGraph

import holoscan as hs


def build_graph(config: str | Path) -> BuiltGraph:
    """Compose the coordinated TX/RX graph with resolved native configuration."""

    path = str(Path(config).resolve())
    cfg = _radar._load_multi(path)
    graph = hs.Graph("multi_waveform")
    options = hs.CompileOptions()
    extents = cfg["extents"]
    for section in ("range_processor", "doppler_processor", "tdm_sparse_compensation"):
        cfg[section]["waveform"] = cfg["initial_waveform"]

    tx = graph.op("holoscan.radar._app.MultiTx", "waveform_source", path=path)
    source = graph.op("holoscan.radar._app.MultiRx", "adc_source", path=path)
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
    features = FeatureAssemblyOpF32(
        graph, "feature_assembly", extents=extents, features=cfg["feature_assembly"]["features"]
    )
    cluster = ClusterDBSCANOpF32(graph, "cluster_dbscan", extents=extents, **cfg["cluster_dbscan"])
    reduction = DetectionReductionOpF32(
        graph, "detection_reduction", extents=extents, **cfg["detection_reduction"]
    )
    report = graph.op("holoscan.radar._app.MultiReport", "cluster_report", path=path)
    graph.partition("transmit").add(tx)
    graph.partition("receive").add(source)
    application = graph.partition("processing")
    application.add(
        deinterleave,
        range_op,
        clutter,
        doppler,
        cfar,
        compensation,
        beamscan,
        features,
        cluster,
        reduction,
        report,
    )
    waveform_depth = cfg["max_frames_in_flight"] + 1
    # Publish each waveform before its matching ADC cube reaches the Exact joins.
    graph.add_flow(tx.waveform, range_op.waveform, queue_depth=waveform_depth)
    graph.add_flow(tx.waveform, doppler.waveform, queue_depth=waveform_depth)
    graph.add_flow(tx.waveform, compensation.waveform, queue_depth=waveform_depth)
    # Deinterleave TDM ADC samples and form per-channel Doppler maps.
    graph.add_flow(source.adc_cube, deinterleave.adc_cube, queue_depth=4)
    graph.add_flow(deinterleave.adc_cube_out, range_op.adc_cube, queue_depth=4)
    graph.add_flow(range_op.range_profiles, clutter.range_profiles, queue_depth=4)
    graph.add_flow(clutter.range_profiles_out, doppler.range_profiles, queue_depth=4)

    # Detect on folded power and compensate selected coherent snapshots.
    graph.add_flow(doppler.rd_maps, cfar.rd_maps, queue_depth=4)
    graph.add_flow(doppler.rd_maps, compensation.rd_maps, queue_depth=4)
    graph.add_flow(cfar.detection_idx, compensation.detection_idx, queue_depth=4)

    # Estimate angle with tick-coherent virtual-array geometry.
    graph.add_flow(compensation.snapshot, beamscan.snapshot, queue_depth=4)
    graph.add_flow(compensation.is_valid, beamscan.is_valid, queue_depth=4)
    graph.add_flow(compensation.array_geometry, beamscan.array_geometry, queue_depth=4)

    # Resolve detections into physical coordinates and retain their scores.
    graph.add_flow(cfar.detection_idx, features.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, features.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, features.velocity_bins, queue_depth=4)
    graph.add_flow(beamscan.aoa_estimates, features.aoa_estimates, queue_depth=4)
    graph.add_flow(cfar.detection_score, features.detection_score, queue_depth=4)

    # Cluster on range-Doppler bins and reduce with the score column.
    graph.add_flow(features.points, cluster.points, queue_depth=4)
    graph.add_flow(cluster.labels, reduction.labels, queue_depth=4)
    graph.add_flow(cfar.detection_idx, reduction.detection_idx, queue_depth=4)
    graph.add_flow(features.points, reduction.points, queue_depth=4)

    # Report clusters against their physical coordinate axes.
    graph.add_flow(features.points, report.points, queue_depth=4)
    graph.add_flow(cluster.labels, report.labels, queue_depth=4)
    graph.add_flow(cluster.sizes, report.sizes, queue_depth=4)
    graph.add_flow(reduction.rep_idx, report.detection_idx, queue_depth=4)
    graph.add_flow(reduction.reduced_points, report.reduced_points, queue_depth=4)
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
        "feature_assembly": ("points",),
        "cluster_dbscan": ("labels", "sizes"),
        "detection_reduction": ("rep_idx", "reduced_points"),
    }
    for operator_name, port_names in outputs.items():
        for port_name in port_names:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_name,
                output_port=port_name,
                device=hs.execution.DeviceId(0),
            )
    if cfg["visualize"]:
        add_dashboard(
            graph,
            application,
            options,
            cfg,
            path,
            "Multi",
            "Multi-Waveform",
            {
                "cfar_power": cfar.cfar_power,
                "detection_idx": cfar.detection_idx,
                "points": features.points,
                "range_bins": range_op.range_bins,
                "velocity_bins": doppler.velocity_bins,
                "labels": cluster.labels,
                "sizes": cluster.sizes,
                "reduced_points": reduction.reduced_points,
                "waveform": tx.waveform,
            },
            clustered=True,
            profiled=True,
            profile_gap=cfg["visualization_profile_max_gap_frames"],
        )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return BuiltGraph(graph, options, cfg["num_frames"])
