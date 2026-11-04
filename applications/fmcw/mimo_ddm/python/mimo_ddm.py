# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear and nonlinear MIMO DDM Graph using the C++ config loader."""

from pathlib import Path

from holoscan_radar import (
    BeamscanEstimatorOpF32,
    CFARDetectorOpF32,
    ClusterDBSCANOpF32,
    DDMDenseDemultiplexOpF32,
    DetectionReductionOpF32,
    DopplerNotchOpF32,
    FeatureAssemblyOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    SnapshotGatherOpF32,
    VirtualArrayAssemblyOpF32,
    _radar,
)
from holoscan_radar.applications.fmcw_dashboard import add_dashboard
from holoscan_radar.applications.siso import BuiltGraph

import holoscan as hs


def build_graph(config: str | Path, *, choose_presenter: bool = False) -> BuiltGraph:
    """Compose a MIMO DDM graph from a validated linear or nonlinear preset."""

    path = str(Path(config).resolve())
    cfg = _radar._load_ddm(path)
    cfg["visualization"]["x11_presentation"] = _radar._resolve_x11_presentation(
        cfg["visualize"], cfg["visualization"]["x11_presentation"], choose_presenter, path
    )
    device = _radar._select_application_device(
        cfg["visualize"], cfg["visualize"], cfg["visualization"]["x11_presentation"]
    )
    graph = hs.Graph("mimo_ddm")
    options = hs.CompileOptions()
    extents = cfg["extents"]
    source = graph.op("holoscan.radar._app.DdmSource", "adc_source", path=path)
    range_op = FMCWRangeProcessorOpF32(
        graph, "range_processor", extents=extents, **cfg["range_processor"]
    )
    doppler = FMCWDopplerProcessorOpF32(
        graph, "doppler_processor", extents=extents, **cfg["doppler_processor"]
    )
    demux = DDMDenseDemultiplexOpF32(
        graph, "ddm_dense_demultiplex", extents=extents, **cfg["ddm_dense_demultiplex"]
    )
    notch = DopplerNotchOpF32(graph, "doppler_notch", extents=extents, **cfg["doppler_notch"])
    vaa = VirtualArrayAssemblyOpF32(
        graph, "virtual_array_assembly", extents=extents, **cfg["virtual_array_assembly"]
    )
    cfar = CFARDetectorOpF32(graph, "cfar_detector", extents=extents, **cfg["cfar_detector"])
    gather = SnapshotGatherOpF32(
        graph, "snapshot_gather", extents=extents, **cfg["snapshot_gather"]
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
    report = graph.op("holoscan.radar._app.DdmReport", "cluster_report", path=path)
    application = graph.partition("application")
    application.add(
        source,
        range_op,
        doppler,
        demux,
        notch,
        vaa,
        cfar,
        gather,
        beamscan,
        features,
        cluster,
        reduction,
        report,
    )
    # Range/Doppler process, demultiplex TX codes, and assemble the virtual array.
    graph.add_flow(source.adc_cube, range_op.adc_cube, queue_depth=4)
    graph.add_flow(range_op.range_profiles, doppler.range_profiles, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, demux.velocity_bins, queue_depth=4)
    graph.add_flow(doppler.rd_maps, demux.rd_maps, queue_depth=4)
    graph.add_flow(demux.rd_maps_out, notch.rd_maps, queue_depth=4)
    graph.add_flow(demux.velocity_bins_out, notch.velocity_bins, queue_depth=4)
    graph.add_flow(notch.rd_maps_out, vaa.rd_maps, queue_depth=4)

    # Detect first, then gather coherent snapshots and geometry for angle.
    graph.add_flow(vaa.rd_maps_out, cfar.rd_maps, queue_depth=4)
    graph.add_flow(vaa.rd_maps_out, gather.rd_maps, queue_depth=4)
    graph.add_flow(cfar.detection_idx, gather.detection_idx, queue_depth=4)
    graph.add_flow(gather.snapshot, beamscan.snapshot, queue_depth=4)
    graph.add_flow(gather.is_valid, beamscan.is_valid, queue_depth=4)
    graph.add_flow(vaa.array_geometry, beamscan.array_geometry, queue_depth=4)

    # Resolve detections against the post-demultiplex velocity axis.
    graph.add_flow(cfar.detection_idx, features.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, features.range_bins, queue_depth=4)
    graph.add_flow(demux.velocity_bins_out, features.velocity_bins, queue_depth=4)
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
    graph.add_flow(demux.velocity_bins_out, report.velocity_bins, queue_depth=4)
    outputs = {
        "adc_source": ("adc_cube",),
        "range_processor": ("range_profiles", "range_bins"),
        "doppler_processor": ("rd_maps", "velocity_bins"),
        "ddm_dense_demultiplex": ("velocity_bins_out", "rd_maps_out"),
        "doppler_notch": ("rd_maps_out",),
        "virtual_array_assembly": ("rd_maps_out",),
        "cfar_detector": (
            "detection_mask",
            "detection_score",
            "noise_estimate",
            "detection_idx",
            "cfar_power",
        ),
        "snapshot_gather": ("snapshot", "is_valid"),
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
                device=hs.execution.DeviceId(device),
            )
    if cfg["visualize"]:
        add_dashboard(
            graph,
            application,
            options,
            cfg,
            path,
            "Ddm",
            "MIMO DDM",
            {
                "cfar_power": cfar.cfar_power,
                "detection_idx": cfar.detection_idx,
                "points": features.points,
                "range_bins": range_op.range_bins,
                "velocity_bins": demux.velocity_bins_out,
                "labels": cluster.labels,
                "sizes": cluster.sizes,
                "reduced_points": reduction.reduced_points,
            },
            device=device,
            clustered=True,
        )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return BuiltGraph(graph, options, cfg["num_frames"])
