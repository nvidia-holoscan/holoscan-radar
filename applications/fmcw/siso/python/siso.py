# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SISO range-Doppler Graph matching the C++ SISO workflow."""

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

from holoscan_radar import (
    CFARDetectorOpF32,
    ClusterDBSCANOpF32,
    ClutterFilterOpF32,
    DetectionReductionOpF32,
    FeatureAssemblyOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    _radar,
)
from holoscan_radar.applications.fmcw_dashboard import add_dashboard

import holoscan as hs


@dataclass(frozen=True)
class BuiltGraph:
    graph: hs.Graph
    options: hs.CompileOptions
    num_frames: int

    def run(self) -> hs.RunSession:
        """Run until all report frames arrive, then stop clock-driven nodes."""

        _radar._reset_app_frames(self.num_frames)
        plan = hs.compile(self.graph, self.options)
        if not plan.ok:
            raise RuntimeError(f"Graph compile failed: {plan.json()}")
        session = hs.run_async(plan)
        seen = 0
        last_progress = monotonic()
        try:
            while (
                _radar._app_frames_captured() < self.num_frames
                and session.termination == hs.RunTermination.RUNNING
            ):
                count = _radar._app_frames_captured()
                if count != seen:
                    seen = count
                    last_progress = monotonic()
                if session.diagnostics:
                    raise RuntimeError(f"Graph diagnostics: {session.diagnostics}")
                if monotonic() - last_progress >= 120:
                    raise TimeoutError(f"Graph stalled after {seen}/{self.num_frames} frames")
                sleep(0.05)
        finally:
            session.request_stop()
            session.wait()
        if session.diagnostics:
            raise RuntimeError(f"Graph diagnostics: {session.diagnostics}")
        return session


def build_graph(config: str | Path, *, choose_presenter: bool = False) -> BuiltGraph:
    """Load the shipped YAML through the C++ loader and compose its SISO Graph."""

    path = str(Path(config).resolve())
    cfg = _radar._load_siso(path)
    cfg["visualization"]["x11_presentation"] = _radar._resolve_x11_presentation(
        cfg["visualize"], cfg["visualization"]["x11_presentation"], choose_presenter, path
    )
    device = _radar._select_application_device(
        cfg["visualize"], cfg["visualize"], cfg["visualization"]["x11_presentation"]
    )
    graph = hs.Graph("siso")
    options = hs.CompileOptions()
    extents = cfg["extents"]

    source = graph.op("holoscan.radar._app.SisoSource", "adc_source", path=path)
    range_op = FMCWRangeProcessorOpF32(
        graph, "range_processor", extents=extents, **cfg["range_processor"]
    )
    clutter = ClutterFilterOpF32(graph, "clutter_filter", extents=extents, **cfg["clutter_filter"])
    doppler = FMCWDopplerProcessorOpF32(
        graph, "doppler_processor", extents=extents, **cfg["doppler_processor"]
    )
    cfar = CFARDetectorOpF32(graph, "cfar_detector", extents=extents, **cfg["cfar_detector"])
    features = FeatureAssemblyOpF32(
        graph, "feature_assembly", extents=extents, features=cfg["feature_assembly"]["features"]
    )
    cluster = ClusterDBSCANOpF32(graph, "cluster_dbscan", extents=extents, **cfg["cluster_dbscan"])
    reduction = DetectionReductionOpF32(
        graph, "detection_reduction", extents=extents, **cfg["detection_reduction"]
    )
    report = graph.op("holoscan.radar._app.SisoReport", "cluster_report", path=path)

    application = graph.partition("application")
    application.add(source, range_op, clutter, doppler, cfar, features, cluster, reduction, report)
    # FMCW front end: ADC cube to range-Doppler map to detections.
    graph.add_flow(source.adc_cube, range_op.adc_cube, queue_depth=4)
    graph.add_flow(range_op.range_profiles, clutter.range_profiles, queue_depth=4)
    graph.add_flow(clutter.range_profiles_out, doppler.range_profiles, queue_depth=4)
    graph.add_flow(doppler.rd_maps, cfar.rd_maps, queue_depth=4)

    # Resolve detected bins against the physical range and velocity axes.
    graph.add_flow(cfar.detection_idx, features.detection_idx, queue_depth=4)
    graph.add_flow(range_op.range_bins, features.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, features.velocity_bins, queue_depth=4)

    # Cluster complete feature rows and select one result per cluster.
    graph.add_flow(features.points, cluster.points, queue_depth=4)
    graph.add_flow(cluster.labels, reduction.labels, queue_depth=4)
    graph.add_flow(cfar.detection_idx, reduction.detection_idx, queue_depth=4)
    graph.add_flow(features.points, reduction.points, queue_depth=4)

    # Report clusters with the same physical coordinate axes.
    graph.add_flow(features.points, report.points, queue_depth=4)
    graph.add_flow(cluster.labels, report.labels, queue_depth=4)
    graph.add_flow(cluster.sizes, report.sizes, queue_depth=4)
    graph.add_flow(reduction.rep_idx, report.detection_idx, queue_depth=4)
    graph.add_flow(reduction.reduced_points, report.reduced_points, queue_depth=4)
    graph.add_flow(range_op.range_bins, report.range_bins, queue_depth=4)
    graph.add_flow(doppler.velocity_bins, report.velocity_bins, queue_depth=4)

    outputs = {
        "adc_source": ("adc_cube",),
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
            "Siso",
            "SISO",
            {
                "cfar_power": cfar.cfar_power,
                "detection_idx": cfar.detection_idx,
                "points": features.points,
                "range_bins": range_op.range_bins,
                "velocity_bins": doppler.velocity_bins,
                "labels": cluster.labels,
                "sizes": cluster.sizes,
                "reduced_points": reduction.reduced_points,
            },
            device=device,
            clustered=True,
        )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return BuiltGraph(graph, options, cfg["num_frames"])
