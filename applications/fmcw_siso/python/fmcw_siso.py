# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SISO FMCW pipeline example (Python): synthetic ADC source feeding
#
#   SimAdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
#                  -> FMCWDopplerProcessorOp -> CFARDetectorOp
#                                           -> DetectionReportOp
#                                           -> FeatureAssemblyOp
#                                                -> ClusterDBSCANOp
#                                                     -> DetectionReductionOp
#                                                          -> ClusterReportOp
#                                            [-> RdMapToImageOp   -> HolovizOp]
#                                            [-> ClusterOverlayOp -> HolovizOp]
#                                            [-> RollingPlotOp    -> HolovizOp]
#
# Configuration is loaded from fmcw_siso.yaml (default: the copy under
# applications/fmcw_siso/cpp/). This file is the graph only - AppConfig /
# load_config live in config.py (colocated); the ADC source and the app
# entry-point scaffolding are shared across apps (applications/common/python/).

from __future__ import annotations

import os
import sys
from pathlib import Path

_BUILD_DIR = os.environ.get("HOLOSCAN_RADAR_BUILD_DIR")
if _BUILD_DIR:
    import holoscan as _holoscan_pkg

    _build_holoscan = os.path.join(_BUILD_DIR, "python", "lib", "holoscan")
    if _build_holoscan not in _holoscan_pkg.__path__:
        _holoscan_pkg.__path__.insert(0, _build_holoscan)

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# Load holoscan.radar before cupy (MatX CUDA cache / atexit ordering), and
# before importing config.py (its common.python.sim import pulls in cupy).
# isort: off
from holoscan.conditions import CountCondition, PeriodicCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.operators import HolovizOp  # noqa: E402
from holoscan.radar import CadenceMeter  # noqa: E402
from holoscan.radar import (
    AssemblyFeature,
    CFARDetectorOp,
    ClusterDBSCANOp,
    ClusterOverlayOp,
    ClusterReportOp,
    ClutterFilterOp,
    DetectionReductionOp,
    DetectionReportOp,
    FeatureAssemblyOp,
    FMCWDopplerProcessorOp,
    FMCWRangeProcessorOp,
    NullSinkOp,
    PanelRect,
    RdMapToImageOp,
    RollingPlotOp,
    cluster_palette,
    install_matx_cache_atexit,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402,F401

import config  # noqa: E402 (colocated, script-local - see config.py docstring)
from common.python.demo_config import (  # noqa: E402
    detection_layer_specs,
    make_view,
    rd_map_box,
    rd_map_region,
    rd_map_view_aspect,
)
from common.python.driver import run_fmcw_app  # noqa: E402
from common.python.ops import SimAdcSourceOp  # noqa: E402

# isort: on

install_matx_cache_atexit()

K_NUM_CLUSTER_COLORS = 30
K_NOISE_LAYER_NAME = "noise"
K_RANGE_PLOT_LAYER = "range_plot"
K_VELOCITY_PLOT_LAYER = "velocity_plot"
K_RANGE_TITLE_LAYER = "range_title"
K_VELOCITY_TITLE_LAYER = "velocity_title"


class FmcwSisoApp(Application):
    def __init__(self, cfg: config.AppConfig):
        super().__init__()
        self.cfg = cfg
        self.cadence: CadenceMeter | None = None

    def compose(self) -> None:
        cfg = self.cfg
        period_ns = int(cfg.frame_period * 1e9)

        cadence = None
        if cfg.meter_cadence:
            cadence = CadenceMeter()
            cadence.frame_period_s = cfg.frame_period
            cadence.log_every = cfg.meter_log_every
            self.cadence = cadence

        source = SimAdcSourceOp(
            self,
            CountCondition(self, count=cfg.num_frames),
            PeriodicCondition(self, recess_period=period_ns),
            scene=cfg.scene,
            frame_period=cfg.frame_period,
            cadence=cadence,
            name="adc_source",
        )

        self.config(str(cfg.config_path))

        range_op = FMCWRangeProcessorOp(
            self, self.from_config("range_processor"), name="range_processor"
        )
        clutter = ClutterFilterOp(self, self.from_config("clutter_filter"), name="clutter_filter")
        doppler = FMCWDopplerProcessorOp(
            self, self.from_config("doppler_processor"), name="doppler_processor"
        )
        # peak_only is a native cfar_detector param (from_config); the app also
        # reads it (cfg.peak_only) to drop the clustering subgraph below.
        cfar = CFARDetectorOp(self, self.from_config("cfar_detector"), name="cfar_detector")
        det_report = DetectionReportOp(self, name="detection_report")
        if cadence is not None:
            det_report.cadence = cadence
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")

        self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
        self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
        self.add_flow(clutter, doppler, {("range_profiles_out", "range_profiles")})
        self.add_flow(doppler, cfar, {("rd_maps", "rd_maps")})
        self.add_flow(cfar, det_report, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, det_report, {("detection_score", "detection_score")})
        self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
        self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        # Clustering subgraph — dropped entirely when peak_only is set (CFAR then
        # emits one detection per target, so DBSCAN + reduction are redundant and
        # DetectionReportOp above reports the CFAR peaks directly). cluster /
        # reduction stay None in that mode and the viz block below adapts.
        cluster = None
        reduction = None
        if not cfg.peak_only:
            feature_assembly = FeatureAssemblyOp(
                self,
                features=[AssemblyFeature.RANGE, AssemblyFeature.VELOCITY],
                name="feature_assembly",
            )
            cluster = ClusterDBSCANOp(
                self, self.from_config("cluster_dbscan"), name="cluster_dbscan"
            )
            cluster_report = ClusterReportOp(self, name="cluster_report")
            reduction = DetectionReductionOp(
                self, self.from_config("reduction"), name="detection_reduction"
            )

            self.add_flow(cfar, feature_assembly, {("detection_idx", "detection_idx")})
            self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
            self.add_flow(doppler, feature_assembly, {("velocity_bins", "velocity_bins")})
            self.add_flow(feature_assembly, cluster, {("points", "points")})
            self.add_flow(feature_assembly, cluster_report, {("points", "points")})
            self.add_flow(cluster, cluster_report, {("labels", "labels")})
            self.add_flow(cluster, cluster_report, {("sizes", "sizes")})

            self.add_flow(cluster, reduction, {("labels", "labels")})
            self.add_flow(cfar, reduction, {("detection_idx", "detection_idx")})
            self.add_flow(feature_assembly, reduction, {("points", "points")})
            self.add_flow(reduction, cluster_report, {("rep_idx", "rep_idx")})
            self.add_flow(range_op, cluster_report, {("range_bins", "range_bins")})
            self.add_flow(doppler, cluster_report, {("velocity_bins", "velocity_bins")})

        if cfg.visualize:
            to_image = RdMapToImageOp(self, name="rd_map_to_image")
            overlay = ClusterOverlayOp(self, name="cluster_overlay")
            overlay.view_aspect = rd_map_view_aspect(
                1200 if cfg.plots.enable else 800, 600, 0.60, cfg.plots.enable
            )
            overlay.frame_period = cfg.frame_period  # scenario clock
            overlay.rd_region = PanelRect(
                *rd_map_region(cfg.plots.enable, 0.60)
            )  # axes/frame/title region
            overlay.rd_view_px_height = 600.0
            overlay.marker_radius = cfg.viz.marker_radius
            overlay.marker_outline = cfg.viz.marker_outline
            overlay.marker_segments = cfg.viz.marker_segments

            input_specs: list[HolovizOp.InputSpec] = []
            rd_spec = HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
            input_specs.append(rd_spec)

            palette = cluster_palette()
            if len(palette) != K_NUM_CLUSTER_COLORS:
                raise RuntimeError(
                    "fmcw_siso: cluster palette size must equal " f"{K_NUM_CLUSTER_COLORS}"
                )

            input_specs.extend(detection_layer_specs(palette))

            # Inset the RD image + discs into the framed box (leaving margins for
            # the axes/frame/title that ClusterOverlayOp draws).
            box = make_view(*rd_map_box(cfg.plots.enable, 0.60))
            for spec in input_specs:
                spec.views = [box]

            plot: RollingPlotOp | None = None
            if cfg.plots.enable:
                # Side panels are native Holoviz primitives emitted by
                # RollingPlotOp (dynamic InputSpecs); no static panel layers.
                range_view = PanelRect(0.63, 0.04, 0.35, 0.44)
                velocity_view = PanelRect(0.63, 0.52, 0.35, 0.44)

                def panel_aspect(v: PanelRect) -> float:
                    return (v.w * 1200.0) / (v.h * 600.0)

                plot = RollingPlotOp(
                    self,
                    window=cfg.plots.window,
                    frame_period=cfg.frame_period,
                    name="rolling_plot",
                )
                plot.primitive = True
                plot.range_view = range_view
                plot.velocity_view = velocity_view
                plot.range_aspect = panel_aspect(range_view)
                plot.velocity_aspect = panel_aspect(velocity_view)
                plot.marker_radius = cfg.viz.marker_radius
                plot.marker_outline = cfg.viz.marker_outline
                plot.marker_segments = cfg.viz.marker_segments

            visualizer = HolovizOp(
                self,
                tensors=input_specs,
                color_lut=make_turbo_lut(),
                window_title="FMCW SISO - Range-Doppler + clusters",
                width=1200 if cfg.plots.enable else 800,
                height=600,
                vsync=cfg.viz.vsync,
                headless=cfg.headless,
                name="holoviz",
            )

            self.add_flow(doppler, to_image, {("rd_maps", "rd_maps")})
            self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
            # Cluster labels color the dots when clustering runs; in peak_only
            # mode there are no labels, so the overlay colors each CFAR peak by
            # its row index (ClusterOverlayOp treats absent labels that way).
            self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
            self.add_flow(doppler, overlay, {("rd_maps", "rd_maps")})
            # Physical axis bins for the RD-map tick labels (range X, velocity Y).
            self.add_flow(range_op, overlay, {("range_bins", "range_bins")})
            self.add_flow(doppler, overlay, {("velocity_bins", "velocity_bins")})
            if not cfg.peak_only:
                self.add_flow(cluster, overlay, {("labels", "labels")})
                # rep_idx colors each cluster by its representative RD cell
                # (stable across frames) instead of the flickery size-rank label.
                self.add_flow(reduction, overlay, {("rep_idx", "rep_idx")})
            self.add_flow(overlay, visualizer, {("overlay", "receivers")})
            # Scenario clock overlay: its spec heads the primitive-panel
            # InputSpec chain when panels are on (→ rolling plot), else feeds
            # input_specs directly.
            if plot is not None:
                self.add_flow(overlay, plot, {("time_specs", "specs_in")})
            else:
                self.add_flow(overlay, visualizer, {("time_specs", "input_specs")})

            if plot is not None:
                # Rolling plots read the per-cluster representative (clustered) or
                # the raw CFAR peaks (peak_only); both are (N, 2) (doppler, range).
                if not cfg.peak_only:
                    self.add_flow(reduction, plot, {("rep_idx", "rep_idx")})
                    # labels flag clustered mode; the plot then colors each rep by
                    # its RD cell (matching the RD-map / boresight markers).
                    self.add_flow(cluster, plot, {("labels", "labels")})
                else:
                    self.add_flow(cfar, plot, {("detection_idx", "rep_idx")})
                self.add_flow(range_op, plot, {("range_bins", "range_bins")})
                self.add_flow(doppler, plot, {("velocity_bins", "velocity_bins")})
                self.add_flow(plot, visualizer, {("plots", "receivers")})
                # Dynamic InputSpecs for the primitive panels → input_specs port.
                self.add_flow(plot, visualizer, {("plot_specs", "input_specs")})


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    cpp_dir = Path(__file__).resolve().parent.parent / "cpp"
    return run_fmcw_app(
        FmcwSisoApp, argv, "fmcw_siso", cpp_dir, "fmcw_siso.yaml", config.load_config
    )


if __name__ == "__main__":
    raise SystemExit(main())
