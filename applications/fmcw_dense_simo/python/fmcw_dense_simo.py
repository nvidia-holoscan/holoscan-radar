# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SIMO FMCW dense angle-then-cluster pipeline (Python):
#
#   SimAdcSourceOp -> Range -> Clutter -> Doppler
#     -> PhaseShiftBeamformerOp -> CFARDetectorOp (RDA cube)
#     -> AnglePeakPickOp -> CubeAoaLookupOp -> FeatureAssembly [r,v,az,el]
#     -> ClusterDBSCAN -> DetectionReduction -> DenseClusterReportOp
#     [-> viz via RdChannelSliceOp, ClusterOverlayOp, RollingPlotOp, BoresightViewOp]
#
# Configuration is loaded from fmcw_dense_simo.yaml (default: cpp copy). This
# file is the graph only - AppConfig / load_config live in config.py, the
# dense-path bridges in custom_ops.py (both colocated); operator blocks flow
# through Holoscan's from_config().

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
from holoscan.radar import BoresightViewOp  # noqa: E402
from holoscan.radar import install_matx_cache_atexit  # noqa: E402
from holoscan.radar import (
    AssemblyFeature,
    CadenceMeter,
    CFARDetectorOp,
    ClusterDBSCANOp,
    ClusterOverlayOp,
    ClutterFilterOp,
    DetectionReductionOp,
    DetectionReportOp,
    FeatureAssemblyOp,
    FMCWDopplerProcessorOp,
    FMCWRangeProcessorOp,
    NullSinkOp,
    PanelRect,
    PhaseShiftBeamformerOp,
    RdChannelSliceOp,
    RdMapToImageOp,
    RollingPlotOp,
    cluster_palette,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402,F401

import config  # noqa: E402 (colocated, script-local - see config.py docstring)
import custom_ops  # noqa: E402 (colocated, script-local - see config.py docstring)
from common.python.demo_config import (  # noqa: E402
    detection_layer_specs,
    grid_extent,
    make_view,
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
K_BORESIGHT_PLOT_LAYER = "boresight_plot"
K_BORESIGHT_TITLE_LAYER = "boresight_title"


class FmcwDenseSimoApp(Application):
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
        beamformer = PhaseShiftBeamformerOp(
            self,
            self.from_config("phase_shift_beamformer"),
            array_geometry=cfg.array_geometry,
            name="phase_shift_beamformer",
        )
        # peak_only is a native cfar_detector param (from_config), applied
        # per-beam (nci_axes=[]); AnglePeakPickOp below still does the cross-beam
        # collapse. The app reads cfg.peak_only to drop the clustering subgraph.
        cfar = CFARDetectorOp(self, self.from_config("cfar_detector"), name="cfar_detector")
        peak_pick = custom_ops.AnglePeakPickOp(self, name="angle_peak_pick")
        aoa_lookup = custom_ops.CubeAoaLookupOp(
            self,
            az_grid=cfg.az_angles,
            el_grid=cfg.el_angles,
            name="cube_aoa_lookup",
        )

        self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
        self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
        self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
        self.add_flow(doppler, beamformer, {("rd_maps", "rd_maps")})
        self.add_flow(beamformer, cfar, {("rda_cube", "rd_maps")})

        self.add_flow(
            cfar,
            peak_pick,
            {
                ("detection_idx", "detection_idx"),
                ("detection_score", "detection_score"),
            },
        )
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        self.add_flow(peak_pick, aoa_lookup, {("detection_idx", "detection_idx")})
        row_index_sink = NullSinkOp(self, name="aoa_row_index_sink")
        self.add_flow(aoa_lookup, row_index_sink, {("row_index", "input")})

        # Clustering subgraph — dropped when peak_only is set. The cube peaks
        # (already one per RD cell after AnglePeakPickOp) are reported directly
        # via DetectionReportOp: CubeAoaLookupOp supplies az/el and the 4-col
        # detection_idx is sliced to (doppler, range). The dense score is a
        # rank-4 per-beam cube with no 2-D map, so it is omitted from the report.
        cluster = None
        reduction = None
        if not cfg.peak_only:
            feature_assembly = FeatureAssemblyOp(
                self,
                features=[
                    AssemblyFeature.RANGE,
                    AssemblyFeature.VELOCITY,
                    AssemblyFeature.AZ,
                    AssemblyFeature.EL,
                ],
                name="feature_assembly",
            )
            cluster = ClusterDBSCANOp(
                self, self.from_config("cluster_dbscan"), name="cluster_dbscan"
            )
            reduction = DetectionReductionOp(self, self.from_config("reduction"), name="reduction")
            report = custom_ops.DenseClusterReportOp(
                self,
                az_grid=cfg.az_angles,
                el_grid=cfg.el_angles,
                cadence=cadence,
                name="dense_cluster_report",
            )

            self.add_flow(aoa_lookup, feature_assembly, {("aoa_estimates", "aoa_estimates")})
            self.add_flow(peak_pick, feature_assembly, {("detection_idx", "detection_idx")})
            self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
            self.add_flow(doppler, feature_assembly, {("velocity_bins", "velocity_bins")})
            self.add_flow(feature_assembly, cluster, {("points", "points")})

            self.add_flow(cluster, reduction, {("labels", "labels")})
            self.add_flow(peak_pick, reduction, {("detection_idx", "detection_idx")})
            self.add_flow(feature_assembly, reduction, {("points", "points")})

            self.add_flow(feature_assembly, report, {("points", "points")})
            self.add_flow(cluster, report, {("labels", "labels"), ("sizes", "sizes")})
            self.add_flow(reduction, report, {("rep_idx", "rep_idx")})
            self.add_flow(range_op, report, {("range_bins", "range_bins")})
            self.add_flow(doppler, report, {("velocity_bins", "velocity_bins")})
        else:
            det_report = DetectionReportOp(self, name="detection_report")
            if cadence is not None:
                det_report.cadence = cadence
            det_idx_rd = custom_ops.TrailingIdxColumnsOp(self, name="det_idx_rd_report_slice")
            self.add_flow(peak_pick, det_idx_rd, {("detection_idx", "idx")})
            self.add_flow(det_idx_rd, det_report, {("idx", "detection_idx")})
            self.add_flow(aoa_lookup, det_report, {("aoa_estimates", "aoa_estimates")})
            self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
            self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})

        if cfg.visualize:
            self._compose_display(doppler, peak_pick, cluster, reduction, range_op)

    def _compose_display(self, doppler, detections, cluster, reduction, range_op):
        cfg = self.cfg
        ch_slice = RdChannelSliceOp(self, channel=0, name="rd_channel_slice")
        to_image = RdMapToImageOp(self, name="rd_map_to_image")
        overlay = ClusterOverlayOp(self, name="cluster_overlay")
        overlay.view_aspect = rd_map_view_aspect(
            1200 if cfg.plots.enable else 800,
            760 if cfg.plots.enable else 600,
            0.60,
            cfg.plots.enable,
        )
        overlay.frame_period = cfg.frame_period  # scenario clock
        overlay.marker_radius = cfg.viz.marker_radius
        overlay.marker_outline = cfg.viz.marker_outline
        overlay.marker_segments = cfg.viz.marker_segments
        det_idx_rd = custom_ops.TrailingIdxColumnsOp(self, name="det_idx_rd_slice")

        input_specs: list[HolovizOp.InputSpec] = []
        input_specs.append(HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT))
        palette = cluster_palette()
        input_specs.extend(detection_layer_specs(palette))

        plot: RollingPlotOp | None = None
        boresight: BoresightViewOp | None = None
        rep_idx_rd: custom_ops.TrailingIdxColumnsOp | None = None
        rep_aoa: custom_ops.CubeAoaLookupOp | None = None

        if cfg.plots.enable:
            left = make_view(0.0, 0.0, 0.60, 1.0)
            for spec in input_specs:
                spec.views = [left]

            # Side panels are native Holoviz primitives (dynamic InputSpecs);
            # stacked right-column view rects.
            range_view = PanelRect(0.63, 0.02, 0.35, 0.30)
            velocity_view = PanelRect(0.63, 0.35, 0.35, 0.30)
            boresight_view = PanelRect(0.63, 0.68, 0.35, 0.30)

            def panel_aspect(v: PanelRect) -> float:
                return (v.w * 1200.0) / (v.h * 760.0)

            plot = RollingPlotOp(
                self,
                window=cfg.plots.window,
                frame_period=cfg.frame_period,
                name="rolling_plot",
            )
            plot.velocity_ylim_lo = cfg.plots.velocity_ylim_lo
            plot.velocity_ylim_hi = cfg.plots.velocity_ylim_hi
            plot.primitive = True
            plot.range_view = range_view
            plot.velocity_view = velocity_view
            plot.range_aspect = panel_aspect(range_view)
            plot.velocity_aspect = panel_aspect(velocity_view)
            plot.marker_radius = cfg.viz.marker_radius
            plot.marker_outline = cfg.viz.marker_outline
            plot.marker_segments = cfg.viz.marker_segments
            rep_idx_rd = custom_ops.TrailingIdxColumnsOp(self, name="rep_idx_rd_slice")
            rep_aoa = custom_ops.CubeAoaLookupOp(
                self,
                az_grid=cfg.az_angles,
                el_grid=cfg.el_angles,
                name="cube_aoa_lookup_rep",
            )
            az_lo, az_hi = grid_extent(cfg.az_angles, -0.35, 0.35)
            el_lo, el_hi = grid_extent(cfg.el_angles, -0.1, 0.1)
            boresight = BoresightViewOp(
                self,
                az_lo=az_lo,
                az_hi=az_hi,
                el_lo=el_lo,
                el_hi=el_hi,
                name="boresight_view",
            )
            boresight.primitive = True
            boresight.view = boresight_view
            boresight.aspect = panel_aspect(boresight_view)
            boresight.marker_radius = cfg.viz.marker_radius
            boresight.marker_outline = cfg.viz.marker_outline
            boresight.marker_segments = cfg.viz.marker_segments

        visualizer = HolovizOp(
            self,
            tensors=input_specs,
            color_lut=make_turbo_lut(),
            window_title="FMCW SIMO dense - RD (ch0) + clusters",
            width=1200 if cfg.plots.enable else 800,
            height=760 if cfg.plots.enable else 600,
            vsync=cfg.viz.vsync,
            headless=cfg.headless,
            name="holoviz",
        )

        self.add_flow(doppler, ch_slice, {("rd_maps", "rd_maps")})
        self.add_flow(ch_slice, to_image, {("rd_map", "rd_maps")})
        self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
        self.add_flow(detections, det_idx_rd, {("detection_idx", "idx")})
        self.add_flow(det_idx_rd, overlay, {("idx", "detection_idx")})
        self.add_flow(ch_slice, overlay, {("rd_map", "rd_maps")})
        # Cluster labels color the dots when clustering runs; peak_only has no
        # labels, so the overlay colors each peak by its row index instead.
        if not cfg.peak_only:
            self.add_flow(cluster, overlay, {("labels", "labels")})
            if rep_idx_rd is not None:
                # Color clusters by their representative RD cell (stable across
                # frames) via the same sliced rep the plot uses; falls back to
                # the DBSCAN label when plots are off (rep_idx_rd is None).
                self.add_flow(rep_idx_rd, overlay, {("idx", "rep_idx")})
        self.add_flow(overlay, visualizer, {("overlay", "receivers")})
        # Scenario clock overlay: its spec heads the primitive-panel InputSpec
        # chain when panels are on (overlay → rolling plot → boresight →
        # input_specs), else feeds input_specs directly.
        if plot is not None:
            self.add_flow(overlay, plot, {("time_specs", "specs_in")})
        else:
            self.add_flow(overlay, visualizer, {("time_specs", "input_specs")})

        # Plot/boresight read the per-cluster representative (clustered) or the
        # raw cube peaks (peak_only); both carry the 4-col idx.
        rep_src = reduction if not cfg.peak_only else detections
        rep_port = "rep_idx" if not cfg.peak_only else "detection_idx"
        if plot is not None and rep_idx_rd is not None and rep_aoa is not None:
            self.add_flow(rep_src, rep_idx_rd, {(rep_port, "idx")})
            self.add_flow(rep_idx_rd, plot, {("idx", "rep_idx")})
            if not cfg.peak_only:
                # labels flag clustered mode; the plot then colors each rep by
                # its RD cell (matching the RD-map / boresight markers).
                self.add_flow(cluster, plot, {("labels", "labels")})
            self.add_flow(range_op, plot, {("range_bins", "range_bins")})
            self.add_flow(doppler, plot, {("velocity_bins", "velocity_bins")})
            self.add_flow(plot, visualizer, {("plots", "receivers")})
        if boresight is not None and rep_aoa is not None:
            self.add_flow(rep_src, rep_aoa, {(rep_port, "detection_idx")})
            if not cfg.peak_only and rep_idx_rd is not None:
                # Match the overlay/plot: color each cluster by its rep RD cell.
                # row_index (passed as labels) indexes rep_idx_rd row-for-row.
                self.add_flow(rep_idx_rd, boresight, {("idx", "rep_idx")})
            self.add_flow(
                rep_aoa,
                boresight,
                {("aoa_estimates", "aoa_estimates"), ("row_index", "labels")},
            )
            self.add_flow(boresight, visualizer, {("plot", "receivers")})
            # Primitive-panel InputSpec chain: rolling specs → boresight →
            # HolovizOp.input_specs.
            self.add_flow(plot, boresight, {("plot_specs", "specs_in")})
            self.add_flow(boresight, visualizer, {("specs_out", "input_specs")})


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    cpp_dir = Path(__file__).resolve().parent.parent / "cpp"
    return run_fmcw_app(
        FmcwDenseSimoApp,
        argv,
        "fmcw_dense_simo",
        cpp_dir,
        "fmcw_dense_simo.yaml",
        config.load_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
