# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MIMO FMCW sparse angle-then-cluster pipeline (Python, TDM or DDM via YAML):
#
# TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
#        -> Compensation -> VirtualArrayAssemblyOp -> ...
# DDM: MimoDdmAdcSourceOp -> Range -> Doppler -> Compensation -> Clutter
#        -> VirtualArrayAssemblyOp -> ...
#
# Configuration is loaded from fmcw_mimo.yaml (default), fmcw_mimo_ddm.yaml,
# or fmcw_mimo_ddm_phase_codes.yaml (non-linear correlate). This file is the
# graph only - AppConfig / load_config live in config.py, the TDM/DDM ADC
# sources in custom_ops.py (both colocated). Every operator block flows
# through Holoscan's from_config() except mimo_coherent_compensation, which
# must stay manual (see demo_config.parse_mimo_coherent_compensation).

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
# before importing config.py (its common.python.mimo_sim import pulls in
# cupy).
# isort: off
from holoscan.conditions import CountCondition, PeriodicCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.operators import HolovizOp  # noqa: E402
from holoscan.radar import (  # noqa: E402
    AssemblyFeature,
    BeamscanEstimatorOp,
    BoresightViewOp,
    CadenceMeter,
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
    MIMOCoherentCompensationOp,
    MIMODeinterleaveOp,
    MultiplexingScheme,
    NullSinkOp,
    PanelRect,
    RdMapToImageOp,
    RollingPlotOp,
    SnapshotGatherOp,
    VirtualArrayAssemblyOp,
    cluster_palette,
    install_matx_cache_atexit,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402,F401

import config  # noqa: E402 (colocated, script-local - see config.py docstring)
import custom_ops  # noqa: E402 (colocated, script-local - see config.py docstring)
from common.python.demo_config import (  # noqa: E402
    detection_layer_specs,
    make_view,
    rd_map_box,
    rd_map_region,
    rd_map_view_aspect,
)
from common.python.driver import run_fmcw_app  # noqa: E402
from common.python.ops import TensorMapDiscardOp  # noqa: E402

# isort: on

install_matx_cache_atexit()

K_NOISE_LAYER_NAME = "noise"
K_RANGE_PLOT_LAYER = "range_plot"
K_VELOCITY_PLOT_LAYER = "velocity_plot"
K_RANGE_TITLE_LAYER = "range_title"
K_VELOCITY_TITLE_LAYER = "velocity_title"
K_BORESIGHT_PLOT_LAYER = "boresight_plot"
K_BORESIGHT_TITLE_LAYER = "boresight_title"


class FmcwMimoApp(Application):
    def __init__(self, cfg: config.AppConfig):
        super().__init__()
        self.cfg = cfg
        self.cadence: CadenceMeter | None = None

    def compose(self) -> None:
        cfg = self.cfg
        self.config(str(cfg.config_path))
        ddm = cfg.scheme == MultiplexingScheme.DDM
        window_title = (
            "FMCW MIMO DDM - virtual RD + clusters"
            if ddm
            else "FMCW MIMO TDM - virtual RD + clusters"
        )
        period_ns = int(cfg.frame_period * 1e9)

        cadence = None
        if cfg.meter_cadence:
            cadence = CadenceMeter()
            cadence.frame_period_s = cfg.frame_period
            cadence.log_every = cfg.meter_log_every
            self.cadence = cadence

        range_op = FMCWRangeProcessorOp(
            self, self.from_config("range_processor"), name="range_processor"
        )
        clutter = ClutterFilterOp(self, self.from_config("clutter_filter"), name="clutter_filter")
        doppler = FMCWDopplerProcessorOp(
            self, self.from_config("doppler_processor"), name="doppler_processor"
        )
        # Avoid from_config: it overwrites the app-expanded ddm_code (see
        # demo_config.parse_mimo_coherent_compensation's docstring).
        compensation = MIMOCoherentCompensationOp(
            self,
            **cfg.mimo_coherent_compensation,
        )
        vaa = VirtualArrayAssemblyOp(
            self,
            self.from_config("virtual_array_assembly"),
            tx_positions=cfg.tx_positions_flat,
            rx_positions=cfg.rx_positions_flat,
            wavelength=cfg.wavelength,
            name="virtual_array_assembly",
        )

        if ddm:
            source = custom_ops.MimoDdmAdcSourceOp(
                self,
                CountCondition(self, count=cfg.num_frames),
                PeriodicCondition(self, recess_period=period_ns),
                scene=cfg.scene,
                doppler_offsets=cfg.ddm_doppler_offsets,
                phase_codes=(list(cfg.ddm_code.phase_codes) if cfg.ddm_correlate else None),
                use_phase_codes=cfg.ddm_correlate,
                frame_period=cfg.frame_period,
                cadence=cadence,
                name="adc_source",
            )
            self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
            self.add_flow(range_op, doppler, {("range_profiles", "range_profiles")})
            self.add_flow(doppler, compensation, {("rd_maps", "rd_maps")})
            self.add_flow(compensation, clutter, {("rd_maps_out", "range_profiles")})
            self.add_flow(clutter, vaa, {("range_profiles_out", "rd_maps")})
        else:
            source = custom_ops.MimoTdmAdcSourceOp(
                self,
                CountCondition(self, count=cfg.num_frames),
                PeriodicCondition(self, recess_period=period_ns),
                scene=cfg.scene,
                tx_schedule=cfg.tx_schedule,
                frame_period=cfg.frame_period,
                cadence=cadence,
                name="adc_source",
            )
            deinterleave = MIMODeinterleaveOp(
                self, self.from_config("mimo_deinterleave"), name="mimo_deinterleave"
            )
            self.add_flow(source, deinterleave, {("adc_cube", "adc_cube")})
            self.add_flow(deinterleave, range_op, {("adc_cube_out", "adc_cube")})
            self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
            self.add_flow(clutter, doppler, {("range_profiles_out", "range_profiles")})
            self.add_flow(doppler, compensation, {("rd_maps", "rd_maps")})
            self.add_flow(compensation, vaa, {("rd_maps_out", "rd_maps")})

        # peak_only is a native cfar_detector param (from_config); the app reads
        # it (cfg.peak_only) to drop the clustering subgraph below.
        cfar = CFARDetectorOp(self, self.from_config("cfar_detector"), name="cfar_detector")
        gather = SnapshotGatherOp(self, self.from_config("snapshot_gather"), name="snapshot_gather")
        beamscan = BeamscanEstimatorOp(
            self,
            self.from_config("beamscan_estimator"),
            array_geometry=cfg.array_geometry,
            name="beamscan_estimator",
        )

        det_report = DetectionReportOp(self, name="detection_report")
        if cadence is not None:
            det_report.cadence = cadence
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")
        spectrum_sink = TensorMapDiscardOp(self, name="angle_spectrum_discard")

        ddm_velocity_bins: custom_ops.DdmReportVelocityBinsOp | None = None
        if ddm:
            ddm_velocity_bins = custom_ops.DdmReportVelocityBinsOp(
                self,
                waveform=cfg.report_waveform,
                num_doppler=cfg.report_num_doppler,
                bin_spacing_hz=(
                    (1.0 / cfg.report_waveform.chirp_repetition_interval())
                    / float(cfg.scene.num_chirps)
                ),
                name="ddm_velocity_bins",
            )
            self.add_flow(range_op, ddm_velocity_bins, {("range_bins", "sync")})

        self.add_flow(vaa, cfar, {("rd_maps_out", "rd_maps")})
        self.add_flow(vaa, gather, {("rd_maps_out", "rd_maps")})
        self.add_flow(cfar, gather, {("detection_idx", "detection_idx")})
        self.add_flow(gather, beamscan, {("snapshot", "snapshot"), ("is_valid", "is_valid")})
        self.add_flow(vaa, beamscan, {("array_geometry", "array_geometry")})
        self.add_flow(beamscan, spectrum_sink, {("angle_spectrum", "input")})

        self.add_flow(cfar, det_report, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, det_report, {("detection_score", "detection_score")})
        self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
        if ddm and ddm_velocity_bins is not None:
            self.add_flow(ddm_velocity_bins, det_report, {("velocity_bins", "velocity_bins")})
        else:
            self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        # Clustering subgraph — dropped when peak_only is set. The angle branch
        # (gather/beamscan) still runs; in peak_only its aoa feeds
        # DetectionReportOp (optional aoa_estimates) instead of feature_assembly.
        cluster = None
        reduction = None
        if not cfg.peak_only:
            feature_assembly = FeatureAssemblyOp(
                self,
                features=[
                    AssemblyFeature.RANGE,
                    AssemblyFeature.VELOCITY,
                    AssemblyFeature.AZ_ESTIMATE,
                    AssemblyFeature.EL_ESTIMATE,
                ],
                name="feature_assembly",
            )
            cluster = ClusterDBSCANOp(
                self, self.from_config("cluster_dbscan"), name="cluster_dbscan"
            )
            cluster_report = ClusterReportOp(self, name="cluster_report")
            reduction = DetectionReductionOp(self, self.from_config("reduction"), name="reduction")

            self.add_flow(beamscan, feature_assembly, {("aoa_estimates", "aoa_estimates")})
            self.add_flow(cfar, feature_assembly, {("detection_idx", "detection_idx")})
            self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
            if ddm and ddm_velocity_bins is not None:
                self.add_flow(
                    ddm_velocity_bins,
                    feature_assembly,
                    {("velocity_bins", "velocity_bins")},
                )
            else:
                self.add_flow(doppler, feature_assembly, {("velocity_bins", "velocity_bins")})
            self.add_flow(feature_assembly, cluster, {("points", "points")})
            self.add_flow(feature_assembly, cluster_report, {("points", "points")})
            self.add_flow(cluster, cluster_report, {("labels", "labels"), ("sizes", "sizes")})
            self.add_flow(cluster, reduction, {("labels", "labels")})
            self.add_flow(cfar, reduction, {("detection_idx", "detection_idx")})
            self.add_flow(feature_assembly, reduction, {("points", "points")})
            self.add_flow(reduction, cluster_report, {("rep_idx", "rep_idx")})
            self.add_flow(cfar, cluster_report, {("detection_idx", "detection_idx")})
            self.add_flow(range_op, cluster_report, {("range_bins", "range_bins")})
            if ddm and ddm_velocity_bins is not None:
                self.add_flow(
                    ddm_velocity_bins,
                    cluster_report,
                    {("velocity_bins", "velocity_bins")},
                )
            else:
                self.add_flow(doppler, cluster_report, {("velocity_bins", "velocity_bins")})
        else:
            self.add_flow(beamscan, det_report, {("aoa_estimates", "aoa_estimates")})

        if cfg.visualize:
            # The RD display is CFAR's own thresholded surface
            # (cfar_power: post-NCI integrated power here), so the picture
            # and the overlaid detections are literally the same data.
            to_image = RdMapToImageOp(self, name="rd_map_to_image")
            overlay = ClusterOverlayOp(self, name="cluster_overlay")
            overlay.view_aspect = rd_map_view_aspect(
                1200 if cfg.plots.enable else 800,
                760 if cfg.plots.enable else 600,
                0.60,
                cfg.plots.enable,
            )
            overlay.frame_period = cfg.frame_period  # scenario clock
            overlay.rd_region = PanelRect(
                *rd_map_region(cfg.plots.enable, 0.60)
            )  # axes/frame/title region
            overlay.rd_view_px_height = float(760 if cfg.plots.enable else 600)
            overlay.marker_radius = cfg.viz.marker_radius
            overlay.marker_outline = cfg.viz.marker_outline
            overlay.marker_segments = cfg.viz.marker_segments

            input_specs: list[HolovizOp.InputSpec] = []
            input_specs.append(HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT))
            palette = cluster_palette()
            input_specs.extend(detection_layer_specs(palette))

            # Inset the RD image + discs into the framed box (leaving margins for
            # the axes/frame/title that ClusterOverlayOp draws).
            box = make_view(*rd_map_box(cfg.plots.enable, 0.60))
            for spec in input_specs:
                spec.views = [box]

            plot: RollingPlotOp | None = None
            boresight: BoresightViewOp | None = None
            if cfg.plots.enable:
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
                boresight = BoresightViewOp(
                    self,
                    az_lo=cfg.az_lo,
                    az_hi=cfg.az_hi,
                    el_lo=cfg.el_lo,
                    el_hi=cfg.el_hi,
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
                window_title=window_title,
                width=1200 if cfg.plots.enable else 800,
                height=760 if cfg.plots.enable else 600,
                vsync=cfg.viz.vsync,
                headless=cfg.headless,
                name="holoviz",
            )

            self.add_flow(cfar, to_image, {("cfar_power", "rd_maps")})
            self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
            # Overlay/plot/boresight color or index by cluster when clustering
            # runs; in peak_only mode they fall back to the raw CFAR peaks.
            self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
            self.add_flow(cfar, overlay, {("cfar_power", "rd_maps")})
            # Physical axis bins for the RD-map tick labels (range X, velocity Y);
            # velocity comes from the DDM report bins in DDM mode, like the plot.
            self.add_flow(range_op, overlay, {("range_bins", "range_bins")})
            if ddm and ddm_velocity_bins is not None:
                self.add_flow(ddm_velocity_bins, overlay, {("velocity_bins", "velocity_bins")})
            else:
                self.add_flow(doppler, overlay, {("velocity_bins", "velocity_bins")})
            if not cfg.peak_only:
                self.add_flow(cluster, overlay, {("labels", "labels")})
                # rep_idx colors each cluster by its representative RD cell
                # (stable across frames) instead of the flickery size-rank label.
                self.add_flow(reduction, overlay, {("rep_idx", "rep_idx")})
            self.add_flow(overlay, visualizer, {("overlay", "receivers")})
            # Scenario clock overlay: its spec heads the primitive-panel
            # InputSpec chain when panels are on (overlay → rolling plot →
            # boresight → input_specs), else feeds input_specs directly.
            if plot is not None:
                self.add_flow(overlay, plot, {("time_specs", "specs_in")})
            else:
                self.add_flow(overlay, visualizer, {("time_specs", "input_specs")})

            if plot is not None:
                if not cfg.peak_only:
                    self.add_flow(reduction, plot, {("rep_idx", "rep_idx")})
                    # labels flag clustered mode; the plot then colors each rep by
                    # its RD cell (matching the RD-map / boresight markers).
                    self.add_flow(cluster, plot, {("labels", "labels")})
                else:
                    self.add_flow(cfar, plot, {("detection_idx", "rep_idx")})
                self.add_flow(range_op, plot, {("range_bins", "range_bins")})
                if ddm and ddm_velocity_bins is not None:
                    self.add_flow(ddm_velocity_bins, plot, {("velocity_bins", "velocity_bins")})
                else:
                    self.add_flow(doppler, plot, {("velocity_bins", "velocity_bins")})
                self.add_flow(plot, visualizer, {("plots", "receivers")})
            if boresight is not None:
                self.add_flow(beamscan, boresight, {("aoa_estimates", "aoa_estimates")})
                if not cfg.peak_only:
                    self.add_flow(cluster, boresight, {("labels", "labels")})
                    self.add_flow(reduction, boresight, {("rep_idx", "rep_idx")})
                self.add_flow(boresight, visualizer, {("plot", "receivers")})
                # Primitive-panel InputSpec chain: rolling specs → boresight →
                # HolovizOp.input_specs.
                self.add_flow(plot, boresight, {("plot_specs", "specs_in")})
                self.add_flow(boresight, visualizer, {("specs_out", "input_specs")})


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    cpp_dir = Path(__file__).resolve().parent.parent / "cpp"
    return run_fmcw_app(
        FmcwMimoApp, argv, "fmcw_mimo", cpp_dir, "fmcw_mimo.yaml", config.load_config
    )


if __name__ == "__main__":
    raise SystemExit(main())
