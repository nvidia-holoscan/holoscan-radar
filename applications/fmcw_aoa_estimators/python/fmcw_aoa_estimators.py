# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FMCW interchangeable-AoA-estimator pipeline (Python):
#
#   SimAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
#     -> SnapshotGather (K>1) --(dynamic)--> { MVDR | MUSIC | Root-MUSIC | ESPRIT }
#     -> SnapshotGather (K=1) --(dynamic)--> { Beamscan }
#     -> FeatureAssembly [r,v,az,el] -> ClusterDBSCAN -> DetectionReduction
#     -> ClusterReportOp  [-> viz]
#
# Divergence from the C++ app (documented in custom_ops.py): the C++ app funnels
# all five estimators through one pass-through router + a K=1 slice operator.
# Holoscan's Python emit path cannot produce an empty (0-detection) device
# tensor a C++ operator accepts, and bool is_valid is not receivable in a Python
# operator, so the Python graph keeps the snapshot/is_valid data plane entirely
# in C++: two SnapshotGatherOps (multi-look for the covariance/gridless four,
# single-look for Beamscan) fan out to the estimators, and selection is done
# purely with dynamic-flow callbacks that touch no tensors. A shared
# EstimatorSelector keeps the two callbacks in lockstep so exactly one estimator
# runs per frame. Config is loaded from fmcw_aoa_estimators.yaml (default: the
# cpp copy). AppConfig / load_config live in config.py, the selection policy in
# custom_ops.py (both colocated); operator blocks flow through from_config().

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
from holoscan.radar import install_matx_cache_atexit  # noqa: E402
from holoscan.radar import (
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
    ESPRITEstimatorOp,
    FeatureAssemblyOp,
    FMCWDopplerProcessorOp,
    FMCWRangeProcessorOp,
    MUSICEstimatorOp,
    MVDREstimatorOp,
    NullSinkOp,
    PanelRect,
    RdMapToImageOp,
    RollingPlotOp,
    RootMUSICEstimatorOp,
    SnapshotGatherOp,
    cluster_palette,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402,F401

import config  # noqa: E402 (colocated, script-local - see config.py docstring)
from common.python.demo_config import (  # noqa: E402
    K_BOX_L,
    K_BOX_R,
    K_BOX_T,
    detection_layer_specs,
    make_view,
    rd_map_box,
    rd_map_region,
    rd_map_view_aspect,
)
from common.python.driver import run_fmcw_app  # noqa: E402
from common.python.ops import SimAdcSourceOp, TensorMapDiscardOp  # noqa: E402
from custom_ops import (  # noqa: E402
    K_RD_TITLE_PX,
    EstimatorLabelOp,
    EstimatorSelector,
    FrameCaptureOp,
)

# isort: on

install_matx_cache_atexit()

K_NUM_CLUSTER_COLORS = 30
# RD-map column width when the side panels are shown (mirrors HolovizLayout).
K_LEFT_WIDTH = 0.60

# Rotation order (index 0 = beamscan, fed by the K=1 gather; indices 1..4 =
# the covariance/gridless estimators, fed by the K>1 gather).
DISPLAY_NAMES = ["beamscan", "mvdr", "music", "root_music", "esprit"]
BEAM_TARGET = "beamscan_estimator"
# Covariance/gridless estimator index -> operator name (fed by the K>1 gather).
COV_TARGETS = {
    1: "mvdr_estimator",
    2: "music_estimator",
    3: "root_music_estimator",
    4: "esprit_estimator",
}


class FmcwAoaEstimatorsApp(Application):
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
        # nci_fold is not exposed by the Python CFAR binding (unlike the C++
        # Arg); the operator derives the fold from the RX axis length itself.
        cfar = CFARDetectorOp(self, self.from_config("cfar_detector"), name="cfar_detector")
        # Multi-look gather (K>1) for the covariance/gridless four, and a
        # single-look gather (K=1) for Beamscan. Two gathers rather than a
        # Python K=1 slice: an empty snapshot slice cannot survive Holoscan's
        # Python emit path (see custom_ops.py).
        gather_cov = SnapshotGatherOp(
            self, self.from_config("snapshot_gather"), name="snapshot_gather_cov"
        )
        gather_beam = SnapshotGatherOp(self, snapshot_window=[0, 0], name="snapshot_gather_beam")

        hold_index = DISPLAY_NAMES.index(cfg.router.hold) if cfg.router.hold in DISPLAY_NAMES else 0
        selector = EstimatorSelector(
            mode=cfg.router.mode,
            rotate_period=cfg.router.rotate_period,
            hold_index=hold_index,
            control_path=cfg.router.control_path,
            log_every=cfg.router.log_every,
            display_names=DISPLAY_NAMES,
        )
        selector.num_participants = 2  # the two gather dynamic-flow callbacks

        beamscan = BeamscanEstimatorOp(
            self,
            self.from_config("beamscan_estimator"),
            array_geometry=cfg.array_geometry,
            name="beamscan_estimator",
        )
        mvdr = MVDREstimatorOp(
            self,
            self.from_config("mvdr_estimator"),
            array_geometry=cfg.array_geometry,
            name="mvdr_estimator",
        )
        music = MUSICEstimatorOp(
            self,
            self.from_config("music_estimator"),
            array_geometry=cfg.array_geometry,
            name="music_estimator",
        )
        root_music = RootMUSICEstimatorOp(
            self,
            self.from_config("root_music_estimator"),
            array_geometry=cfg.array_geometry,
            name="root_music_estimator",
        )
        esprit = ESPRITEstimatorOp(
            self,
            self.from_config("esprit_estimator"),
            array_geometry=cfg.array_geometry,
            name="esprit_estimator",
        )

        det_report = DetectionReportOp(self, name="detection_report")
        if cadence is not None:
            det_report.cadence = cadence
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")
        spectrum_sink = TensorMapDiscardOp(self, name="angle_spectrum_discard")
        # A dynamic-flow operator must activate >=1 edge every tick (Holoscan
        # dereferences its dynamic_flows() list unconditionally). Each gather
        # routes to its drain on the frames its estimator is not selected.
        cov_drain = NullSinkOp(self, name="cov_drain")
        beam_drain = NullSinkOp(self, name="beam_drain")
        # SnapshotGatherOp always emits is_valid (bool); a Python fragment cannot
        # DLPack-wrap a bool tensor, so it is drained, never delivered to an
        # estimator. The estimators treat a missing mask as all-valid.
        cov_valid_sink = NullSinkOp(self, name="cov_valid_sink")
        beam_valid_sink = NullSinkOp(self, name="beam_valid_sink")

        self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
        self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
        self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
        self.add_flow(doppler, cfar, {("rd_maps", "rd_maps")})

        # Both gathers get the same rd_maps + detection_idx every frame, so they
        # tick in lockstep. The snapshot/is_valid data plane is C++-only here
        # (bool- and empty-safe); only dynamic-flow edges are toggled per frame.
        for gather in (gather_cov, gather_beam):
            self.add_flow(doppler, gather, {("rd_maps", "rd_maps")})
            self.add_flow(cfar, gather, {("detection_idx", "detection_idx")})

        # Static fan-out (every branch + a drain declared so each transmitter
        # has a receiver); the callbacks activate exactly one edge per frame.
        for est in (mvdr, music, root_music, esprit):
            self.add_flow(gather_cov, est, {("snapshot", "snapshot")})
        self.add_flow(gather_cov, cov_drain, {("snapshot", "input")})
        self.add_flow(gather_beam, beamscan, {("snapshot", "snapshot")})
        self.add_flow(gather_beam, beam_drain, {("snapshot", "input")})
        # Drain the bool is_valid outputs. They must have a receiver or the
        # gather never ticks, but a Python fragment cannot DLPack-wrap a bool
        # tensor, so Holoscan logs a benign "Unsupported DLPack data type
        # (code: 6)" per emit here. The estimators run without the mask
        # (treated as all-valid), matching the C++ app's behavior closely enough
        # for the demo.
        self.add_flow(gather_cov, cov_valid_sink, {("is_valid", "input")})
        self.add_flow(gather_beam, beam_valid_sink, {("is_valid", "input")})

        self.add_flow(beamscan, spectrum_sink, {("angle_spectrum", "input")})
        # Every estimator branch, for the fan-ins below. Only the selected one
        # emits per frame, so each downstream aoa_estimates input sees exactly
        # one estimate per frame.
        estimators = (beamscan, mvdr, music, root_music, esprit)

        # Order-independent selection: each callback queries the shared selector
        # once per frame; the selector advances the frame only after both have,
        # so both agree on the active estimator regardless of invocation order.
        def cov_callback(op):
            target = COV_TARGETS.get(selector.select(), "cov_drain")
            op.add_dynamic_flow(op.find_all_flow_info(lambda f: f.next_operator.name == target))

        def beam_callback(op):
            target = BEAM_TARGET if selector.select() == 0 else "beam_drain"
            op.add_dynamic_flow(op.find_all_flow_info(lambda f: f.next_operator.name == target))

        self.set_dynamic_flows(gather_cov, cov_callback)
        self.set_dynamic_flows(gather_beam, beam_callback)

        self.add_flow(cfar, det_report, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, det_report, {("detection_score", "detection_score")})
        self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
        self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        # Clustering subgraph - dropped when peak_only is set. In that mode CFAR
        # already emits one detection per target, so the per-detection angle is
        # reported through DetectionReportOp's optional aoa_estimates input
        # instead of via the (removed) cluster report.
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
            cluster_report = ClusterReportOp(self, name="cluster_report")
            reduction = DetectionReductionOp(self, self.from_config("reduction"), name="reduction")

            for est in estimators:
                self.add_flow(est, feature_assembly, {("aoa_estimates", "aoa_estimates")})
            self.add_flow(cfar, feature_assembly, {("detection_idx", "detection_idx")})
            self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
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
            self.add_flow(doppler, cluster_report, {("velocity_bins", "velocity_bins")})
        else:
            for est in estimators:
                self.add_flow(est, det_report, {("aoa_estimates", "aoa_estimates")})

        if cfg.visualize:
            self._compose_viz(
                cfg,
                range_op,
                doppler,
                cfar,
                cluster,
                reduction,
                estimators,
                selector,
            )

    def _compose_viz(self, cfg, range_op, doppler, cfar, cluster, reduction, estimators, selector):
        width = 1200 if cfg.plots.enable else 800
        height = 760 if cfg.plots.enable else 600

        # The RD display is CFAR's own thresholded surface
        # (cfar_power: post-NCI integrated power here), so the picture
        # and the overlaid detections are literally the same data.
        to_image = RdMapToImageOp(self, name="rd_map_to_image")
        overlay = ClusterOverlayOp(self, name="cluster_overlay")
        overlay.view_aspect = rd_map_view_aspect(width, height, K_LEFT_WIDTH, cfg.plots.enable)
        overlay.frame_period = cfg.frame_period  # scenario clock
        overlay.rd_region = PanelRect(
            *rd_map_region(cfg.plots.enable, K_LEFT_WIDTH)
        )  # axes/frame/title region
        overlay.rd_view_px_height = float(height)
        # EstimatorLabelOp draws the title (it carries the live estimator
        # name), so suppress the overlay's own.
        overlay.rd_title = ""
        overlay.marker_radius = cfg.viz.marker_radius
        overlay.marker_outline = cfg.viz.marker_outline
        overlay.marker_segments = cfg.viz.marker_segments

        input_specs: list[HolovizOp.InputSpec] = []
        rd_spec = HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
        input_specs.append(rd_spec)

        palette = cluster_palette()
        if len(palette) != K_NUM_CLUSTER_COLORS:
            raise RuntimeError(
                f"fmcw_aoa_estimators: cluster palette size must equal {K_NUM_CLUSTER_COLORS}"
            )

        input_specs.extend(detection_layer_specs(palette))

        # Inset the RD image + discs into the framed box (leaving margins for
        # the axes/frame/title that ClusterOverlayOp draws).
        box = make_view(*rd_map_box(cfg.plots.enable, K_LEFT_WIDTH))
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
                return (v.w * float(width)) / (v.h * float(height))

            plot = RollingPlotOp(
                self, window=cfg.plots.window, frame_period=cfg.frame_period, name="rolling_plot"
            )
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

        # Diagnostic headless frame capture: AOA_CAPTURE_DIR enables Holoviz's
        # render_buffer_output and dumps composited frames to PNG (off by
        # default; requires headless so the buffer is produced).
        capture_dir = os.environ.get("AOA_CAPTURE_DIR")
        viz_kwargs = {}
        if capture_dir:
            from holoscan.resources import UnboundedAllocator

            viz_kwargs["enable_render_buffer_output"] = True
            viz_kwargs["allocator"] = UnboundedAllocator(self, name="viz_alloc")

        visualizer = HolovizOp(
            self,
            tensors=input_specs,
            color_lut=make_turbo_lut(),
            window_title="FMCW AoA estimators - RD (NCI) + clusters",
            width=width,
            height=height,
            vsync=cfg.viz.vsync,
            headless=cfg.headless,
            name="holoviz",
            **viz_kwargs,
        )
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            capture = FrameCaptureOp(self, out_dir=capture_dir, name="frame_capture")
            self.add_flow(visualizer, capture, {("render_buffer_output", "render")})

        # Live caption of the active estimator, drawn over the top of the RD map
        # (a dynamic TEXT layer appended to Holoviz's compose-time specs). It is
        # the tail of the InputSpec chain - see the wiring below.
        # Same anchor emit_rd_axes would have used: centred on the inset image
        # box, one title height above its top edge.
        _rx, _ry, _rw, _rh = rd_map_region(cfg.plots.enable, K_LEFT_WIDTH)
        _title_sz = K_RD_TITLE_PX / float(height)
        label = EstimatorLabelOp(
            self,
            selector=selector,
            display_names=DISPLAY_NAMES,
            view_px_height=float(height),
            window_aspect=float(width) / float(height),
            title_center_x=_rx + 0.5 * (K_BOX_L + K_BOX_R) * _rw,
            title_top_y=_ry + K_BOX_T * _rh - _title_sz - 0.012,
            name="estimator_label",
        )

        self.add_flow(cfar, to_image, {("cfar_power", "rd_maps")})
        self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
        # The active-estimator caption: its text position rides `receivers`; its
        # spec closes the input_specs chain assembled below.
        self.add_flow(doppler, label, {("rd_maps", "rd_maps")})
        self.add_flow(label, visualizer, {("input_specs", "input_specs")})
        self.add_flow(label, visualizer, {("label_pos", "receivers")})
        # Overlay/plot/boresight color or index by cluster when clustering runs;
        # in peak_only mode they fall back to the raw CFAR peaks.
        self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, overlay, {("cfar_power", "rd_maps")})
        # Physical axis bins for the RD-map tick labels (range X, velocity Y).
        self.add_flow(range_op, overlay, {("range_bins", "range_bins")})
        self.add_flow(doppler, overlay, {("velocity_bins", "velocity_bins")})
        if not cfg.peak_only:
            self.add_flow(cluster, overlay, {("labels", "labels")})
            # rep_idx colors each cluster by its representative RD cell (stable
            # across frames) instead of the flickery size-rank label.
            self.add_flow(reduction, overlay, {("rep_idx", "rep_idx")})
        self.add_flow(overlay, visualizer, {("overlay", "receivers")})
        # HolovizOp has one input_specs port, so every dynamic-spec producer is
        # chained into a single stream that terminates at the caption:
        #   overlay -> rolling plot -> boresight -> estimator label
        # (without panels: overlay -> estimator label).
        if plot is not None:
            self.add_flow(overlay, plot, {("time_specs", "specs_in")})
        else:
            self.add_flow(overlay, label, {("time_specs", "specs_in")})

        if plot is not None:
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
        if boresight is not None:
            # Boresight az/el is fed by whichever estimator ran this frame.
            for est in estimators:
                self.add_flow(est, boresight, {("aoa_estimates", "aoa_estimates")})
            if not cfg.peak_only:
                self.add_flow(cluster, boresight, {("labels", "labels")})
                self.add_flow(reduction, boresight, {("rep_idx", "rep_idx")})
            self.add_flow(boresight, visualizer, {("plot", "receivers")})
            self.add_flow(plot, boresight, {("plot_specs", "specs_in")})
            self.add_flow(boresight, label, {("specs_out", "specs_in")})


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    cpp_dir = Path(__file__).resolve().parent.parent / "cpp"
    return run_fmcw_app(
        FmcwAoaEstimatorsApp,
        argv,
        "fmcw_aoa_estimators",
        cpp_dir,
        "fmcw_aoa_estimators.yaml",
        config.load_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
