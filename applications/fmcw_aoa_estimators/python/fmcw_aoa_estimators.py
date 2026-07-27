# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
    RdChannelSliceOp,
    RdMapToImageOp,
    RollingPlotOp,
    RootMUSICEstimatorOp,
    SnapshotGatherOp,
    cluster_palette,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402,F401

import config  # noqa: E402 (colocated, script-local - see config.py docstring)
from common.python.demo_config import cluster_layer_name, make_view  # noqa: E402
from common.python.driver import run_fmcw_app  # noqa: E402
from common.python.ops import SimAdcSourceOp, TensorMapDiscardOp  # noqa: E402
from custom_ops import EstimatorLabelOp, EstimatorSelector, FrameCaptureOp  # noqa: E402

# isort: on

install_matx_cache_atexit()

K_NUM_CLUSTER_COLORS = 8
K_NOISE_LAYER_NAME = "noise"
K_RANGE_PLOT_LAYER = "range_plot"
K_VELOCITY_PLOT_LAYER = "velocity_plot"
K_RANGE_TITLE_LAYER = "range_title"
K_VELOCITY_TITLE_LAYER = "velocity_title"
K_BORESIGHT_PLOT_LAYER = "boresight_plot"
K_BORESIGHT_TITLE_LAYER = "boresight_title"

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
        cluster = ClusterDBSCANOp(self, self.from_config("cluster_dbscan"), name="cluster_dbscan")
        cluster_report = ClusterReportOp(self, name="cluster_report")
        reduction = DetectionReductionOp(self, self.from_config("reduction"), name="reduction")

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

        # Each estimator feeds the same aoa_estimates input; only the selected
        # one emits per frame, so FeatureAssembly sees exactly one estimate.
        for est in (beamscan, mvdr, music, root_music, esprit):
            self.add_flow(est, feature_assembly, {("aoa_estimates", "aoa_estimates")})
        self.add_flow(beamscan, spectrum_sink, {("angle_spectrum", "input")})

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

        if cfg.visualize:
            self._compose_viz(
                cfg,
                range_op,
                doppler,
                cfar,
                cluster,
                reduction,
                (beamscan, mvdr, music, root_music, esprit),
                selector,
            )

    def _compose_viz(self, cfg, range_op, doppler, cfar, cluster, reduction, estimators, selector):
        ch_slice = RdChannelSliceOp(self, channel=0, name="rd_channel_slice")
        to_image = RdMapToImageOp(self, name="rd_map_to_image")
        overlay = ClusterOverlayOp(self, name="cluster_overlay")

        input_specs: list[HolovizOp.InputSpec] = []
        rd_spec = HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
        input_specs.append(rd_spec)

        palette = cluster_palette()
        if len(palette) != K_NUM_CLUSTER_COLORS:
            raise RuntimeError(
                f"fmcw_aoa_estimators: cluster palette size must equal {K_NUM_CLUSTER_COLORS}"
            )

        def add_crosses(name: str, color: list[float]) -> None:
            spec = HolovizOp.InputSpec(name, HolovizOp.InputType.CROSSES)
            spec.color = color
            spec.line_width = 2.0
            spec.priority = 1
            input_specs.append(spec)

        for c in range(K_NUM_CLUSTER_COLORS):
            add_crosses(cluster_layer_name(c), palette[c])
        add_crosses(K_NOISE_LAYER_NAME, [0.6, 0.6, 0.6, 1.0])

        plot: RollingPlotOp | None = None
        boresight: BoresightViewOp | None = None
        if cfg.plots.enable:
            left = make_view(0.0, 0.0, 0.60, 1.0)
            for spec in input_specs:
                spec.views = [left]

            def add_plot(img_name: str, title_name: str, title: str, oy: float, h: float) -> None:
                image = HolovizOp.InputSpec(img_name, HolovizOp.InputType.COLOR)
                image.views = [make_view(0.63, oy, 0.35, h)]
                input_specs.append(image)
                text = HolovizOp.InputSpec(title_name, HolovizOp.InputType.TEXT)
                text.text = [title]
                text.color = [1.0, 1.0, 1.0, 1.0]
                text.priority = 2
                text.views = [make_view(0.63, oy, 0.35, h)]
                input_specs.append(text)

            panel_h = 0.30
            add_plot(
                K_RANGE_PLOT_LAYER, K_RANGE_TITLE_LAYER, "Range (m) vs time (s)", 0.02, panel_h
            )
            add_plot(
                K_VELOCITY_PLOT_LAYER,
                K_VELOCITY_TITLE_LAYER,
                "Velocity (m/s) vs time (s)",
                0.35,
                panel_h,
            )
            add_plot(
                K_BORESIGHT_PLOT_LAYER,
                K_BORESIGHT_TITLE_LAYER,
                "Boresight az/el (deg)",
                0.68,
                panel_h,
            )
            plot = RollingPlotOp(
                self, window=cfg.plots.window, frame_period=cfg.frame_period, name="rolling_plot"
            )
            boresight = BoresightViewOp(
                self,
                az_lo=cfg.az_lo,
                az_hi=cfg.az_hi,
                el_lo=cfg.el_lo,
                el_hi=cfg.el_hi,
                name="boresight_view",
            )

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
            window_title="FMCW AoA estimators - RD (ch0) + clusters",
            width=1200 if cfg.plots.enable else 800,
            height=760 if cfg.plots.enable else 600,
            headless=cfg.headless,
            name="holoviz",
            **viz_kwargs,
        )
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            capture = FrameCaptureOp(self, out_dir=capture_dir, name="frame_capture")
            self.add_flow(visualizer, capture, {("render_buffer_output", "render")})

        # Live caption of the active estimator, drawn over the top of the RD map
        # (a dynamic TEXT layer appended to Holoviz's compose-time specs).
        label = EstimatorLabelOp(
            self,
            selector=selector,
            display_names=DISPLAY_NAMES,
            name="estimator_label",
        )

        self.add_flow(doppler, ch_slice, {("rd_maps", "rd_maps")})
        self.add_flow(ch_slice, to_image, {("rd_map", "rd_maps")})
        self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
        self.add_flow(doppler, label, {("rd_maps", "rd_maps")})
        self.add_flow(label, visualizer, {("input_specs", "input_specs")})
        self.add_flow(label, visualizer, {("label_pos", "receivers")})
        self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
        self.add_flow(cluster, overlay, {("labels", "labels")})
        self.add_flow(ch_slice, overlay, {("rd_map", "rd_maps")})
        self.add_flow(overlay, visualizer, {("overlay", "receivers")})

        if plot is not None:
            self.add_flow(reduction, plot, {("rep_idx", "rep_idx")})
            self.add_flow(range_op, plot, {("range_bins", "range_bins")})
            self.add_flow(doppler, plot, {("velocity_bins", "velocity_bins")})
            self.add_flow(plot, visualizer, {("plots", "receivers")})
        if boresight is not None:
            # Boresight az/el is fed by whichever estimator ran this frame.
            for est in estimators:
                self.add_flow(est, boresight, {("aoa_estimates", "aoa_estimates")})
            self.add_flow(cluster, boresight, {("labels", "labels")})
            self.add_flow(boresight, visualizer, {("plot", "receivers")})


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
