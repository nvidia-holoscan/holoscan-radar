# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SIMO FMCW sparse angle-then-cluster pipeline (Python):
#
#   SimoAdcSourceOp -> Range -> Clutter -> Doppler -> CFAR (NCI over RX)
#     -> SnapshotGather -> BeamscanEstimator -> FeatureAssembly [r,v,az,el]
#     -> ClusterDBSCAN -> DetectionReduction -> ClusterReportOp
#     [-> RdChannelSliceOp -> RdMapToImageOp -> HolovizOp]
#     [-> ClusterOverlayOp -> HolovizOp]
#     [-> RollingPlotOp / BoresightViewOp -> HolovizOp]
#
# Configuration is loaded from fmcw_simo.yaml (default: the copy under
# applications/fmcw_simo/cpp/). Operator blocks can use Holoscan
# ``from_config()``; this demo still parses some blocks into kwargs for
# shared helpers in ``common.python.demo_config``.

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_BUILD_DIR = os.environ.get("HOLOSCAN_RADAR_BUILD_DIR")
if _BUILD_DIR:
    import holoscan as _holoscan_pkg

    _build_holoscan = os.path.join(_BUILD_DIR, "python", "lib", "holoscan")
    if _build_holoscan not in _holoscan_pkg.__path__:
        _holoscan_pkg.__path__.insert(0, _build_holoscan)

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import cupy as cp  # noqa: E402  # after holoscan.radar (MatX CUDA cache)
from common.python.demo_config import (  # noqa: E402
    K_C,
    PlotsConfig,
    grid_extent,
    make_ura_geometry,
    parse_beamscan_estimator,
    parse_cfar_detector,
    parse_cluster_dbscan,
    parse_clutter_filter,
    parse_doppler_processor,
    parse_plots,
    parse_range_processor,
    parse_reduction,
    parse_run,
    parse_snapshot_gather,
    parse_target,
    parse_waveform,
)

from holoscan.conditions import CountCondition, PeriodicCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
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
    NullSinkOp,
    RdChannelSliceOp,
    RdMapToImageOp,
    RollingPlotOp,
    SnapshotGatherOp,
    cluster_palette,
    gui_available,
    install_matx_cache_atexit,  # noqa: E402
    make_turbo_lut,
)

install_matx_cache_atexit()

logger = logging.getLogger("fmcw_simo")

K_PI = math.pi
K_NUM_CLUSTER_COLORS = 8
K_NOISE_LAYER_NAME = "noise"
K_RANGE_PLOT_LAYER = "range_plot"
K_VELOCITY_PLOT_LAYER = "velocity_plot"
K_RANGE_TITLE_LAYER = "range_title"
K_VELOCITY_TITLE_LAYER = "velocity_title"
K_BORESIGHT_PLOT_LAYER = "boresight_plot"
K_BORESIGHT_TITLE_LAYER = "boresight_title"


@dataclass
class SimWaveform:
    carrier_frequency: float = 77e9
    bandwidth: float = 150e6
    chirp_duration: float = 20e-6
    sample_rate: float = 12.8e6
    chirp_repetition_interval: float = 25e-6

    def slope(self) -> float:
        return self.bandwidth / self.chirp_duration


@dataclass
class PointTarget:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0 + 0.0j


@dataclass
class SimScene:
    targets: list[PointTarget] = field(default_factory=list)
    waveform: SimWaveform = field(default_factory=SimWaveform)
    num_chirps: int = 64
    samples_per_chirp: int = 256
    rx_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )
    snr_db: float | None = None
    # Fixed seed, reseeded every frame — same demo contract as C++
    # point_target_sim (frozen noise realization for reproducibility).
    seed: int = 0x5EED5EED


@dataclass
class SimResult:
    # Device (CuPy) complex64 IF cube, already shaped for emit.
    cube: object
    shape: tuple[int, ...]


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def simulate(scene: SimScene) -> SimResult:
    """Simulate one SIMO CPI on GPU (CuPy); returns shaped complex64 cube."""
    num_rx = len(scene.rx_positions)
    nc = scene.num_chirps
    ns = scene.samples_per_chirp
    c = K_C
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    pri = scene.waveform.chirp_repetition_interval
    tx_pos = (0.0, 0.0, 0.0)

    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    t = cp.arange(ns, dtype=cp.float64) / fs
    k = cp.arange(nc, dtype=cp.float64)
    cube = cp.zeros((num_rx, nc, ns), dtype=cp.complex128)
    for target in scene.targets:
        rng = _norm(target.position)
        if rng <= 0.0:
            raise ValueError("point_target_sim: target at the origin")
        look = (
            target.position[0] / rng,
            target.position[1] / rng,
            target.position[2] / rng,
        )
        look_v = cp.asarray(look, dtype=cp.float64)
        v_r = _dot(target.velocity, look)
        apert_rx = rx_pos @ look_v + _dot(tx_pos, look)
        range_k = rng + v_r * k * pri
        tau = (2.0 * range_k[cp.newaxis, :] - apert_rx[:, cp.newaxis]) / c
        beat = slope * tau
        phase = (2.0 * K_PI) * (
            beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :]
            - f_c * tau[:, :, cp.newaxis]
        )
        cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        signal_power = 1.0
        noise_power = signal_power / (10.0 ** (scene.snr_db / 10.0))
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (
            rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
        )

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def advance_scene(scene: SimScene, dt: float) -> None:
    for target in scene.targets:
        target.position = (
            target.position[0] + target.velocity[0] * dt,
            target.position[1] + target.velocity[1] * dt,
            target.position[2] + target.velocity[2] * dt,
        )


@dataclass
class AppConfig:
    scene: SimScene
    array_geometry: object
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    num_frames: int = 1
    frame_period: float = 0.1
    meter_cadence: bool = True
    meter_log_every: int = 0
    visualize: bool = False
    skip_display_check: bool = False
    headless: bool = False
    az_lo: float = -0.35
    az_hi: float = 0.35
    el_lo: float = -0.1
    el_hi: float = 0.1
    range_processor: dict = field(default_factory=dict)
    clutter_filter: dict = field(default_factory=dict)
    doppler_processor: dict = field(default_factory=dict)
    cfar_detector: dict = field(default_factory=dict)
    snapshot_gather: dict = field(default_factory=dict)
    beamscan_estimator: dict = field(default_factory=dict)
    cluster_dbscan: dict = field(default_factory=dict)
    reduction: dict = field(default_factory=dict)


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    root = yaml.safe_load(path.read_text())

    waveform = parse_waveform(root["waveform"])
    wavelength = waveform.wavelength()
    scene = SimScene(
        waveform=SimWaveform(
            carrier_frequency=waveform.carrier_frequency(),
            bandwidth=waveform.bandwidth(),
            chirp_duration=waveform.chirp_duration(),
            sample_rate=waveform.sample_rate(),
            chirp_repetition_interval=waveform.chirp_repetition_interval(),
        ),
        num_chirps=int(root["cube"]["num_chirps"]),
        samples_per_chirp=int(root["cube"]["samples_per_chirp"]),
    )
    run = parse_run(root)

    rx = root.get("rx_array")
    if not rx:
        raise ValueError("rx_array section is required for SIMO")
    az_elements = int(rx["az_elements"])
    el_elements = int(rx["el_elements"])
    spacing = float(rx["spacing"])
    if az_elements < 2 or el_elements < 2:
        raise ValueError(
            "SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2"
        )
    array_geometry = make_ura_geometry(az_elements, el_elements, spacing, wavelength)
    scene.rx_positions = []
    for iz in range(el_elements):
        for iy in range(az_elements):
            scene.rx_positions.append(
                (0.0, float(iy) * spacing * wavelength, float(iz) * spacing * wavelength)
            )

    for t in root.get("targets", []):
        position, velocity, amplitude = parse_target(t, app_name="fmcw_simo")
        scene.targets.append(
            PointTarget(position=position, velocity=velocity, amplitude=amplitude)
        )

    noise = root.get("noise_snr_db")
    if noise is not None:
        scene.snr_db = float(noise)

    plots = parse_plots(root)
    be = root.get("beamscan_estimator", {})
    az_lo, az_hi = grid_extent(
        [float(x) for x in be.get("az_angles", [])] or None, -0.35, 0.35
    )
    el_lo, el_hi = grid_extent(
        [float(x) for x in be.get("el_angles", [])] or None, -0.1, 0.1
    )

    return AppConfig(
        scene=scene,
        array_geometry=array_geometry,
        plots=plots,
        num_frames=run.num_frames,
        frame_period=run.frame_period,
        meter_cadence=run.meter_cadence,
        meter_log_every=run.meter_log_every,
        visualize=bool(root.get("visualize", False)),
        skip_display_check=bool(root.get("skip_display_check", False)),
        headless=bool(root.get("headless", False)),
        az_lo=az_lo,
        az_hi=az_hi,
        el_lo=el_lo,
        el_hi=el_hi,
        range_processor=parse_range_processor(root.get("range_processor", {})),
        clutter_filter=parse_clutter_filter(root.get("clutter_filter", {})),
        doppler_processor=parse_doppler_processor(root.get("doppler_processor", {})),
        cfar_detector=parse_cfar_detector(root.get("cfar_detector", {})),
        snapshot_gather=parse_snapshot_gather(root.get("snapshot_gather", {})),
        beamscan_estimator=parse_beamscan_estimator(
            root.get("beamscan_estimator", {})
        ),
        cluster_dbscan=parse_cluster_dbscan(root.get("cluster_dbscan", {})),
        reduction=parse_reduction(root.get("reduction", {})),
    )


def cluster_layer_name(color: int) -> str:
    return f"cluster_{color}"


def make_view(ox: float, oy: float, w: float, h: float) -> HolovizOp.InputSpec.View:
    view = HolovizOp.InputSpec.View()
    view.offset_x = ox
    view.offset_y = oy
    view.width = w
    view.height = h
    return view


class SimoAdcSourceOp(Operator):
    """Emit one synthetic SIMO complex-IF ADC cube per tick."""

    def __init__(
        self,
        fragment,
        *args,
        scene: SimScene,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        result = simulate(self.scene)
        if len(result.shape) != 3:
            raise RuntimeError(
                "SimoAdcSourceOp: expected rank-3 (N_rx>=2, chirps, samples)"
            )
        if result.shape[0] < 2:
            raise RuntimeError("SimoAdcSourceOp: SIMO requires N_rx >= 2")
        op_output.emit({"adc_cube": result.cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


class TensorMapDiscardOp(Operator):
    """Drain an unused TensorMap port."""

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("input")


class FmcwSimoApp(Application):
    def __init__(self, cfg: AppConfig):
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

        source = SimoAdcSourceOp(
            self,
            CountCondition(self, count=cfg.num_frames),
            PeriodicCondition(self, recess_period=period_ns),
            scene=cfg.scene,
            frame_period=cfg.frame_period,
            cadence=cadence,
            name="adc_source",
        )

        range_op = FMCWRangeProcessorOp(self, **cfg.range_processor)
        clutter = ClutterFilterOp(self, **cfg.clutter_filter)
        doppler = FMCWDopplerProcessorOp(self, **cfg.doppler_processor)
        cfar = CFARDetectorOp(self, **cfg.cfar_detector)
        gather = SnapshotGatherOp(self, **cfg.snapshot_gather)
        beamscan = BeamscanEstimatorOp(
            self,
            array_geometry=cfg.array_geometry,
            **cfg.beamscan_estimator,
        )

        det_report = DetectionReportOp(self, name="detection_report")
        if cadence is not None:
            det_report.cadence = cadence
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")
        spectrum_sink = TensorMapDiscardOp(self, name="angle_spectrum_discard")

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
        cluster = ClusterDBSCANOp(self, **cfg.cluster_dbscan)
        cluster_report = ClusterReportOp(self, name="cluster_report")
        reduction = DetectionReductionOp(self, **cfg.reduction)

        self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
        self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
        self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
        self.add_flow(doppler, cfar, {("rd_maps", "rd_maps")})

        self.add_flow(doppler, gather, {("rd_maps", "rd_maps")})
        self.add_flow(cfar, gather, {("detection_idx", "detection_idx")})
        self.add_flow(gather, beamscan, {("snapshot", "snapshot"), ("is_valid", "is_valid")})
        self.add_flow(beamscan, feature_assembly, {("aoa_estimates", "aoa_estimates")})
        self.add_flow(beamscan, spectrum_sink, {("angle_spectrum", "input")})

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
            ch_slice = RdChannelSliceOp(self, channel=0, name="rd_channel_slice")
            to_image = RdMapToImageOp(self, name="rd_map_to_image")
            overlay = ClusterOverlayOp(self, name="cluster_overlay")

            input_specs: list[HolovizOp.InputSpec] = []
            rd_spec = HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
            input_specs.append(rd_spec)

            palette = cluster_palette()
            if len(palette) != K_NUM_CLUSTER_COLORS:
                raise RuntimeError(
                    f"fmcw_simo: cluster palette size must equal {K_NUM_CLUSTER_COLORS}"
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

                def add_plot(
                    img_name: str, title_name: str, title: str, oy: float, h: float
                ) -> None:
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
                    K_RANGE_PLOT_LAYER,
                    K_RANGE_TITLE_LAYER,
                    "Range (m) vs time (s)",
                    0.02,
                    panel_h,
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
                    self,
                    window=cfg.plots.window,
                    frame_period=cfg.frame_period,
                    name="rolling_plot",
                )
                boresight = BoresightViewOp(
                    self,
                    az_lo=cfg.az_lo,
                    az_hi=cfg.az_hi,
                    el_lo=cfg.el_lo,
                    el_hi=cfg.el_hi,
                    name="boresight_view",
                )

            visualizer = HolovizOp(
                self,
                tensors=input_specs,
                color_lut=make_turbo_lut(),
                window_title="FMCW SIMO - RD (ch0) + clusters",
                width=1200 if cfg.plots.enable else 800,
                height=760 if cfg.plots.enable else 600,
                headless=cfg.headless,
                name="holoviz",
            )

            self.add_flow(doppler, ch_slice, {("rd_maps", "rd_maps")})
            self.add_flow(ch_slice, to_image, {("rd_map", "rd_maps")})
            self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
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
                self.add_flow(beamscan, boresight, {("aoa_estimates", "aoa_estimates")})
                self.add_flow(cluster, boresight, {("labels", "labels")})
                self.add_flow(boresight, visualizer, {("plot", "receivers")})


def resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    return Path(__file__).resolve().parent.parent / "cpp" / "fmcw_simo.yaml"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    argv = argv if argv is not None else sys.argv
    try:
        cfg = load_config(resolve_config_path(argv))
        if (
            cfg.visualize
            and not cfg.headless
            and not cfg.skip_display_check
            and not gui_available()
        ):
            logger.warning(
                "fmcw_simo: visualize enabled but no display; running without GUI"
            )
            cfg.visualize = False

        app = FmcwSimoApp(cfg)
        app.run()
        if app.cadence is not None:
            app.cadence.log_summary()
    except Exception as exc:
        logger.error("fmcw_simo: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
