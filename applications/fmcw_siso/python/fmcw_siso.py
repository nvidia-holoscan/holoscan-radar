# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SISO FMCW pipeline example (Python): synthetic ADC source feeding
#
#   AdcSourceOp -> FMCWRangeProcessorOp -> ClutterFilterOp
#               -> FMCWDopplerProcessorOp -> CFARDetectorOp
#                                        -> DetectionReportOp
#                                        -> FeatureAssemblyOp
#                                             -> ClusterDBSCANOp
#                                                  -> DetectionReductionOp
#                                                       -> ClusterReportOp
#                                         [-> RdMapToImageOp   -> HolovizOp]
#                                         [-> ClusterOverlayOp -> HolovizOp]
#                                         [-> RollingPlotOp    -> HolovizOp]
#
# Configuration is loaded from fmcw_siso.yaml (default: the copy under
# applications/fmcw_siso/cpp/). App-level fields (cube, targets, run, plots)
# are parsed in Python; per-operator blocks flow through Holoscan
# ``from_config()`` like the C++ app.

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Extend holoscan.__path__ with the build-tree package when HOLOSCAN_RADAR_BUILD_DIR
# is set (mirrors tests/python/conftest.py).
_BUILD_DIR = os.environ.get("HOLOSCAN_RADAR_BUILD_DIR")
if _BUILD_DIR:
    import holoscan as _holoscan_pkg

    _build_holoscan = os.path.join(_BUILD_DIR, "python", "lib", "holoscan")
    if _build_holoscan not in _holoscan_pkg.__path__:
        _holoscan_pkg.__path__.insert(0, _build_holoscan)

import cupy as cp  # noqa: E402  # after holoscan.radar (MatX CUDA cache)

from holoscan.conditions import CountCondition, PeriodicCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.operators import HolovizOp  # noqa: E402
from holoscan.radar import (  # noqa: E402
    AssemblyFeature,
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
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    NullSinkOp,
    RdMapToImageOp,
    RollingPlotOp,
    cluster_palette,
    gui_available,
    install_matx_cache_atexit,
    make_turbo_lut,
)

install_matx_cache_atexit()

logger = logging.getLogger("fmcw_siso")

K_C = 299792458.0
K_PI = math.pi
K_NUM_CLUSTER_COLORS = 8
K_NOISE_LAYER_NAME = "noise"
K_RANGE_PLOT_LAYER = "range_plot"
K_VELOCITY_PLOT_LAYER = "velocity_plot"
K_RANGE_TITLE_LAYER = "range_title"
K_VELOCITY_TITLE_LAYER = "velocity_title"


# ---------------------------------------------------------------------------
# Minimal analytic point-target simulator (SISO path from point_target_sim.hpp)
# ---------------------------------------------------------------------------


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
    """Simulate one CPI on GPU (CuPy); returns shaped complex64 cube."""
    nc = scene.num_chirps
    ns = scene.samples_per_chirp
    c = K_C
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    pri = scene.waveform.chirp_repetition_interval

    t = cp.arange(ns, dtype=cp.float64) / fs
    k = cp.arange(nc, dtype=cp.float64)
    cube = cp.zeros((nc, ns), dtype=cp.complex128)
    for target in scene.targets:
        rng = _norm(target.position)
        if rng <= 0.0:
            raise ValueError("point_target_sim: target at the origin")
        look = (
            target.position[0] / rng,
            target.position[1] / rng,
            target.position[2] / rng,
        )
        v_r = _dot(target.velocity, look)
        range_k = rng + v_r * k * pri
        tau = 2.0 * range_k / c
        beat = slope * tau
        phase = (2.0 * K_PI) * (
            beat[:, cp.newaxis] * t[cp.newaxis, :] - f_c * tau[:, cp.newaxis]
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


# ---------------------------------------------------------------------------
# App-level config (simulator / run / plots); operators use from_config.
# ---------------------------------------------------------------------------


@dataclass
class PlotsConfig:
    enable: bool = True
    window: int = 120


@dataclass
class AppConfig:
    config_path: Path
    scene: SimScene
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    num_frames: int = 1
    frame_period: float = 0.1
    meter_cadence: bool = True
    meter_log_every: int = 0
    visualize: bool = False
    skip_display_check: bool = False
    headless: bool = False


def parse_waveform(node: Any) -> FMCWWaveformDescriptor:
    if not isinstance(node, dict):
        raise ValueError("waveform must be a mapping")
    spec = FMCWWaveformSpec()
    if "carrier_frequency" in node:
        spec.carrier_frequency = float(node["carrier_frequency"])
    if "bandwidth" in node:
        spec.bandwidth = float(node["bandwidth"])
    if "chirp_slope" in node:
        spec.chirp_slope = float(node["chirp_slope"])
    if "chirp_duration" in node:
        spec.chirp_duration = float(node["chirp_duration"])
    if "sample_rate" in node:
        spec.sample_rate = float(node["sample_rate"])
    if "chirp_repetition_interval" in node:
        spec.chirp_repetition_interval = float(node["chirp_repetition_interval"])
    return FMCWWaveformDescriptor(spec)


def _require_positive(key: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{key} must be > 0, got {value}")


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    root = yaml.safe_load(path.read_text())

    waveform = parse_waveform(root["waveform"])
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

    num_frames = int(root["run"]["num_frames"])
    frame_period = float(root["run"]["frame_period"])
    meter_cadence = bool(root["run"].get("meter_cadence", True))
    meter_log_every = int(root["run"].get("meter_log_every", 0))
    _require_positive("run.num_frames", num_frames)
    _require_positive("run.frame_period", frame_period)
    _require_positive("cube.num_chirps", scene.num_chirps)
    _require_positive("cube.samples_per_chirp", scene.samples_per_chirp)
    if meter_log_every < 0:
        raise ValueError("run.meter_log_every must be >= 0")

    if not math.isfinite(frame_period):
        raise ValueError("run.frame_period must be finite")
    frame_period_ns = frame_period * 1e9
    if frame_period_ns < 1.0 or frame_period_ns > float(2**63 - 1):
        raise ValueError(
            f"run.frame_period must be within [1 ns, INT64_MAX ns], got "
            f"{frame_period} s"
        )

    for t in root.get("targets", []):
        scene.targets.append(
            PointTarget(
                position=(float(t["range"]), 0.0, 0.0),
                velocity=(float(t["velocity"]), 0.0, 0.0),
                amplitude=complex(float(t["amplitude"]), 0.0),
            )
        )

    noise = root.get("noise_snr_db")
    if noise is not None:
        scene.snr_db = float(noise)

    plots = PlotsConfig()
    plots_node = root.get("plots")
    if isinstance(plots_node, dict):
        if "enable" in plots_node:
            plots.enable = bool(plots_node["enable"])
        if "window" in plots_node:
            plots.window = int(plots_node["window"])
    if plots.window <= 1:
        raise ValueError(f"plots.window must be > 1, got {plots.window}")

    return AppConfig(
        config_path=path.resolve(),
        scene=scene,
        plots=plots,
        num_frames=num_frames,
        frame_period=frame_period,
        meter_cadence=meter_cadence,
        meter_log_every=meter_log_every,
        visualize=bool(root.get("visualize", False)),
        skip_display_check=bool(root.get("skip_display_check", False)),
        headless=bool(root.get("headless", False)),
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


# ---------------------------------------------------------------------------
# Synthetic ADC source
# ---------------------------------------------------------------------------


class AdcSourceOp(Operator):
    """Emit one synthetic complex-IF ADC cube per tick."""

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
        if len(result.shape) != 2:
            raise RuntimeError(
                "AdcSourceOp: SISO source expects a rank-2 cube, got rank "
                f"{len(result.shape)}"
            )
        op_output.emit({"adc_cube": result.cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


# ---------------------------------------------------------------------------
# Application graph
# ---------------------------------------------------------------------------


class FmcwSisoApp(Application):
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

        source = AdcSourceOp(
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
        clutter = ClutterFilterOp(
            self, self.from_config("clutter_filter"), name="clutter_filter"
        )
        doppler = FMCWDopplerProcessorOp(
            self, self.from_config("doppler_processor"), name="doppler_processor"
        )
        cfar = CFARDetectorOp(
            self, self.from_config("cfar_detector"), name="cfar_detector"
        )
        det_report = DetectionReportOp(self, name="detection_report")
        if cadence is not None:
            det_report.cadence = cadence
        mask_sink = NullSinkOp(self, name="detection_mask_sink")
        noise_sink = NullSinkOp(self, name="noise_estimate_sink")

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

        self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
        self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
        self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
        self.add_flow(doppler, cfar, {("rd_maps", "rd_maps")})
        self.add_flow(cfar, det_report, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, det_report, {("detection_score", "detection_score")})
        self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
        self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        self.add_flow(cfar, feature_assembly, {("detection_idx", "detection_idx")})
        self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
        self.add_flow(
            doppler, feature_assembly, {("velocity_bins", "velocity_bins")}
        )
        self.add_flow(feature_assembly, cluster, {("points", "points")})
        self.add_flow(feature_assembly, cluster_report, {("points", "points")})
        self.add_flow(cluster, cluster_report, {("labels", "labels")})
        self.add_flow(cluster, cluster_report, {("sizes", "sizes")})

        self.add_flow(cluster, reduction, {("labels", "labels")})
        self.add_flow(cfar, reduction, {("detection_idx", "detection_idx")})
        self.add_flow(feature_assembly, reduction, {("points", "points")})
        self.add_flow(reduction, cluster_report, {("rep_idx", "rep_idx")})
        self.add_flow(range_op, cluster_report, {("range_bins", "range_bins")})
        self.add_flow(
            doppler, cluster_report, {("velocity_bins", "velocity_bins")}
        )

        if cfg.visualize:
            to_image = RdMapToImageOp(self, name="rd_map_to_image")
            overlay = ClusterOverlayOp(self, name="cluster_overlay")

            input_specs: list[HolovizOp.InputSpec] = []
            rd_spec = HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
            input_specs.append(rd_spec)

            palette = cluster_palette()
            if len(palette) != K_NUM_CLUSTER_COLORS:
                raise RuntimeError(
                    "fmcw_siso: cluster palette size must equal "
                    f"{K_NUM_CLUSTER_COLORS}"
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
            if cfg.plots.enable:
                left = make_view(0.0, 0.0, 0.60, 1.0)
                for spec in input_specs:
                    spec.views = [left]

                def add_plot(
                    img_name: str, title_name: str, title: str, oy: float
                ) -> None:
                    image = HolovizOp.InputSpec(img_name, HolovizOp.InputType.COLOR)
                    image.views = [make_view(0.63, oy, 0.35, 0.44)]
                    input_specs.append(image)
                    text = HolovizOp.InputSpec(
                        title_name, HolovizOp.InputType.TEXT
                    )
                    text.text = [title]
                    text.color = [1.0, 1.0, 1.0, 1.0]
                    text.priority = 2
                    text.views = [make_view(0.63, oy, 0.35, 0.44)]
                    input_specs.append(text)

                add_plot(
                    K_RANGE_PLOT_LAYER,
                    K_RANGE_TITLE_LAYER,
                    "Range (m) vs time (s)",
                    0.04,
                )
                add_plot(
                    K_VELOCITY_PLOT_LAYER,
                    K_VELOCITY_TITLE_LAYER,
                    "Velocity (m/s) vs time (s)",
                    0.52,
                )

                plot = RollingPlotOp(
                    self,
                    window=cfg.plots.window,
                    frame_period=cfg.frame_period,
                    name="rolling_plot",
                )

            visualizer = HolovizOp(
                self,
                tensors=input_specs,
                color_lut=make_turbo_lut(),
                window_title="FMCW SISO - Range-Doppler + clusters",
                width=1200 if cfg.plots.enable else 800,
                height=600,
                headless=cfg.headless,
                name="holoviz",
            )

            self.add_flow(doppler, to_image, {("rd_maps", "rd_maps")})
            self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
            self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
            self.add_flow(cluster, overlay, {("labels", "labels")})
            self.add_flow(doppler, overlay, {("rd_maps", "rd_maps")})
            self.add_flow(overlay, visualizer, {("overlay", "receivers")})

            if plot is not None:
                self.add_flow(reduction, plot, {("rep_idx", "rep_idx")})
                self.add_flow(range_op, plot, {("range_bins", "range_bins")})
                self.add_flow(doppler, plot, {("velocity_bins", "velocity_bins")})
                self.add_flow(plot, visualizer, {("plots", "receivers")})


def resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    return Path(__file__).resolve().parent.parent / "cpp" / "fmcw_siso.yaml"


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
                "fmcw_siso: visualize is enabled but no usable display was "
                "detected (Holoviz could not initialize a window); running "
                "without the RD-map window"
            )
            cfg.visualize = False

        app = FmcwSisoApp(cfg)
        app.run()
        if app.cadence is not None:
            app.cadence.log_summary()
    except Exception as exc:
        logger.error("fmcw_siso: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
