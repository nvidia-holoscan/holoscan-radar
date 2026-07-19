# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MIMO FMCW sparse angle-then-cluster pipeline (Python, TDM or DDM via YAML):
#
# TDM: MimoTdmAdcSourceOp -> Deinterleave -> Range -> Clutter -> Doppler
#        -> Compensation -> VirtualArrayAssemblyOp -> ...
# DDM: MimoDdmAdcSourceOp -> Range -> Doppler -> Compensation -> Clutter
#        -> VirtualArrayAssemblyOp -> ...
#
# Configuration is loaded from fmcw_mimo.yaml (default) or fmcw_mimo_ddm.yaml.
# Prefer Holoscan ``from_config()`` for operator blocks; some app-level fields
# are still parsed via ``common.python.demo_config``.

from __future__ import annotations

import cmath
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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

from common.python.demo_config import (  # noqa: E402
    K_C,
    PlotsConfig,
    flatten_positions,
    grid_extent,
    make_virtual_geometry,
    parse_beamscan_estimator,
    parse_cfar_detector,
    parse_clutter_filter,
    parse_cluster_dbscan,
    parse_ddm_code,
    parse_doppler_processor,
    parse_mimo_coherent_compensation,
    parse_mimo_deinterleave,
    parse_plots,
    parse_range_processor,
    parse_reduction,
    parse_run,
    parse_snapshot_gather,
    parse_target,
    parse_tx_schedule,
    parse_virtual_array_assembly,
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
    ClutterFilterOp,
    ClusterDBSCANOp,
    ClusterOverlayOp,
    ClusterReportOp,
    DetectionReductionOp,
    DetectionReportOp,
    FeatureAssemblyOp,
    FMCWDopplerProcessorOp,
    FMCWRangeProcessorOp,
    MIMOCoherentCompensationOp,
    MIMODeinterleaveOp,
    MultiplexingScheme,
    NullSinkOp,
    RdChannelSliceOp,
    RdMapToImageOp,
    RollingPlotOp,
    SnapshotGatherOp,
    TXSchedule,
    VirtualArrayAssemblyOp,
    cluster_palette,
    gui_available,
    make_turbo_lut,
)

import cupy as cp  # noqa: E402  # after holoscan.radar (MatX CUDA cache)
from holoscan.radar import install_matx_cache_atexit  # noqa: E402

install_matx_cache_atexit()

logger = logging.getLogger("fmcw_mimo")

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

    def wavelength(self) -> float:
        return K_C / self.carrier_frequency


@dataclass
class PointTarget:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0 + 0.0j


@dataclass
class TargetTruth:
    range: float
    radial_velocity: float
    look_direction: tuple[float, float, float]


@dataclass
class MimoScene:
    targets: list[PointTarget] = field(default_factory=list)
    waveform: SimWaveform = field(default_factory=SimWaveform)
    num_tx: int = 2
    num_chirps: int = 64
    samples_per_chirp: int = 256
    tx_positions: list[tuple[float, float, float]] = field(default_factory=list)
    rx_positions: list[tuple[float, float, float]] = field(default_factory=list)
    propagation_speed: float = K_C
    snr_db: float | None = None
    # Fixed seed, reseeded every frame — same demo contract as C++
    # (frozen noise realization for reproducibility).
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


def derive_truth(target: PointTarget) -> TargetTruth:
    rng = _norm(target.position)
    if rng <= 0.0:
        raise ValueError("point_target_sim: target at the origin")
    look = (
        target.position[0] / rng,
        target.position[1] / rng,
        target.position[2] / rng,
    )
    return TargetTruth(
        range=rng,
        radial_velocity=_dot(target.velocity, look),
        look_direction=look,
    )


def expand_chirp_tx_index(schedule: TXSchedule, slow_time: int) -> list[int]:
    schedule.validate()
    if slow_time <= 0 or slow_time % schedule.num_tx != 0:
        raise ValueError("TXSchedule: slow_time must be a positive multiple of num_tx")
    if schedule.chirp_tx_index:
        if len(schedule.chirp_tx_index) != slow_time:
            raise ValueError("TXSchedule: chirp_tx_index length must equal slow_time")
        pattern = list(schedule.chirp_tx_index)
    else:
        cycle = (
            list(range(schedule.num_tx))
            if not schedule.tx_order
            else list(schedule.tx_order)
        )
        pattern = [cycle[k % schedule.num_tx] for k in range(slow_time)]
    for tx in pattern:
        if tx < 0 or tx >= schedule.num_tx:
            raise ValueError("TXSchedule: chirp TX index out of range")
    return pattern


def uniform_ddm_offsets(num_tx: int) -> list[float]:
    return [float(m) / float(num_tx) for m in range(num_tx)]


def fill_sin_phase_codes(code, num_chirps: int) -> None:
    """High-crosstalk NL codes (unit-test / perf recipe). Not for clean demos."""
    if code.num_tx < 1 or num_chirps < 1:
        raise ValueError("fill_sin_phase_codes: invalid args")
    code.doppler_offsets = []
    code.guard_bands = 0
    codes: list[complex] = []
    for m in range(code.num_tx):
        for k in range(num_chirps):
            phase = math.sin(0.7 * float(k) + float(m)) * 2.0
            codes.append(complex(math.cos(phase), math.sin(phase)))
    code.phase_codes = codes


def fill_zadoff_chu_phase_codes(code, num_chirps: int) -> None:
    """Low-crosstalk non-linear Zadoff-Chu roots (coprime to N).

    Quadratic phase fails the linear-phase classifier so the correlate path
    is used; distinct roots keep periodic cross-correlation low.
    """
    if code.num_tx < 1 or num_chirps < 1:
        raise ValueError("fill_zadoff_chu_phase_codes: invalid args")
    roots: list[int] = []
    for u in range(1, num_chirps):
        if math.gcd(u, num_chirps) == 1:
            roots.append(u)
            if len(roots) >= code.num_tx:
                break
    if len(roots) < code.num_tx:
        raise ValueError(
            "fill_zadoff_chu_phase_codes: need num_tx distinct roots coprime "
            "to num_chirps (try a longer cube.num_chirps)"
        )
    code.doppler_offsets = []
    code.guard_bands = 0
    odd_n = (num_chirps % 2) != 0
    codes: list[complex] = []
    for m in range(code.num_tx):
        u = float(roots[m])
        for k in range(num_chirps):
            kd = float(k)
            if odd_n:
                arg = -math.pi * u * kd * (kd + 1.0) / float(num_chirps)
            else:
                arg = -math.pi * u * kd * kd / float(num_chirps)
            codes.append(complex(math.cos(arg), math.sin(arg)))
    code.phase_codes = codes


def ddm_codes_are_linear(code) -> bool:
    """Mirror C++ ddm_classify_codes (1e-3 rad constant-step test)."""
    code.validate()
    if code.doppler_offsets:
        return True
    if not code.phase_codes:
        return True
    num_tx = int(code.num_tx)
    num_chirps = len(code.phase_codes) // num_tx
    if num_chirps < 2:
        return False
    two_pi = 2.0 * math.pi
    for m in range(num_tx):
        row = code.phase_codes[m * num_chirps : (m + 1) * num_chirps]
        delta = cmath.phase(row[1] * row[0].conjugate())
        for k in range(num_chirps - 1):
            step = cmath.phase(row[k + 1] * row[k].conjugate())
            d = step - delta
            d -= two_pi * round(d / two_pi)
            if abs(d) > 1e-3:
                return False
    return True


def resolve_ddm_offsets(code) -> list[float]:
    """Resolve linear-phase DDM offsets (raises if codes are non-linear)."""
    code.validate()
    if code.doppler_offsets:
        return [o - math.floor(o) for o in code.doppler_offsets]
    if not code.phase_codes:
        return uniform_ddm_offsets(code.num_tx)
    if not ddm_codes_are_linear(code):
        raise ValueError(
            "resolve_ddm_offsets: phase_codes are non-linear; use correlate path"
        )
    num_tx = int(code.num_tx)
    num_chirps = len(code.phase_codes) // num_tx
    offsets: list[float] = []
    for m in range(num_tx):
        row = code.phase_codes[m * num_chirps : (m + 1) * num_chirps]
        delta = cmath.phase(row[1] * row[0].conjugate())
        o = delta / (2.0 * math.pi)
        o -= math.floor(o)
        offsets.append(o)
    return offsets


def simulate_mimo_tdm_interleaved(
    scene: MimoScene, schedule: TXSchedule
) -> SimResult:
    """Analytic TDM-interleaved IF cube on GPU (CuPy)."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_tdm_interleaved: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    slow_time = num_tx * scene.num_chirps
    samples = scene.samples_per_chirp
    physical_pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    pattern = cp.asarray(expand_chirp_tx_index(schedule, slow_time), dtype=cp.int64)
    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError(
            "simulate_mimo_tdm_interleaved: tx/rx positions must be (N, 3)"
        )

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(slow_time, dtype=cp.float64)
    cube = cp.zeros((num_rx, slow_time, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * physical_pri
        apert_tx_k = (tx_pos @ look)[pattern]
        apert_rx = rx_pos @ look
        tau = (
            2.0 * range_k[cp.newaxis, :]
            - apert_tx_k[cp.newaxis, :]
            - apert_rx[:, cp.newaxis]
        ) / c
        beat = slope * tau
        phase = (2.0 * K_PI) * (
            beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :]
            - f_c * tau[:, :, cp.newaxis]
        )
        cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (
            rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
        )

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def simulate_mimo_ddm_coded(
    scene: MimoScene, offsets: list[float]
) -> SimResult:
    """Analytic DDM-coded IF cube on GPU (CuPy)."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_ddm_coded: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    num_chirps = scene.num_chirps
    samples = scene.samples_per_chirp
    pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    if len(offsets) != num_tx:
        raise ValueError("simulate_mimo_ddm_coded: offsets length must equal num_tx")

    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError("simulate_mimo_ddm_coded: tx/rx positions must be (N, 3)")
    offsets_arr = cp.asarray(offsets, dtype=cp.float64)

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(num_chirps, dtype=cp.float64)
    cube = cp.zeros((num_rx, num_chirps, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * pri
        apert_tx = tx_pos @ look
        apert_rx = rx_pos @ look
        for tx in range(num_tx):
            tau = (
                2.0 * range_k[cp.newaxis, :]
                - apert_tx[tx]
                - apert_rx[:, cp.newaxis]
            ) / c
            beat = slope * tau
            code_phase = (2.0 * K_PI) * offsets_arr[tx] * k
            phase = (2.0 * K_PI) * (
                beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :]
                - f_c * tau[:, :, cp.newaxis]
            ) + code_phase[cp.newaxis, :, cp.newaxis]
            cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (
            rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
        )

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def simulate_mimo_ddm_phase_coded(
    scene: MimoScene, phase_codes: list[complex]
) -> SimResult:
    """Analytic DDM phase-coded IF cube on GPU (CuPy) — non-linear codes."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_ddm_phase_coded: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    num_chirps = scene.num_chirps
    samples = scene.samples_per_chirp
    pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    if len(phase_codes) != num_tx * num_chirps:
        raise ValueError(
            "simulate_mimo_ddm_phase_coded: phase_codes length must be "
            "num_tx * num_chirps"
        )

    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError(
            "simulate_mimo_ddm_phase_coded: tx/rx positions must be (N, 3)"
        )
    codes = cp.asarray(phase_codes, dtype=cp.complex128).reshape(num_tx, num_chirps)

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(num_chirps, dtype=cp.float64)
    cube = cp.zeros((num_rx, num_chirps, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * pri
        apert_tx = tx_pos @ look
        apert_rx = rx_pos @ look
        for tx in range(num_tx):
            tau = (
                2.0 * range_k[cp.newaxis, :]
                - apert_tx[tx]
                - apert_rx[:, cp.newaxis]
            ) / c
            beat = slope * tau
            phase = (2.0 * K_PI) * (
                beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :]
                - f_c * tau[:, :, cp.newaxis]
            )
            cube += (
                target.amplitude
                * codes[tx][cp.newaxis, :, cp.newaxis]
                * cp.exp(1j * phase)
            )

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (
            rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)
        )

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def advance_scene(scene: MimoScene, dt: float) -> None:
    for target in scene.targets:
        target.position = (
            target.position[0] + target.velocity[0] * dt,
            target.position[1] + target.velocity[1] * dt,
            target.position[2] + target.velocity[2] * dt,
        )


@dataclass
class AppConfig:
    scheme: MultiplexingScheme = MultiplexingScheme.TDM
    scene: MimoScene = field(default_factory=MimoScene)
    tx_schedule: TXSchedule = field(default_factory=TXSchedule)
    ddm_code: object = field(default_factory=object)
    ddm_doppler_offsets: list[float] = field(default_factory=list)
    ddm_correlate: bool = False
    array_geometry: object = field(default_factory=object)
    tx_positions_flat: list[float] = field(default_factory=list)
    rx_positions_flat: list[float] = field(default_factory=list)
    wavelength: float = 0.0
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
    report_waveform: object = field(default_factory=object)
    report_num_doppler: int = 0
    range_processor: dict = field(default_factory=dict)
    clutter_filter: dict = field(default_factory=dict)
    doppler_processor: dict = field(default_factory=dict)
    cfar_detector: dict = field(default_factory=dict)
    snapshot_gather: dict = field(default_factory=dict)
    beamscan_estimator: dict = field(default_factory=dict)
    cluster_dbscan: dict = field(default_factory=dict)
    reduction: dict = field(default_factory=dict)
    mimo_deinterleave: dict = field(default_factory=dict)
    mimo_coherent_compensation: dict = field(default_factory=dict)
    virtual_array_assembly: dict = field(default_factory=dict)


def _scale_position(iy: float, iz: float, spacing_wl: float, wavelength: float):
    return (0.0, iy * spacing_wl * wavelength, iz * spacing_wl * wavelength)


def load_scene_geometry(cfg: AppConfig, root: dict, wavelength: float) -> None:
    tx = root["tx_array"]
    tx_spacing = float(tx["spacing"])
    cfg.scene.tx_positions = [
        (0.0, float(itx) * tx_spacing * wavelength, 0.0)
        for itx in range(cfg.scene.num_tx)
    ]

    rx = root.get("rx_array")
    if not rx:
        raise ValueError("rx_array section is required for MIMO")
    az_elements = int(rx["az_elements"])
    el_elements = int(rx["el_elements"])
    rx_spacing = float(rx["spacing"])
    if az_elements < 2 or el_elements < 2:
        raise ValueError(
            "MIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2"
        )
    cfg.scene.rx_positions = []
    for iz in range(el_elements):
        for iy in range(az_elements):
            cfg.scene.rx_positions.append(
                _scale_position(float(iy), float(iz), rx_spacing, wavelength)
            )

    cfg.array_geometry = make_virtual_geometry(
        cfg.scene.tx_positions, cfg.scene.rx_positions, wavelength
    )


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    root = yaml.safe_load(path.read_text())
    cfg = AppConfig()

    waveform = parse_waveform(root["waveform"])
    cfg.report_waveform = waveform
    cfg.scene.waveform = SimWaveform(
        carrier_frequency=waveform.carrier_frequency(),
        bandwidth=waveform.bandwidth(),
        chirp_duration=waveform.chirp_duration(),
        sample_rate=waveform.sample_rate(),
        chirp_repetition_interval=waveform.chirp_repetition_interval(),
    )

    mcc = root.get("mimo_coherent_compensation", {})
    if not mcc or "multiplexing_scheme" not in mcc:
        raise ValueError(
            "fmcw_mimo: mimo_coherent_compensation.multiplexing_scheme is required"
        )
    cfg.scheme = parse_mimo_coherent_compensation(mcc)["multiplexing_scheme"]
    if cfg.scheme not in (MultiplexingScheme.TDM, MultiplexingScheme.DDM):
        raise ValueError("fmcw_mimo: multiplexing_scheme must be tdm or ddm")

    run = parse_run(root)
    cfg.num_frames = run.num_frames
    cfg.frame_period = run.frame_period
    cfg.meter_cadence = run.meter_cadence
    cfg.meter_log_every = run.meter_log_every
    cfg.scene.samples_per_chirp = int(root["cube"]["samples_per_chirp"])

    tx = root.get("tx_array")
    if not tx:
        raise ValueError("tx_array section is required for MIMO")
    cfg.scene.num_tx = int(tx["num_tx"])
    if cfg.scene.num_tx < 1:
        raise ValueError("tx_array.num_tx must be >= 1")

    wavelength = waveform.wavelength()
    cfg.wavelength = wavelength
    load_scene_geometry(cfg, root, wavelength)

    if cfg.scheme == MultiplexingScheme.DDM:
        cfg.scene.waveform.chirp_repetition_interval = (
            waveform.chirp_repetition_interval()
        )
        cube = root["cube"]
        if "num_chirps" in cube:
            cfg.scene.num_chirps = int(cube["num_chirps"])
        else:
            cfg.scene.num_chirps = int(cube["chirps_per_tx"])
        if "ddm_code" not in mcc:
            raise ValueError(
                "fmcw_mimo: mimo_coherent_compensation.ddm_code is required for DDM"
            )
        cfg.ddm_code = parse_ddm_code(mcc["ddm_code"])
        if cfg.ddm_code.num_tx != cfg.scene.num_tx:
            raise ValueError("fmcw_mimo: ddm_code.num_tx must match tx_array.num_tx")
        if "generate_phase_codes" in mcc["ddm_code"]:
            kind = str(mcc["ddm_code"]["generate_phase_codes"])
            if kind in ("zadoff_chu", "zc"):
                fill_zadoff_chu_phase_codes(cfg.ddm_code, cfg.scene.num_chirps)
            elif kind == "sin":
                fill_sin_phase_codes(cfg.ddm_code, cfg.scene.num_chirps)
            else:
                raise ValueError(
                    f"fmcw_mimo: unsupported generate_phase_codes '{kind}' "
                    "(supported: zadoff_chu, zc, sin)"
                )
        cfg.ddm_code.validate()
        cfg.ddm_correlate = not ddm_codes_are_linear(cfg.ddm_code)
        if cfg.ddm_correlate:
            if cfg.ddm_code.guard_bands != 0:
                raise ValueError(
                    "fmcw_mimo: non-linear DDM phase_codes require guard_bands == 0"
                )
            num_chirps_codes = len(cfg.ddm_code.phase_codes) // cfg.ddm_code.num_tx
            if num_chirps_codes != cfg.scene.num_chirps:
                raise ValueError(
                    "fmcw_mimo: non-linear phase_codes length/num_tx must equal "
                    "cube.num_chirps"
                )
            cfg.report_num_doppler = cfg.scene.num_chirps
        else:
            cfg.ddm_doppler_offsets = resolve_ddm_offsets(cfg.ddm_code)
            if cfg.scene.num_chirps % cfg.scene.num_tx != 0:
                raise ValueError(
                    "fmcw_mimo: cube.num_chirps must be divisible by "
                    "tx_array.num_tx for linear DDM"
                )
            band = cfg.scene.num_chirps // cfg.scene.num_tx
            cfg.report_num_doppler = band - cfg.ddm_code.guard_bands
            if cfg.report_num_doppler <= 0:
                raise ValueError(
                    "fmcw_mimo: DDM guard_bands leaves no Doppler bins per TX"
                )
    else:
        sim = root.get("sim")
        if not sim or "physical_chirp_repetition_interval" not in sim:
            raise ValueError(
                "fmcw_mimo: sim.physical_chirp_repetition_interval is required for TDM"
            )
        cfg.scene.waveform.chirp_repetition_interval = float(
            sim["physical_chirp_repetition_interval"]
        )
        cfg.scene.num_chirps = int(root["cube"]["chirps_per_tx"])
        if "mimo_deinterleave" not in root:
            raise ValueError("fmcw_mimo: mimo_deinterleave section is required for TDM")
        cfg.tx_schedule = parse_tx_schedule(root["mimo_deinterleave"]["tx_schedule"])
        if cfg.tx_schedule.num_tx != cfg.scene.num_tx:
            raise ValueError(
                "mimo_deinterleave.tx_schedule.num_tx must match tx_array.num_tx"
            )
        cfg.report_num_doppler = cfg.scene.num_chirps

    cfg.tx_positions_flat = flatten_positions(cfg.scene.tx_positions)
    cfg.rx_positions_flat = flatten_positions(cfg.scene.rx_positions)

    for t in root.get("targets", []):
        position, velocity, amplitude = parse_target(t, app_name="fmcw_mimo")
        cfg.scene.targets.append(
            PointTarget(position=position, velocity=velocity, amplitude=amplitude)
        )

    noise = root.get("noise_snr_db")
    if noise is not None:
        cfg.scene.snr_db = float(noise)

    cfg.plots = parse_plots(root)
    cfg.visualize = bool(root.get("visualize", False))
    cfg.skip_display_check = bool(root.get("skip_display_check", False))
    cfg.headless = bool(root.get("headless", False))

    be = root.get("beamscan_estimator", {})
    cfg.az_lo, cfg.az_hi = grid_extent(
        [float(x) for x in be.get("az_angles", [])] or None, -0.35, 0.35
    )
    cfg.el_lo, cfg.el_hi = grid_extent(
        [float(x) for x in be.get("el_angles", [])] or None, -0.1, 0.1
    )

    cfg.range_processor = parse_range_processor(root.get("range_processor", {}))
    cfg.clutter_filter = parse_clutter_filter(root.get("clutter_filter", {}))
    cfg.doppler_processor = parse_doppler_processor(root.get("doppler_processor", {}))
    cfg.cfar_detector = parse_cfar_detector(root.get("cfar_detector", {}))
    cfg.snapshot_gather = parse_snapshot_gather(root.get("snapshot_gather", {}))
    cfg.beamscan_estimator = parse_beamscan_estimator(
        root.get("beamscan_estimator", {})
    )
    cfg.cluster_dbscan = parse_cluster_dbscan(root.get("cluster_dbscan", {}))
    cfg.reduction = parse_reduction(root.get("reduction", {}))
    cfg.mimo_deinterleave = parse_mimo_deinterleave(root.get("mimo_deinterleave", {}))
    cfg.mimo_coherent_compensation = parse_mimo_coherent_compensation(
        root.get("mimo_coherent_compensation", {})
    )
    # Prefer the app-resolved descriptor (may expand generate_phase_codes).
    if cfg.scheme == MultiplexingScheme.DDM:
        cfg.mimo_coherent_compensation["ddm_code"] = cfg.ddm_code
    cfg.virtual_array_assembly = parse_virtual_array_assembly(
        root.get("virtual_array_assembly", {})
    )
    return cfg


def cluster_layer_name(color: int) -> str:
    return f"cluster_{color}"


def make_view(ox: float, oy: float, w: float, h: float) -> HolovizOp.InputSpec.View:
    view = HolovizOp.InputSpec.View()
    view.offset_x = ox
    view.offset_y = oy
    view.width = w
    view.height = h
    return view


class DdmReportVelocityBinsOp(Operator):
    """Post-DDM-demux velocity axis for reporting (mirrors C++ app-local op)."""

    def __init__(
        self,
        fragment,
        *args,
        waveform,
        num_doppler: int,
        bin_spacing_hz: float,
        **kwargs,
    ):
        self.waveform = waveform
        self.num_doppler = num_doppler
        self.bin_spacing_hz = bin_spacing_hz
        self._velocity_bins: cp.ndarray | None = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("sync")
        spec.output("velocity_bins")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("sync")
        if self._velocity_bins is None:
            if self.num_doppler <= 0:
                raise RuntimeError("DdmReportVelocityBinsOp: num_doppler must be positive")
            if self.bin_spacing_hz <= 0.0:
                raise RuntimeError(
                    "DdmReportVelocityBinsOp: bin_spacing_hz must be > 0"
                )
            wavelength = self.waveform.wavelength()
            axis = np.zeros(self.num_doppler, dtype=np.float32)
            for i in range(self.num_doppler):
                f = (i - self.num_doppler // 2) * self.bin_spacing_hz
                axis[i] = -f * wavelength / 2.0
            self._velocity_bins = cp.asarray(axis)
        op_output.emit({"velocity_bins": self._velocity_bins}, "velocity_bins")


class MimoTdmAdcSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        scene: MimoScene,
        tx_schedule: TXSchedule,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.tx_schedule = tx_schedule
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        result = simulate_mimo_tdm_interleaved(self.scene, self.tx_schedule)
        if len(result.shape) != 3:
            raise RuntimeError("MimoTdmAdcSourceOp: expected rank-3 (N_rx, M*L, samples)")
        # Keep a reference so the DLPack buffer outlives this compute().
        self._adc_cube = result.cube
        op_output.emit({"adc_cube": self._adc_cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


class MimoDdmAdcSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        scene: MimoScene,
        doppler_offsets: list[float] | None = None,
        phase_codes: list[complex] | None = None,
        use_phase_codes: bool = False,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.doppler_offsets = doppler_offsets or []
        self.phase_codes = phase_codes or []
        self.use_phase_codes = use_phase_codes
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        if self.use_phase_codes:
            result = simulate_mimo_ddm_phase_coded(self.scene, self.phase_codes)
        else:
            result = simulate_mimo_ddm_coded(self.scene, self.doppler_offsets)
        if len(result.shape) != 3:
            raise RuntimeError(
                "MimoDdmAdcSourceOp: expected rank-3 (N_rx, num_chirps, samples)"
            )
        self._adc_cube = result.cube
        op_output.emit({"adc_cube": self._adc_cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


class TensorMapDiscardOp(Operator):
    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("input")


class FmcwMimoApp(Application):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.cadence: CadenceMeter | None = None

    def compose(self) -> None:
        cfg = self.cfg
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

        range_op = FMCWRangeProcessorOp(self, **cfg.range_processor)
        clutter = ClutterFilterOp(self, **cfg.clutter_filter)
        doppler = FMCWDopplerProcessorOp(self, **cfg.doppler_processor)
        compensation = MIMOCoherentCompensationOp(
            self, **cfg.mimo_coherent_compensation
        )
        vaa = VirtualArrayAssemblyOp(
            self,
            tx_positions=cfg.tx_positions_flat,
            rx_positions=cfg.rx_positions_flat,
            wavelength=cfg.wavelength,
            **cfg.virtual_array_assembly,
        )

        if ddm:
            source = MimoDdmAdcSourceOp(
                self,
                CountCondition(self, count=cfg.num_frames),
                PeriodicCondition(self, recess_period=period_ns),
                scene=cfg.scene,
                doppler_offsets=cfg.ddm_doppler_offsets,
                phase_codes=list(cfg.ddm_code.phase_codes)
                if cfg.ddm_correlate
                else None,
                use_phase_codes=cfg.ddm_correlate,
                frame_period=cfg.frame_period,
                cadence=cadence,
                name="adc_source",
            )
            self.add_flow(source, range_op, {("adc_cube", "adc_cube")})
            self.add_flow(range_op, doppler, {("range_profiles", "range_profiles")})
            self.add_flow(doppler, compensation, {("rd_maps", "rd_maps")})
            self.add_flow(compensation, clutter, {("rd_maps", "range_profiles")})
            self.add_flow(clutter, vaa, {("range_profiles", "rd_maps")})
        else:
            source = MimoTdmAdcSourceOp(
                self,
                CountCondition(self, count=cfg.num_frames),
                PeriodicCondition(self, recess_period=period_ns),
                scene=cfg.scene,
                tx_schedule=cfg.tx_schedule,
                frame_period=cfg.frame_period,
                cadence=cadence,
                name="adc_source",
            )
            deinterleave = MIMODeinterleaveOp(self, **cfg.mimo_deinterleave)
            self.add_flow(source, deinterleave, {("adc_cube", "adc_cube")})
            self.add_flow(deinterleave, range_op, {("adc_cube", "adc_cube")})
            self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
            self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
            self.add_flow(doppler, compensation, {("rd_maps", "rd_maps")})
            self.add_flow(compensation, vaa, {("rd_maps", "rd_maps")})

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

        ddm_velocity_bins: DdmReportVelocityBinsOp | None = None
        if ddm:
            ddm_velocity_bins = DdmReportVelocityBinsOp(
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

        self.add_flow(vaa, cfar, {("rd_maps", "rd_maps")})
        self.add_flow(vaa, gather, {("rd_maps", "rd_maps")})
        self.add_flow(cfar, gather, {("detection_idx", "detection_idx")})
        self.add_flow(gather, beamscan, {("snapshot", "snapshot"), ("is_valid", "is_valid")})
        self.add_flow(vaa, beamscan, {("array_geometry", "array_geometry")})
        self.add_flow(beamscan, feature_assembly, {("aoa_estimates", "aoa_estimates")})
        self.add_flow(beamscan, spectrum_sink, {("angle_spectrum", "input")})

        self.add_flow(cfar, det_report, {("detection_idx", "detection_idx")})
        self.add_flow(cfar, det_report, {("detection_score", "detection_score")})
        self.add_flow(range_op, det_report, {("range_bins", "range_bins")})
        if ddm and ddm_velocity_bins is not None:
            self.add_flow(
                ddm_velocity_bins, det_report, {("velocity_bins", "velocity_bins")}
            )
        else:
            self.add_flow(doppler, det_report, {("velocity_bins", "velocity_bins")})
        self.add_flow(cfar, mask_sink, {("detection_mask", "input")})
        self.add_flow(cfar, noise_sink, {("noise_estimate", "input")})

        self.add_flow(cfar, feature_assembly, {("detection_idx", "detection_idx")})
        self.add_flow(range_op, feature_assembly, {("range_bins", "range_bins")})
        if ddm and ddm_velocity_bins is not None:
            self.add_flow(
                ddm_velocity_bins,
                feature_assembly,
                {("velocity_bins", "velocity_bins")},
            )
        else:
            self.add_flow(
                doppler, feature_assembly, {("velocity_bins", "velocity_bins")}
            )
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
            self.add_flow(
                doppler, cluster_report, {("velocity_bins", "velocity_bins")}
            )

        if cfg.visualize:
            ch_slice = RdChannelSliceOp(self, channel=0, name="rd_channel_slice")
            to_image = RdMapToImageOp(self, name="rd_map_to_image")
            overlay = ClusterOverlayOp(self, name="cluster_overlay")

            input_specs: list[HolovizOp.InputSpec] = []
            input_specs.append(
                HolovizOp.InputSpec("rd_map", HolovizOp.InputType.COLOR_LUT)
            )
            palette = cluster_palette()

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
                plot.velocity_ylim_lo = cfg.plots.velocity_ylim_lo
                plot.velocity_ylim_hi = cfg.plots.velocity_ylim_hi
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
                window_title=window_title,
                width=1200 if cfg.plots.enable else 800,
                height=760 if cfg.plots.enable else 600,
                headless=cfg.headless,
                name="holoviz",
            )

            self.add_flow(vaa, ch_slice, {("rd_maps", "rd_maps")})
            self.add_flow(ch_slice, to_image, {("rd_map", "rd_maps")})
            self.add_flow(to_image, visualizer, {("rd_image", "receivers")})
            self.add_flow(cfar, overlay, {("detection_idx", "detection_idx")})
            self.add_flow(cluster, overlay, {("labels", "labels")})
            self.add_flow(ch_slice, overlay, {("rd_map", "rd_maps")})
            self.add_flow(overlay, visualizer, {("overlay", "receivers")})

            if plot is not None:
                self.add_flow(reduction, plot, {("rep_idx", "rep_idx")})
                self.add_flow(range_op, plot, {("range_bins", "range_bins")})
                if ddm and ddm_velocity_bins is not None:
                    self.add_flow(
                        ddm_velocity_bins, plot, {("velocity_bins", "velocity_bins")}
                    )
                else:
                    self.add_flow(
                        doppler, plot, {("velocity_bins", "velocity_bins")}
                    )
                self.add_flow(plot, visualizer, {("plots", "receivers")})
            if boresight is not None:
                self.add_flow(beamscan, boresight, {("aoa_estimates", "aoa_estimates")})
                self.add_flow(cluster, boresight, {("labels", "labels")})
                self.add_flow(boresight, visualizer, {("plot", "receivers")})


def resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    return Path(__file__).resolve().parent.parent / "cpp" / "fmcw_mimo.yaml"


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
                "fmcw_mimo: visualize enabled but no display; running without GUI"
            )
            cfg.visualize = False

        app = FmcwMimoApp(cfg)
        app.run()
        if app.cadence is not None:
            app.cadence.log_summary()
    except Exception as exc:
        logger.error("fmcw_mimo: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
