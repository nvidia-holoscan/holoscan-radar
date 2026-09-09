# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared YAML helpers for Python demo apps (simulator / app-level fields and
# optional kwargs parsers). Prefer Holoscan ``Fragment.from_config("block")``
# for operator parameters — radar bindings forward Arg/ArgList.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# holoscan.radar must load before common.python.sim (which imports cupy) -
# MatX CUDA cache / atexit ordering, same rule the app entry points follow.
# isort: off
from holoscan.operators import HolovizOp
from holoscan.radar import (
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    DDMCodeDescriptor,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    TXSchedule,
    WindowSpec,
    WindowType,
)

from common.python.sim import PointTarget

# isort: on


def port_tensor(message: Any, key: str) -> Any:
    """Extract a tensor from a Holoscan port message (TensorMap or bare tensor).

    Matches C++ ``receive_single``: the port name is the contract; a single-entry
    TensorMap is accepted regardless of map key (emit keys often keep the
    upstream name when ``add_flow`` remaps ports, e.g. detection_idx → idx).
    """
    if isinstance(message, dict):
        if key in message:
            return message[key]
        if len(message) == 1:
            return next(iter(message.values()))
        raise KeyError(
            f"port_tensor: key {key!r} not in TensorMap with keys " f"{list(message.keys())}"
        )
    return message


def tensor_to_numpy(tensor_like: Any) -> np.ndarray:
    """Copy a received tensor / CuPy / numpy object to a contiguous numpy array."""
    import cupy as cp

    shape = getattr(tensor_like, "shape", None)
    if shape is not None:
        try:
            if int(np.prod(tuple(shape))) == 0:
                dt = getattr(tensor_like, "dtype", None)
                if dt is not None:
                    code = int(getattr(dt, "code", 2))
                    bits = int(getattr(dt, "bits", 32))
                    if code == 0:
                        np_dtype = np.dtype(f"int{bits}")
                    elif code == 1:
                        np_dtype = np.dtype(f"uint{bits}")
                    elif code == 6:
                        np_dtype = np.bool_
                    elif code == 5:
                        np_dtype = np.dtype(f"complex{bits * 2}")
                    else:
                        np_dtype = np.dtype(f"float{bits}")
                else:
                    np_dtype = np.float64
                return np.empty(tuple(shape), dtype=np_dtype)
        except (TypeError, ValueError):
            pass

    if hasattr(tensor_like, "__dlpack__"):
        try:
            return np.ascontiguousarray(np.from_dlpack(tensor_like))
        except (TypeError, RuntimeError, BufferError, ValueError):
            pass
    try:
        return np.ascontiguousarray(cp.asnumpy(cp.asarray(tensor_like)))
    except (TypeError, RuntimeError, ValueError, BufferError):
        pass
    return np.ascontiguousarray(np.asarray(tensor_like))


def parse_enum(enum_cls: type, value: str, type_name: str) -> Any:
    normalized = value.strip().lower()
    for name in dir(enum_cls):
        if not name.isupper() or name.startswith("_"):
            continue
        if name.lower() == normalized:
            return getattr(enum_cls, name)
    raise ValueError(f"unknown {type_name}: {value!r}")


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


def parse_window_spec(node: Any) -> WindowSpec:
    spec = WindowSpec()
    if isinstance(node, str):
        spec.type = parse_enum(WindowType, node, "WindowType")
        return spec
    if not isinstance(node, dict):
        raise ValueError(f"window spec must be a string or mapping, got {type(node)}")
    if "type" in node:
        spec.type = parse_enum(WindowType, node["type"], "WindowType")
    if "sidelobe_attenuation_db" in node:
        spec.sidelobe_attenuation_db = float(node["sidelobe_attenuation_db"])
    if "kaiser_beta" in node and node["kaiser_beta"] is not None:
        spec.kaiser_beta = float(node["kaiser_beta"])
    if "taylor_nbar" in node:
        spec.taylor_nbar = int(node["taylor_nbar"])
    if "tukey_alpha" in node:
        spec.tukey_alpha = float(node["tukey_alpha"])
    if "coefficients" in node:
        spec.coefficients = [float(x) for x in node["coefficients"]]
    if "coherent_gain_normalization" in node:
        spec.coherent_gain_normalization = bool(node["coherent_gain_normalization"])
    return spec


def parse_angle_convention(node: Any) -> AngleConvention:
    conv = AngleConvention()
    if isinstance(node, dict) and "preset" in node:
        conv.preset = parse_enum(AngleConventionPreset, node["preset"], "AngleConventionPreset")
    return conv


def parse_tdm_dense_compensation(block: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for TDMDenseCompensationOp."""
    kwargs: dict[str, Any] = {"name": "tdm_dense_compensation"}
    if "waveform" in block:
        kwargs["waveform"] = parse_waveform(block["waveform"])
    if "tx_schedule" in block:
        kwargs["tx_schedule"] = parse_tx_schedule(block["tx_schedule"])
    return kwargs


def parse_ddm_dense_demultiplex(block: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for DDMDenseDemultiplexOp."""
    kwargs: dict[str, Any] = {"name": "ddm_dense_demultiplex"}
    if "ddm_code" in block:
        kwargs["ddm_code"] = parse_ddm_code(block["ddm_code"])
    return kwargs


def parse_tx_schedule(node: Any) -> TXSchedule:
    sched = TXSchedule()
    if not isinstance(node, dict):
        raise ValueError("tx_schedule must be a mapping")
    if "num_tx" in node:
        sched.num_tx = int(node["num_tx"])
    if "tx_order" in node:
        sched.tx_order = [int(x) for x in node["tx_order"]]
    if "chirp_tx_index" in node:
        sched.chirp_tx_index = [int(x) for x in node["chirp_tx_index"]]
    return sched


def parse_ddm_code(node: Any) -> DDMCodeDescriptor:
    code = DDMCodeDescriptor()
    if not isinstance(node, dict):
        raise ValueError("ddm_code must be a mapping")
    if "num_tx" in node:
        code.num_tx = int(node["num_tx"])
    if "phase_codes" in node:
        raw = list(node["phase_codes"])
        # YAML stores interleaved real/imag pairs (matches C++ yaml_converters).
        if len(raw) % 2 != 0:
            raise ValueError("ddm_code.phase_codes must be interleaved real/imag pairs")
        code.phase_codes = [
            complex(float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw), 2)
        ]
    if "doppler_offsets" in node:
        code.doppler_offsets = [float(x) for x in node["doppler_offsets"]]
    if "guard_bands" in node:
        code.guard_bands = int(node["guard_bands"])
    return code


def make_ura_geometry(
    az_elements: int, el_elements: int, spacing_wl: float, wavelength: float
) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.URA
    positions: list[float] = []
    for iz in range(el_elements):
        for iy in range(az_elements):
            positions.extend(
                [
                    0.0,
                    float(iy) * spacing_wl * wavelength,
                    float(iz) * spacing_wl * wavelength,
                ]
            )
    geom.element_positions = positions
    geom.validate()
    return geom


def make_virtual_geometry(
    tx_positions: list[tuple[float, float, float]],
    rx_positions: list[tuple[float, float, float]],
    wavelength: float,
) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.CONFORMAL
    positions: list[float] = []
    for tp in tx_positions:
        for rp in rx_positions:
            positions.extend([tp[0] + rp[0], tp[1] + rp[1], tp[2] + rp[2]])
    geom.element_positions = positions
    geom.validate()
    return geom


def flatten_positions(positions: list[tuple[float, float, float]]) -> list[float]:
    flat: list[float] = []
    for p in positions:
        flat.extend(p)
    return flat


def grid_extent(
    grid: list[float] | None,
    def_lo: float,
    def_hi: float,
) -> tuple[float, float]:
    if grid:
        lo = min(grid)
        hi = max(grid)
    else:
        lo, hi = def_lo, def_hi
    if hi - lo < 1e-6:
        mid = 0.5 * (lo + hi)
        lo = mid - 0.1
        hi = mid + 0.1
    return lo, hi


def parse_target(node: dict[str, Any], *, app_name: str) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    complex,
]:
    range_m = float(node["range"])
    azimuth = float(node["azimuth"])
    elevation = float(node["elevation"])
    amplitude = complex(float(node["amplitude"]), 0.0)
    vel = node.get("velocity")
    if not isinstance(vel, list) or len(vel) != 3:
        raise ValueError(f"{app_name}: each target's velocity must be a 3-vector [vx, vy, vz]")
    direction = (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
    position = (
        range_m * direction[0],
        range_m * direction[1],
        range_m * direction[2],
    )
    velocity = (float(vel[0]), float(vel[1]), float(vel[2]))
    return position, velocity, amplitude


def parse_cube_dims(root: dict[str, Any]) -> tuple[int, int]:
    """cube.{num_chirps, samples_per_chirp} + positivity validation.

    Mirrors ``radar_app/app/config.hpp::parse_cube``. MIMO parses its own
    cube (``chirps_per_tx`` vs ``num_chirps`` depending on TDM/DDM) and does
    not use this.
    """
    cube = root["cube"]
    num_chirps = int(cube["num_chirps"])
    samples_per_chirp = int(cube["samples_per_chirp"])
    if num_chirps <= 0:
        raise ValueError(f"cube.num_chirps must be > 0, got {num_chirps}")
    if samples_per_chirp <= 0:
        raise ValueError(f"cube.samples_per_chirp must be > 0, got {samples_per_chirp}")
    return num_chirps, samples_per_chirp


def parse_noise(root: dict[str, Any], scene: Any) -> None:
    """``noise_snr_db`` -> ``scene.snr_db`` (unset leaves it None / noise-free)."""
    noise = root.get("noise_snr_db")
    if noise is not None:
        scene.snr_db = float(noise)


def parse_targets_range1d(root: dict[str, Any]) -> list[PointTarget]:
    """1-D targets (SISO): (range, velocity) along boresight, real amplitude."""
    targets: list[PointTarget] = []
    for node in root.get("targets", []):
        range_m = float(node["range"])
        velocity = float(node["velocity"])
        amplitude = float(node["amplitude"])
        targets.append(
            PointTarget(
                position=(range_m, 0.0, 0.0),
                velocity=(velocity, 0.0, 0.0),
                amplitude=complex(amplitude, 0.0),
            )
        )
    return targets


@dataclass
class PlotsConfig:
    enable: bool = True
    window: int = 120
    velocity_ylim_lo: float = float("nan")
    velocity_ylim_hi: float = float("nan")


def parse_plots(root: dict[str, Any]) -> PlotsConfig:
    plots = PlotsConfig()
    node = root.get("plots")
    if isinstance(node, dict):
        if "enable" in node:
            plots.enable = bool(node["enable"])
        if "window" in node:
            plots.window = int(node["window"])
        ylim = node.get("velocity_ylim")
        if ylim is not None:
            if not isinstance(ylim, list) or len(ylim) != 2:
                raise ValueError("plots.velocity_ylim must be a [lo, hi] pair")
            plots.velocity_ylim_lo = float(ylim[0])
            plots.velocity_ylim_hi = float(ylim[1])
    if plots.window <= 1:
        raise ValueError(f"plots.window must be > 1, got {plots.window}")
    return plots


@dataclass
class VisualizationConfig:
    """Detection-marker geometry for the RD-map overlay (ClusterOverlayOp), as a
    fraction of the image extent. Python analogue of
    ``config.hpp::CommonRunConfig::VisualizationConfig``.
    """

    marker_radius: float = 0.004
    marker_outline: float = 0.002
    marker_segments: int = 20
    # Vertical sync. True -> HolovizOp uses a FIFO present mode (frame-rate
    # capped, internally queued); False -> IMMEDIATE (uncapped, may tear). FIFO
    # issues far fewer swap-chain acquires per window resize, making the resize
    # path markedly more robust against Holoviz's fatal "Failed to acquire next
    # swap chain image" on an out-of-date surface.
    vsync: bool = True


def parse_visualization(root: dict[str, Any]) -> tuple[bool, VisualizationConfig]:
    """Parse the ``visualization:`` block: the ``enabled`` master live-Holoviz
    switch (falling back to the legacy top-level ``visualize:`` bool when the
    block is absent) plus the detection-marker geometry. Mirrors the
    ``visualization`` handling in ``config.hpp::parse_viz_toggles``. Returns
    ``(enabled, VisualizationConfig)``.
    """
    viz = VisualizationConfig()
    node = root.get("visualization")
    if isinstance(node, dict):
        enabled = bool(node.get("enabled", False))
        if "marker_radius" in node:
            viz.marker_radius = float(node["marker_radius"])
        if "marker_outline" in node:
            viz.marker_outline = float(node["marker_outline"])
        if "marker_segments" in node:
            viz.marker_segments = int(node["marker_segments"])
        if "vsync" in node:
            viz.vsync = bool(node["vsync"])
    else:
        enabled = bool(root.get("visualize", False))  # legacy top-level toggle
    if viz.marker_radius <= 0.0:
        raise ValueError("visualization.marker_radius must be > 0")
    if viz.marker_outline < 0.0:
        raise ValueError("visualization.marker_outline must be >= 0")
    if viz.marker_segments < 3:
        raise ValueError(f"visualization.marker_segments must be >= 3, got {viz.marker_segments}")
    return enabled, viz


@dataclass
class CommonRunConfig:
    """Run cadence + visualization toggles shared by every FMCW demo app.

    Python analogue of ``radar_app/app/config.hpp::CommonRunConfig``: each
    app's ``AppConfig`` inherits from this (dataclass inheritance) and adds
    its own scene + app-specific fields. ``run_fmcw_app()`` (driver.py) reads
    ``visualize`` / ``headless`` / ``skip_display_check`` through the derived
    config.
    """

    config_path: Path | None = None
    num_frames: int = 1
    frame_period: float = 0.1
    meter_cadence: bool = True
    meter_log_every: int = 0
    visualize: bool = False
    skip_display_check: bool = False
    headless: bool = False
    # CFAR peak-grouping mode. A native CFARDetectorOp parameter (set through the
    # cfar_detector YAML block via from_config); each app also reads the same key
    # as a read-only mirror of that operator block to gate its graph shape (drop
    # the DBSCAN clustering + reduction stages when set). Populated per app in
    # its load_config, not by parse_common_run_config below.
    peak_only: bool = False
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)


def parse_common_run_config(root: dict[str, Any], config_path: Path) -> CommonRunConfig:
    """Parse the app-owned run/viz-toggle/plots sections (waveform, cube, and
    targets are app-specific and stay in each app's own ``load_config``).
    Mirrors ``radar_app/app/config.hpp``'s ``parse_run`` + ``parse_viz_toggles``
    + ``parse_plots``, folded into one call (Python prefers returning a value
    over C++'s in-place mutation of an out-param).
    """
    run = root["run"]
    num_frames = int(run["num_frames"])
    frame_period = float(run["frame_period"])
    if num_frames <= 0:
        raise ValueError(f"run.num_frames must be > 0, got {num_frames}")
    if frame_period <= 0 or not math.isfinite(frame_period):
        raise ValueError("run.frame_period must be finite and > 0")
    frame_period_ns = frame_period * 1e9
    if frame_period_ns < 1.0 or frame_period_ns > float(2**63 - 1):
        raise ValueError(
            f"run.frame_period must be within [1 ns, INT64_MAX ns], got " f"{frame_period} s"
        )
    meter_log_every = int(run.get("meter_log_every", 0))
    if meter_log_every < 0:
        raise ValueError("run.meter_log_every must be >= 0")
    visualize, viz = parse_visualization(root)
    return CommonRunConfig(
        config_path=config_path,
        num_frames=num_frames,
        frame_period=frame_period,
        meter_cadence=bool(run.get("meter_cadence", True)),
        meter_log_every=meter_log_every,
        visualize=visualize,
        skip_display_check=bool(root.get("skip_display_check", False)),
        headless=bool(root.get("headless", False)),
        plots=parse_plots(root),
        viz=viz,
    )


def cluster_layer_name(color: int) -> str:
    return f"cluster_{color}"


K_DETECTION_OUTLINE_LAYER = "detection_outline"
K_NOISE_LAYER_NAME = "noise"


def detection_layer_specs(palette: list[list[float]]) -> list[HolovizOp.InputSpec]:
    """Holoviz InputSpecs for the CFAR detection markers, matching the tensors
    ClusterOverlayOp emits. Detections render as filled triangle-fan discs: a
    single black outline layer of larger discs (priority 1) sits under one
    colored disc layer per cluster palette color plus a muted noise layer
    (priority 2). All draw over the RD image (COLOR_LUT, default priority 0)."""
    specs: list[HolovizOp.InputSpec] = []
    outline = HolovizOp.InputSpec(K_DETECTION_OUTLINE_LAYER, HolovizOp.InputType.TRIANGLES)
    outline.color = [0.0, 0.0, 0.0, 1.0]
    outline.priority = 1
    specs.append(outline)

    def add_discs(name: str, color: list[float]) -> None:
        spec = HolovizOp.InputSpec(name, HolovizOp.InputType.TRIANGLES)
        spec.color = color
        spec.priority = 2
        specs.append(spec)

    for c in range(len(palette)):
        add_discs(cluster_layer_name(c), palette[c])
    add_discs(K_NOISE_LAYER_NAME, [0.6, 0.6, 0.6, 1.0])
    return specs


# Shared panel-box margins (mirror kBoxL/R/T/B in viz.hpp): the fraction of a
# panel/region occupied by the plot box, leaving margins for title + tick labels.
K_BOX_L, K_BOX_R, K_BOX_T, K_BOX_B = 0.15, 0.98, 0.14, 0.85


def rd_map_region(paneled: bool, left_width: float) -> tuple[float, float, float, float]:
    """RD-map outer region (window-normalized): the left column when the side
    panels are shown, else the full window. Mirrors holoviz_layout.hpp."""
    return (0.0, 0.0, left_width, 1.0) if paneled else (0.0, 0.0, 1.0, 1.0)


def rd_map_box(paneled: bool, left_width: float) -> tuple[float, float, float, float]:
    """The inset RD image box within the region, using the shared kBox margins so
    the image is framed like the side-panel plot boxes rather than stretching to
    the window edge. The RD image + detection discs use this as their View;
    emit_rd_axes draws the frame on its edges. Mirrors holoviz_layout.hpp."""
    x, y, w, h = rd_map_region(paneled, left_width)
    return (
        x + K_BOX_L * w,
        y + K_BOX_T * h,
        (K_BOX_R - K_BOX_L) * w,
        (K_BOX_B - K_BOX_T) * h,
    )


def rd_map_view_aspect(width: int, height: int, left_width: float, paneled: bool) -> float:
    """On-screen aspect ratio (width / height in pixels) of the inset RD-map image
    box (see rd_map_box). ClusterOverlayOp divides its disc x radius by this so
    the detection markers render round rather than stretched."""
    _, _, bw, bh = rd_map_box(paneled, left_width)
    return (bw * float(width)) / (bh * float(height))


def make_view(ox: float, oy: float, w: float, h: float) -> HolovizOp.InputSpec.View:
    view = HolovizOp.InputSpec.View()
    view.offset_x = ox
    view.offset_y = oy
    view.width = w
    view.height = h
    return view
