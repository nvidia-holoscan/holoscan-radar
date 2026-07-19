# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared YAML helpers for Python demo apps (simulator / app-level fields and
# optional kwargs parsers). Prefer Holoscan ``Fragment.from_config("block")``
# for operator parameters — radar bindings forward Arg/ArgList.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from holoscan.radar import (
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    ClutterFilterMethod,
    CfarVariant,
    DDMCodeDescriptor,
    FftNorm,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    MultiplexingScheme,
    OverlapPolicy,
    PeakInterpolation,
    RepresentativePolicy,
    ScoreMethod,
    SnapshotEdgeMode,
    TXSchedule,
    WindowSpec,
    WindowType,
)

K_C = 299792458.0


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
            f"port_tensor: key {key!r} not in TensorMap with keys "
            f"{list(message.keys())}"
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
        conv.preset = parse_enum(
            AngleConventionPreset, node["preset"], "AngleConventionPreset"
        )
    return conv


def parse_range_processor(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "range_processor"}
    if "waveform" in block:
        kwargs["waveform"] = parse_waveform(block["waveform"])
    if "range_window" in block:
        kwargs["range_window"] = parse_window_spec(block["range_window"])
    if "apply_dechirp" in block:
        kwargs["apply_dechirp"] = bool(block["apply_dechirp"])
    if "fft_norm" in block:
        kwargs["fft_norm"] = parse_enum(FftNorm, block["fft_norm"], "FftNorm")
    if "propagation_speed" in block:
        kwargs["propagation_speed"] = float(block["propagation_speed"])
    if "nfft" in block:
        kwargs["nfft"] = int(block["nfft"])
    return kwargs


def parse_clutter_filter(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "clutter_filter"}
    if "method" in block:
        kwargs["method"] = parse_enum(
            ClutterFilterMethod, block["method"], "ClutterFilterMethod"
        )
    return kwargs


def parse_doppler_processor(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "doppler_processor"}
    if "waveform" in block:
        kwargs["waveform"] = parse_waveform(block["waveform"])
    if "doppler_window" in block:
        kwargs["doppler_window"] = parse_window_spec(block["doppler_window"])
    if "center_zero_doppler" in block:
        kwargs["center_zero_doppler"] = bool(block["center_zero_doppler"])
    if "fft_norm" in block:
        kwargs["fft_norm"] = parse_enum(FftNorm, block["fft_norm"], "FftNorm")
    if "nfft" in block:
        kwargs["nfft"] = int(block["nfft"])
    return kwargs


def parse_cfar_detector(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "cfar_detector"}
    if "cfar_variant" in block:
        kwargs["cfar_variant"] = parse_enum(
            CfarVariant, block["cfar_variant"], "CfarVariant"
        )
    if "os_rank" in block:
        kwargs["os_rank"] = int(block["os_rank"])
    if "guard_cells" in block:
        kwargs["guard_cells"] = [int(x) for x in block["guard_cells"]]
    if "training_cells" in block:
        kwargs["training_cells"] = [int(x) for x in block["training_cells"]]
    if "pfa" in block:
        kwargs["pfa"] = float(block["pfa"])
    if "alpha_lut" in block:
        kwargs["alpha_lut"] = [float(x) for x in block["alpha_lut"]]
    if "threshold_scale" in block:
        kwargs["threshold_scale"] = float(block["threshold_scale"])
    if "nci_axes" in block:
        kwargs["nci_axes"] = [int(x) for x in block["nci_axes"]]
    if "max_detections" in block:
        kwargs["max_detections"] = int(block["max_detections"])
    if "max_cached_kernels" in block:
        kwargs["max_cached_kernels"] = int(block["max_cached_kernels"])
    if "blocking_alpha_lut_rebuild" in block:
        kwargs["blocking_alpha_lut_rebuild"] = bool(
            block["blocking_alpha_lut_rebuild"]
        )
    return kwargs


def parse_snapshot_gather(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "snapshot_gather"}
    if "snapshot_window" in block:
        kwargs["snapshot_window"] = [int(x) for x in block["snapshot_window"]]
    if "doppler_edge" in block:
        kwargs["doppler_edge"] = parse_enum(
            SnapshotEdgeMode, block["doppler_edge"], "SnapshotEdgeMode"
        )
    if "range_edge" in block:
        kwargs["range_edge"] = parse_enum(
            SnapshotEdgeMode, block["range_edge"], "SnapshotEdgeMode"
        )
    return kwargs


def parse_beamscan_estimator(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "beamscan_estimator"}
    if "angle_convention" in block:
        kwargs["angle_convention"] = parse_angle_convention(block["angle_convention"])
    if "az_angles" in block:
        kwargs["az_angles"] = [float(x) for x in block["az_angles"]]
    if "el_angles" in block:
        kwargs["el_angles"] = [float(x) for x in block["el_angles"]]
    if "array_taper" in block:
        kwargs["array_taper"] = parse_window_spec(block["array_taper"])
    if "peak_interpolation" in block:
        kwargs["peak_interpolation"] = parse_enum(
            PeakInterpolation, block["peak_interpolation"], "PeakInterpolation"
        )
    return kwargs


def parse_phase_shift_beamformer(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "phase_shift_beamformer"}
    if "angle_convention" in block:
        kwargs["angle_convention"] = parse_angle_convention(block["angle_convention"])
    if "az_angles" in block:
        kwargs["az_angles"] = [float(x) for x in block["az_angles"]]
    if "el_angles" in block:
        kwargs["el_angles"] = [float(x) for x in block["el_angles"]]
    if "array_taper" in block:
        kwargs["array_taper"] = parse_window_spec(block["array_taper"])
    return kwargs


def parse_cluster_dbscan(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "cluster_dbscan"}
    if "epsilon" in block:
        kwargs["epsilon"] = [float(x) for x in block["epsilon"]]
    if "min_points" in block:
        kwargs["min_points"] = int(block["min_points"])
    if "max_clusters" in block:
        kwargs["max_clusters"] = int(block["max_clusters"])
    return kwargs


def parse_reduction(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "detection_reduction"}
    if "representative" in block:
        kwargs["representative"] = parse_enum(
            RepresentativePolicy, block["representative"], "RepresentativePolicy"
        )
    if "score_method" in block:
        kwargs["score_method"] = parse_enum(
            ScoreMethod, block["score_method"], "ScoreMethod"
        )
    if "feature_weights" in block:
        kwargs["feature_weights"] = [float(x) for x in block["feature_weights"]]
    return kwargs


def parse_mimo_deinterleave(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "mimo_deinterleave"}
    if "tx_schedule" in block:
        kwargs["tx_schedule"] = parse_tx_schedule(block["tx_schedule"])
    return kwargs


def parse_mimo_coherent_compensation(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "mimo_coherent_compensation"}
    if "multiplexing_scheme" in block:
        kwargs["multiplexing_scheme"] = parse_enum(
            MultiplexingScheme, block["multiplexing_scheme"], "MultiplexingScheme"
        )
    if "waveform" in block:
        kwargs["waveform"] = parse_waveform(block["waveform"])
    if "ddm_code" in block:
        kwargs["ddm_code"] = parse_ddm_code(block["ddm_code"])
    return kwargs


def parse_virtual_array_assembly(block: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": "virtual_array_assembly"}
    if "overlap_policy" in block:
        kwargs["overlap_policy"] = parse_enum(
            OverlapPolicy, block["overlap_policy"], "OverlapPolicy"
        )
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
            raise ValueError(
                "ddm_code.phase_codes must be interleaved real/imag pairs"
            )
        code.phase_codes = [
            complex(float(raw[i]), float(raw[i + 1]))
            for i in range(0, len(raw), 2)
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
        raise ValueError(
            f"{app_name}: each target's velocity must be a 3-vector [vx, vy, vz]"
        )
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
class RunConfig:
    num_frames: int
    frame_period: float
    meter_cadence: bool = True
    meter_log_every: int = 0


def parse_run(root: dict[str, Any]) -> RunConfig:
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
            f"run.frame_period must be within [1 ns, INT64_MAX ns], got "
            f"{frame_period} s"
        )
    meter_cadence = bool(run.get("meter_cadence", True))
    meter_log_every = int(run.get("meter_log_every", 0))
    if meter_log_every < 0:
        raise ValueError("run.meter_log_every must be >= 0")
    return RunConfig(
        num_frames=num_frames,
        frame_period=frame_period,
        meter_cadence=meter_cadence,
        meter_log_every=meter_log_every,
    )
