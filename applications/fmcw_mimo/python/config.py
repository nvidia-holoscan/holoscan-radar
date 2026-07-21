# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-owned config for fmcw_mimo: the AppConfig type and its YAML loader
# (TDM/DDM branch, virtual array geometry, targets, the DDM code helpers).
# Colocated with the app; the graph wiring lives in fmcw_mimo.py. Operator
# blocks are NOT parsed here except mimo_coherent_compensation, which must
# stay manual (see parse_mimo_coherent_compensation's docstring) — every
# other block flows through Holoscan's from_config().
#
# Imported bare (``import config``) from fmcw_mimo.py: each app is invoked as
# a script, which puts this file's own directory at sys.path[0] - the Python
# analogue of C++'s colocated ``#include "config.hpp"  // NOLINT(...)``. Do
# not import this module by its bare name from another process: `config` is
# not a qualified package name, so importing two apps' config.py in the same
# interpreter would collide in sys.modules.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from common.python.demo_config import (
    CommonRunConfig,
    flatten_positions,
    grid_extent,
    make_virtual_geometry,
    parse_common_run_config,
    parse_ddm_code,
    parse_mimo_coherent_compensation,
    parse_noise,
    parse_target,
    parse_tx_schedule,
    parse_waveform,
)
from common.python.mimo_sim import (
    MimoScene,
    PointTarget,
    SimWaveform,
    ddm_codes_are_linear,
    fill_sin_phase_codes,
    fill_zadoff_chu_phase_codes,
    resolve_ddm_offsets,
)

from holoscan.radar import MultiplexingScheme, TXSchedule


@dataclass
class AppConfig(CommonRunConfig):
    scheme: MultiplexingScheme = MultiplexingScheme.TDM
    scene: MimoScene = field(default_factory=MimoScene)
    tx_schedule: TXSchedule = field(default_factory=TXSchedule)
    ddm_code: object = None
    ddm_doppler_offsets: list[float] = field(default_factory=list)
    ddm_correlate: bool = False
    array_geometry: object = None
    tx_positions_flat: list[float] = field(default_factory=list)
    rx_positions_flat: list[float] = field(default_factory=list)
    wavelength: float = 0.0
    # Boresight-view axis extents only (beamscan_estimator's az_angles /
    # el_angles are read-only mirrored here to size the plot, not to
    # configure the operator - that flows through from_config()).
    az_lo: float = -0.35
    az_hi: float = 0.35
    el_lo: float = -0.1
    el_hi: float = 0.1
    report_waveform: object = None
    report_num_doppler: int = 0
    # mimo_coherent_compensation stays a manual kwargs dict (see
    # parse_mimo_coherent_compensation): fmcw_mimo.py builds the op from
    # **cfg.mimo_coherent_compensation, not self.from_config(...).
    mimo_coherent_compensation: dict = field(default_factory=dict)


def _scale_position(iy: float, iz: float, spacing_wl: float, wavelength: float):
    return (0.0, iy * spacing_wl * wavelength, iz * spacing_wl * wavelength)


def load_scene_geometry(cfg: AppConfig, root: dict, wavelength: float) -> None:
    tx = root["tx_array"]
    tx_spacing = float(tx["spacing"])
    cfg.scene.tx_positions = [
        (0.0, float(itx) * tx_spacing * wavelength, 0.0) for itx in range(cfg.scene.num_tx)
    ]

    rx = root.get("rx_array")
    if not rx:
        raise ValueError("rx_array section is required for MIMO")
    az_elements = int(rx["az_elements"])
    el_elements = int(rx["el_elements"])
    rx_spacing = float(rx["spacing"])
    if az_elements < 2 or el_elements < 2:
        raise ValueError("MIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2")
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

    common = parse_common_run_config(root, config_path=path.resolve())
    cfg = AppConfig(**vars(common))

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
        raise ValueError("fmcw_mimo: mimo_coherent_compensation.multiplexing_scheme is required")
    cfg.scheme = parse_mimo_coherent_compensation(mcc)["multiplexing_scheme"]
    if cfg.scheme not in (MultiplexingScheme.TDM, MultiplexingScheme.DDM):
        raise ValueError("fmcw_mimo: multiplexing_scheme must be tdm or ddm")

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
        cfg.scene.waveform.chirp_repetition_interval = waveform.chirp_repetition_interval()
        cube = root["cube"]
        if "num_chirps" in cube:
            cfg.scene.num_chirps = int(cube["num_chirps"])
        else:
            cfg.scene.num_chirps = int(cube["chirps_per_tx"])
        if "ddm_code" not in mcc:
            raise ValueError("fmcw_mimo: mimo_coherent_compensation.ddm_code is required for DDM")
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
                raise ValueError("fmcw_mimo: non-linear DDM phase_codes require guard_bands == 0")
            num_chirps_codes = len(cfg.ddm_code.phase_codes) // cfg.ddm_code.num_tx
            if num_chirps_codes != cfg.scene.num_chirps:
                raise ValueError(
                    "fmcw_mimo: non-linear phase_codes length/num_tx must equal " "cube.num_chirps"
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
                raise ValueError("fmcw_mimo: DDM guard_bands leaves no Doppler bins per TX")
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
            raise ValueError("mimo_deinterleave.tx_schedule.num_tx must match tx_array.num_tx")
        cfg.report_num_doppler = cfg.scene.num_chirps

    cfg.tx_positions_flat = flatten_positions(cfg.scene.tx_positions)
    cfg.rx_positions_flat = flatten_positions(cfg.scene.rx_positions)

    for t in root.get("targets", []):
        position, velocity, amplitude = parse_target(t, app_name="fmcw_mimo")
        cfg.scene.targets.append(
            PointTarget(position=position, velocity=velocity, amplitude=amplitude)
        )

    parse_noise(root, cfg.scene)

    # Read-only mirror: az_angles/el_angles are BeamscanEstimatorOp's own
    # from_config() parameters; this only sizes the boresight-view axes.
    be = root.get("beamscan_estimator", {})
    cfg.az_lo, cfg.az_hi = grid_extent(
        [float(x) for x in be.get("az_angles", [])] or None, -0.35, 0.35
    )
    cfg.el_lo, cfg.el_hi = grid_extent(
        [float(x) for x in be.get("el_angles", [])] or None, -0.1, 0.1
    )

    cfg.mimo_coherent_compensation = parse_mimo_coherent_compensation(mcc)
    # Prefer the app-resolved descriptor (may expand generate_phase_codes).
    if cfg.scheme == MultiplexingScheme.DDM:
        cfg.mimo_coherent_compensation["ddm_code"] = cfg.ddm_code
    return cfg
