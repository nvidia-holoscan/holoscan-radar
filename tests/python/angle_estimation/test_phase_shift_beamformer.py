# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_phase_shift_beamformer.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.angle_helpers import linspace  # noqa: E402
from utils.angle_helpers import (
    make_planar_ura_geometry,
    make_steering_snapshot,
    make_ula_geometry,
    radar_boresight_convention,
)
from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import PhaseShiftBeamformerOpF32  # noqa: E402
from holoscan.radar import PhaseShiftBeamformerOpF64

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for PhaseShiftBeamformerOp tests: {exc}")


def run_beamformer(
    snapshot: np.ndarray, *, op_cls=PhaseShiftBeamformerOpF64, **kwargs
) -> np.ndarray:
    dtype = np.complex64 if op_cls is PhaseShiftBeamformerOpF32 else np.complex128
    snap = to_device_array(np.asarray(snapshot, dtype=dtype), dtype=dtype)
    frame = single_map(snap, "rd_maps")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator(
        make_op,
        frame,
        input_port="rd_maps",
        output_port="rda_cube",
    )
    assert len(outs) == 1
    return outs[0]


def peak_beam_at(cube: np.ndarray, num_az: int, num_el: int, d: int, r: int) -> tuple[int, int]:
    best = -1.0
    arg = (-1, -1)
    for ia in range(num_az):
        for ie in range(num_el):
            v = cube[ia, ie, d, r]
            p = float(v.real * v.real + v.imag * v.imag)
            if p > best:
                best = p
                arg = (ia, ie)
    return arg


def test_RecoversKnownBeamBin2D():
    k_ny, k_nz = 4, 4
    truth_az, truth_el = 0.12, 0.05
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    vec = make_steering_snapshot(geom, conv, truth_az, truth_el)
    rd = vec.reshape(geom.num_elements(), 1, 1)
    az = linspace(-0.5, 0.5, 101)
    el = linspace(-0.3, 0.3, 61)
    cube = run_beamformer(
        rd,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az,
        el_angles=el,
    )
    assert cube.shape == (101, 61, 1, 1)
    ia, ie = peak_beam_at(cube, 101, 61, 0, 0)
    assert az[ia] == pytest.approx(truth_az, abs=0.02)
    assert el[ie] == pytest.approx(truth_el, abs=0.02)


def test_RejectsEmptyScanGrid():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    rd = np.ones((k_n, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError):
        run_beamformer(
            rd,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[],
            el_angles=[0.0],
        )


def test_FloatDoubleAgreeOnPeakBeam():
    k_n = 8
    truth_az = 0.09
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    vec = make_steering_snapshot(geom, conv, truth_az, 0.0)
    rd = vec.reshape(k_n, 1, 1)
    az = linspace(-0.3, 0.3, 61)
    kwargs = dict(
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az,
        el_angles=[0.0],
    )
    cube_f = run_beamformer(rd.astype(np.complex64), op_cls=PhaseShiftBeamformerOpF32, **kwargs)
    cube_d = run_beamformer(rd, op_cls=PhaseShiftBeamformerOpF64, **kwargs)
    ia_f, ie_f = peak_beam_at(cube_f, 61, 1, 0, 0)
    ia_d, ie_d = peak_beam_at(cube_d, 61, 1, 0, 0)
    assert ia_f == ia_d
    assert ie_f == ie_d


def test_PreservesRangeDopplerPlanePerCell():
    k_n, k_d, k_r = 8, 2, 2
    az0, az1 = 0.1, -0.15
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    v0 = make_steering_snapshot(geom, conv, az0, 0.0)
    v1 = make_steering_snapshot(geom, conv, az1, 0.0)
    rd = np.zeros((k_n, k_d, k_r), dtype=np.complex128)
    for c in range(k_n):
        rd[c, 0, 0] = v0[c]
        rd[c, 1, 1] = v1[c]
    az = linspace(-0.4, 0.4, 81)
    cube = run_beamformer(
        rd,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az,
        el_angles=[0.0],
    )
    assert cube.shape == (81, 1, k_d, k_r)
    ia0, _ = peak_beam_at(cube, 81, 1, 0, 0)
    ia1, _ = peak_beam_at(cube, 81, 1, 1, 1)
    assert az[ia0] == pytest.approx(az0, abs=0.02)
    assert az[ia1] == pytest.approx(az1, abs=0.02)
    empty = 0.0
    for ia in range(81):
        v = cube[ia, 0, 0, 1]
        empty = max(empty, float(v.real * v.real + v.imag * v.imag))
    assert empty == pytest.approx(0.0, abs=1e-9)


def test_BatchLeadingDimsRecoverIndependentDoa():
    k_n = 8
    az0, az1 = 0.12, -0.1
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    v0 = make_steering_snapshot(geom, conv, az0, 0.0)
    v1 = make_steering_snapshot(geom, conv, az1, 0.0)
    rd = np.stack([v0.reshape(k_n, 1, 1), v1.reshape(k_n, 1, 1)], axis=0)
    az = linspace(-0.4, 0.4, 81)
    cube = run_beamformer(
        rd,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az,
        el_angles=[0.0],
    )
    assert cube.shape == (2, 81, 1, 1, 1)

    def peak_in_slice(b: int) -> int:
        best = -1.0
        arg = -1
        for ia in range(81):
            v = cube[b, ia, 0, 0, 0]
            p = float(v.real * v.real + v.imag * v.imag)
            if p > best:
                best = p
                arg = ia
        return arg

    assert az[peak_in_slice(0)] == pytest.approx(az0, abs=0.02)
    assert az[peak_in_slice(1)] == pytest.approx(az1, abs=0.02)


def test_ConventionRelabelsSamePhysicalDirection():
    from holoscan.radar import AngleConvention, AngleConventionPreset

    k_ny, k_nz = 4, 4
    truth_az, truth_el = -0.1, -0.03
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    az = linspace(-0.5, 0.5, 101)
    el = linspace(-0.3, 0.3, 61)
    for preset in (
        AngleConventionPreset.RADAR_BORESIGHT,
        AngleConventionPreset.AEROSPACE_NED,
    ):
        conv = AngleConvention()
        conv.preset = preset
        vec = make_steering_snapshot(geom, conv, truth_az, truth_el)
        cube = run_beamformer(
            vec.reshape(geom.num_elements(), 1, 1),
            array_geometry=geom,
            angle_convention=conv,
            az_angles=az,
            el_angles=el,
        )
        ia, ie = peak_beam_at(cube, 101, 61, 0, 0)
        assert az[ia] == pytest.approx(truth_az, abs=0.02)
        assert el[ie] == pytest.approx(truth_el, abs=0.02)
