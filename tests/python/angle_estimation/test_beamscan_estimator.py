# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_beamscan_estimator.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.angle_helpers import aerospace_ned_convention  # noqa: E402
from utils.angle_helpers import (
    linspace,
    make_planar_ura_geometry,
    make_steering_snapshot,
    make_ula_geometry,
    physics_spherical_convention,
    radar_boresight_convention,
)
from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import BeamscanEstimatorOpF32  # noqa: E402
from holoscan.radar import BeamscanEstimatorOpF64, PeakInterpolation

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for BeamscanEstimatorOp tests: {exc}")


def run_beamscan(
    snapshot: np.ndarray,
    *,
    op_cls=BeamscanEstimatorOpF64,
    **kwargs,
) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is BeamscanEstimatorOpF32 else np.complex128
    snap = to_device_array(np.asarray(snapshot, dtype=dtype), dtype=dtype)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="snapshot",
        output_ports=["aoa_estimates", "angle_spectrum"],
    )
    assert len(outs["aoa_estimates"]) == 1
    return {"aoa": outs["aoa_estimates"][0], "spectrum": outs["angle_spectrum"][0]}


def test_RecoversKnownDoa2D():
    k_ny, k_nz = 4, 4
    truth_az, truth_el = 0.12, 0.05
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    snap_host = make_steering_snapshot(geom, conv, truth_az, truth_el)
    snap = snap_host.reshape(1, geom.num_elements())

    out = run_beamscan(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 101),
        el_angles=linspace(-0.3, 0.3, 61),
        peak_interpolation=PeakInterpolation.PARABOLIC,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.01)


def test_FloatInstantiationSmoke():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap_host = make_steering_snapshot(geom, conv, 0.05, 0.0)
    out = run_beamscan(
        snap_host.reshape(1, k_n),
        op_cls=BeamscanEstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.2, 0.2, 21),
        el_angles=[0.0],
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(0.05, abs=0.02)


def test_RejectsEmptyScanGrid():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.ones((1, k_n), dtype=np.complex128)
    with pytest.raises(ValueError):
        run_beamscan(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[],
            el_angles=[0.0],
        )


def test_SpectrumShapeMatchesScanGrid():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap0 = make_steering_snapshot(geom, conv, 0.05, 0.0)
    snap = np.stack([snap0, snap0], axis=0)
    out = run_beamscan(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.2, 0.2, 17),
        el_angles=linspace(-0.1, 0.1, 9),
    )
    assert out["aoa"].shape == (2, 2)
    assert out["spectrum"].shape == (2, 17, 9)
    assert float(np.max(out["spectrum"])) > 0.0


def test_EmptyItemsProducesEmptyOutputs():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.empty((0, k_n), dtype=np.complex128)
    out = run_beamscan(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=[-0.1, 0.0, 0.1],
        el_angles=[0.0],
    )
    assert out["aoa"].shape == (0, 2)
    assert out["spectrum"].shape == (0, 3, 1)


def test_RecoversKnownDoaPhysicsSpherical():
    k_ny, k_nz = 4, 4
    truth_az = 0.1
    horizon_el = 0.04
    polar_el = np.pi / 2.0 - horizon_el
    conv = physics_spherical_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    snap = make_steering_snapshot(geom, conv, truth_az, polar_el).reshape(1, geom.num_elements())
    out = run_beamscan(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 101),
        el_angles=linspace(1.2, 1.7, 61),
        peak_interpolation=PeakInterpolation.PARABOLIC,
    )
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(polar_el, abs=0.02)


def test_RecoversKnownDoaAerospaceNed():
    k_ny, k_nz = 4, 4
    truth_az, truth_el = -0.1, -0.03
    conv = aerospace_ned_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    snap = make_steering_snapshot(geom, conv, truth_az, truth_el).reshape(1, geom.num_elements())
    out = run_beamscan(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 101),
        el_angles=linspace(-0.3, 0.3, 61),
        peak_interpolation=PeakInterpolation.PARABOLIC,
    )
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.01)


def test_FloatDoubleAgreeOnDoa():
    k_n = 8
    truth_az = 0.09
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap_host = make_steering_snapshot(geom, conv, truth_az, 0.0)
    kwargs = dict(
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 51),
        el_angles=[0.0],
        peak_interpolation=PeakInterpolation.PARABOLIC,
    )
    out_f = run_beamscan(
        snap_host.reshape(1, k_n).astype(np.complex64),
        op_cls=BeamscanEstimatorOpF32,
        **kwargs,
    )
    out_d = run_beamscan(snap_host.reshape(1, k_n), **kwargs)
    assert out_f["aoa"][0, 0] == pytest.approx(float(out_d["aoa"][0, 0]), abs=2e-3)
    assert out_f["aoa"][0, 1] == pytest.approx(float(out_d["aoa"][0, 1]), abs=2e-3)


def test_ParabolicRefinesOffGridDoa():
    k_n = 8
    truth_az = 0.17
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = make_steering_snapshot(geom, conv, truth_az, 0.0).reshape(1, k_n)
    kwargs = dict(
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.2, 0.2, 11),
        el_angles=linspace(-0.1, 0.1, 3),
    )
    out_none = run_beamscan(snap, peak_interpolation=PeakInterpolation.NONE, **kwargs)
    out_para = run_beamscan(snap, peak_interpolation=PeakInterpolation.PARABOLIC, **kwargs)
    assert out_none["aoa"][0, 0] == pytest.approx(0.16, abs=1e-6)
    assert out_para["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert abs(out_para["aoa"][0, 0] - out_none["aoa"][0, 0]) > 5e-3
