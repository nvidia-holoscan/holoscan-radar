# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_mvdr_estimator.cpp."""

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
from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import MVDREstimatorOpF32, MVDREstimatorOpF64  # noqa: E402

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for MVDREstimatorOp tests: {exc}")


def make_block(
    geom,
    conv,
    az: float,
    el: float,
    num_snapshots: int,
    noise: float,
    seed: int,
) -> np.ndarray:
    a = make_steering_snapshot(geom, conv, az, el)
    c = geom.num_elements()
    rng = np.random.default_rng(seed)
    x = np.empty((c, num_snapshots), dtype=np.complex128)
    for s in range(num_snapshots):
        gp = 2.0 * np.pi * float(s) / float(num_snapshots)
        g = complex(np.cos(gp), np.sin(gp))
        for ch in range(c):
            v = a[ch] * g
            v += complex(rng.normal(0, noise), rng.normal(0, noise))
            x[ch, s] = v
    return x


def make_block_two(
    geom,
    conv,
    az0: float,
    el0: float,
    az1: float,
    el1: float,
    num_snapshots: int,
    noise: float,
    seed: int,
) -> np.ndarray:
    a0 = make_steering_snapshot(geom, conv, az0, el0)
    a1 = make_steering_snapshot(geom, conv, az1, el1)
    c = geom.num_elements()
    rng = np.random.default_rng(seed)
    x = np.empty((c, num_snapshots), dtype=np.complex128)
    for s in range(num_snapshots):
        p0 = rng.uniform(0.0, 2.0 * np.pi)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        g0 = complex(np.cos(p0), np.sin(p0))
        g1 = complex(np.cos(p1), np.sin(p1))
        for ch in range(c):
            v = a0[ch] * g0 + a1[ch] * g1
            v += complex(rng.normal(0, noise), rng.normal(0, noise))
            x[ch, s] = v
    return x


def run_mvdr(snapshot: np.ndarray, *, op_cls=MVDREstimatorOpF64, **kwargs) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is MVDREstimatorOpF32 else np.complex128
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


def run_mvdr_full(
    snapshot: np.ndarray, *, op_cls=MVDREstimatorOpF64, **kwargs
) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is MVDREstimatorOpF32 else np.complex128
    snap = to_device_array(np.asarray(snapshot, dtype=dtype), dtype=dtype)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="snapshot",
        output_ports=[
            "aoa_estimates",
            "angle_spectrum",
            "doa_multi",
            "source_count",
        ],
    )
    return {
        "aoa": outs["aoa_estimates"][0],
        "spectrum": outs["angle_spectrum"][0],
        "doa_multi": outs["doa_multi"][0],
        "source_count": outs["source_count"][0],
    }


def test_RecoversKnownDoaUra():
    k_ny, k_nz, k_s = 4, 4, 32
    truth_az, truth_el = 0.12, 0.05
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.05, 7)
    snap = block.reshape(1, geom.num_elements(), k_s)
    out = run_mvdr(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 101),
        el_angles=linspace(-0.3, 0.3, 61),
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.02)


def test_RecoversKnownDoaUlaAzOnly():
    k_n, k_s = 8, 32
    truth_az = 0.08
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.05, 3)
    snap = block.reshape(1, k_n, k_s)
    out = run_mvdr(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
    )
    assert out["spectrum"].shape == (1, 61, 1)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_FloatInstantiationSmoke():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.06, 0.0, k_s, 0.05, 11)
    snap = block.reshape(1, k_n, k_s).astype(np.complex64)
    out = run_mvdr(
        snap,
        op_cls=MVDREstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(0.06, abs=0.03)


def test_RejectsInvalidMaxSources():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n)
    with pytest.raises(ValueError):
        run_mvdr(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            max_sources=0,
        )


def test_EmptyItemsProducesEmptyOutputs():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.empty((0, k_n, 4), dtype=np.complex128)
    out = run_mvdr(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=[-0.1, 0.0, 0.1],
        el_angles=[0.0],
    )
    assert out["aoa"].shape == (0, 2)
    assert out["spectrum"].shape == (0, 3, 1)


def test_FloatDoubleAgreeOnDoa():
    k_n, k_s = 8, 32
    truth_az = 0.09
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.05, 5)
    kwargs = dict(
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
    )
    out_f = run_mvdr(
        block.reshape(1, k_n, k_s).astype(np.complex64),
        op_cls=MVDREstimatorOpF32,
        **kwargs,
    )
    out_d = run_mvdr(block.reshape(1, k_n, k_s), **kwargs)
    assert out_f["aoa"][0, 0] == pytest.approx(float(out_d["aoa"][0, 0]), abs=1e-2)


def test_MultipleItemsRecoverIndependentDoa():
    k_ny, k_nz, k_s = 4, 4, 24
    az0, el0 = 0.11, 0.04
    az1, el1 = -0.07, -0.02
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    b0 = make_block(geom, conv, az0, el0, k_s, 0.05, 21)
    b1 = make_block(geom, conv, az1, el1, k_s, 0.05, 22)
    snap = np.stack([b0, b1], axis=0)
    out = run_mvdr(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.4, 0.4, 81),
        el_angles=linspace(-0.2, 0.2, 41),
    )
    assert out["aoa"].shape == (2, 2)
    assert out["aoa"][0, 0] == pytest.approx(az0, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(el0, abs=0.02)
    assert out["aoa"][1, 0] == pytest.approx(az1, abs=0.02)
    assert out["aoa"][1, 1] == pytest.approx(el1, abs=0.02)


def test_MultiSourceRecoversTwoDoa():
    k_n, k_s = 12, 128
    az0, az1 = 0.20, -0.15
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 71)
    out = run_mvdr_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 201),
        el_angles=[0.0],
        max_sources=2,
    )
    assert out["doa_multi"].shape == (1, 2, 2)
    assert out["source_count"].shape == (1,)
    assert int(out["source_count"][0]) == 2
    az_a = float(out["doa_multi"][0, 0, 0])
    az_b = float(out["doa_multi"][0, 1, 0])
    matched = (abs(az_a - az0) < 0.03 and abs(az_b - az1) < 0.03) or (
        abs(az_a - az1) < 0.03 and abs(az_b - az0) < 0.03
    )
    assert matched, f"az_a={az_a} az_b={az_b}"
    assert float(out["doa_multi"][0, 0, 0]) == pytest.approx(float(out["aoa"][0, 0]), abs=1e-9)
    assert float(out["doa_multi"][0, 0, 1]) == pytest.approx(float(out["aoa"][0, 1]), abs=1e-9)


def test_MaxSourcesCapacityNanFillsUnusedRows():
    k_n, k_s = 8, 64
    truth_az = 0.1
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.02, 81)
    out = run_mvdr_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
        max_sources=4,
    )
    assert out["doa_multi"].shape == (1, 4, 2)
    count = int(out["source_count"][0])
    assert 1 <= count <= 4
    assert float(out["doa_multi"][0, 0, 0]) == pytest.approx(truth_az, abs=0.02)
    for k in range(4):
        if k < count:
            assert not np.isnan(out["doa_multi"][0, k, 0])
        else:
            assert np.isnan(out["doa_multi"][0, k, 0])
            assert np.isnan(out["doa_multi"][0, k, 1])
