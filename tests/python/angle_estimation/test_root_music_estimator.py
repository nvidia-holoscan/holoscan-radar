# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_root_music_estimator.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.angle_helpers import (  # noqa: E402
    direction_cosine_convention,
    make_planar_ura_geometry,
    make_steering_snapshot,
    make_ula_geometry,
    radar_boresight_convention,
)
from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    ElevationReference,
    RootMUSICEstimatorOpF32,
    RootMUSICEstimatorOpF64,
    SignedAxis,
)

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for RootMUSICEstimatorOp tests: {exc}")


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


def make_block_two_coherent(
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
        ps = rng.uniform(0.0, 2.0 * np.pi)
        g = complex(np.cos(ps), np.sin(ps))
        for ch in range(c):
            v = (a0[ch] + a1[ch]) * g
            v += complex(rng.normal(0, noise), rng.normal(0, noise))
            x[ch, s] = v
    return x


def run_root(
    snapshot: np.ndarray, *, op_cls=RootMUSICEstimatorOpF64, **kwargs
) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is RootMUSICEstimatorOpF32 else np.complex128
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
            "doa_multi",
            "source_count",
            "eigenvalues",
        ],
    )
    assert len(outs["aoa_estimates"]) == 1
    return {
        "aoa": outs["aoa_estimates"][0],
        "doa_multi": outs["doa_multi"][0],
        "source_count": outs["source_count"][0],
        "eigenvalues": outs["eigenvalues"][0],
    }


def test_RecoversKnownDoaUlaFixedOneSource():
    k_n, k_s = 8, 32
    truth_az = 0.08
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.02, 3)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_CustomConventionMapsSignedAxes():
    geom = make_ula_geometry(8, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.boresight_axis = SignedAxis.PLUS_X
    custom.azimuth_positive_axis = SignedAxis.MINUS_Y
    custom.elevation_positive_axis = SignedAxis.MINUS_Z
    block = make_block(geom, radar, 0.18, 0.0, 64, 0.01, 176)

    out = run_root(
        block[None],
        array_geometry=geom,
        angle_convention=custom,
        num_sources=1,
    )

    assert out["aoa"][0, 0] == pytest.approx(-0.18, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_CustomConventionSupportsZenithReference():
    geom = make_ula_geometry(8, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.elevation_reference = ElevationReference.ZENITH
    block = make_block(geom, radar, 0.1, 0.0, 64, 0.01, 177)

    out = run_root(
        block[None],
        array_geometry=geom,
        angle_convention=custom,
        num_sources=1,
    )

    assert out["aoa"][0, 0] == pytest.approx(0.1, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(np.pi / 2.0, abs=1e-9)


def test_RejectsDegenerateCustomConvention():
    geom = make_ula_geometry(4, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.azimuth_positive_axis = custom.boresight_axis
    block = make_block(geom, radar, 0.1, 0.0, 8, 0.01, 178)

    with pytest.raises(ValueError, match="three distinct principal axes"):
        run_root(
            block[None],
            array_geometry=geom,
            angle_convention=custom,
            num_sources=1,
        )


def test_RecoversTwoSources():
    k_n, k_s = 12, 256
    az0, az1 = 0.20, -0.15
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 71)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert out["doa_multi"].shape == (1, 2, 2)
    assert int(out["source_count"][0]) == 2
    az_a = float(out["doa_multi"][0, 0, 0])
    az_b = float(out["doa_multi"][0, 1, 0])
    matched = (abs(az_a - az0) < 0.03 and abs(az_b - az1) < 0.03) or (
        abs(az_a - az1) < 0.03 and abs(az_b - az0) < 0.03
    )
    assert matched, f"az_a={az_a} az_b={az_b}"
    assert float(out["doa_multi"][0, 0, 0]) == pytest.approx(float(out["aoa"][0, 0]), abs=1e-9)


def test_ResolvesCloselySpacedSources():
    k_n, k_s = 16, 512
    az0, az1 = -0.035, 0.035
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.005, 73)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert int(out["source_count"][0]) == 2
    actual = sorted(float(value) for value in out["doa_multi"][0, :, 0])
    assert actual == pytest.approx([az0, az1], abs=0.01)


def test_MaxSourcesCapsReportedRoots():
    k_n, k_s = 12, 256
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, -0.15, 0.0, 0.20, 0.0, k_s, 0.02, 78)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=1,
    )
    assert out["doa_multi"].shape == (1, 1, 2)
    assert int(out["source_count"][0]) == 1
    assert np.isfinite(out["doa_multi"][0, 0, 0])


def test_HighSnrLargeArrayRootRemainsFinite():
    k_n, k_s = 32, 128
    truth_az = 0.17
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 1e-6, 74)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert int(out["source_count"][0]) == 1
    assert np.isfinite(out["aoa"][0, 0])
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=1e-4)


def test_AutoNumSourcesRecoversOneSource():
    k_n, k_s = 8, 64
    truth_az = 0.1
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.02, 9)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        max_sources=4,
    )
    assert int(out["source_count"][0]) == 1
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)


def test_RecoversUraSingleSourceAzEl():
    k_ny, k_nz, k_s = 6, 6, 256
    truth_az, truth_el = 0.15, -0.10
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.02, 7)
    out = run_root(
        block.reshape(1, geom.num_elements(), k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.03)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.03)


def test_RecoversUraTwoSourcesAzElPairing():
    k_ny, k_nz, k_s = 8, 8, 512
    # Distinct in BOTH az and el so a mispaired result is detectable.
    az0, el0 = 0.20, 0.12
    az1, el1 = -0.15, -0.10
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, el0, az1, el1, k_s, 0.01, 79)
    out = run_root(
        block.reshape(1, geom.num_elements(), k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert out["doa_multi"].shape == (1, 2, 2)
    assert out["source_count"][0] == 2

    dm = out["doa_multi"][0]

    def near(pair, taz, tel):
        return abs(pair[0] - taz) < 0.04 and abs(pair[1] - tel) < 0.04

    matched = (near(dm[0], az0, el0) and near(dm[1], az1, el1)) or (
        near(dm[0], az1, el1) and near(dm[1], az0, el0)
    )
    assert matched, f"pair0={dm[0]} pair1={dm[1]}"


def test_RecoversUraSingleSourceAzElFloat():
    k_ny, k_nz, k_s = 6, 6, 256
    truth_az, truth_el = -0.12, 0.08
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.02, 17)
    out = run_root(
        block.reshape(1, geom.num_elements(), k_s),
        op_cls=RootMUSICEstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.03)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.03)


def test_RejectsNonUniformLinearGeometry():
    k_n, k_s = 5, 16
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    positions = list(geom.element_positions)
    positions[3 * 3 + 1] = 1.65
    geom.element_positions = positions
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 75)
    with pytest.raises(ValueError):
        run_root(
            block.reshape(1, k_n, k_s),
            array_geometry=geom,
            angle_convention=conv,
        )


def test_RejectsUlaParallelToBoresight():
    k_n = 4
    geom = ArrayGeometry()
    geom.wavelength = 1.0
    geom.geometry_type = ArrayGeometryType.ULA
    geom.element_positions = [
        coordinate for channel in range(k_n) for coordinate in (0.5 * float(channel), 0.0, 0.0)
    ]
    conv = radar_boresight_convention()
    snapshot = np.ones((1, k_n, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        run_root(
            snapshot,
            array_geometry=geom,
            angle_convention=conv,
        )


def test_EigenvaluesAscendingAndNonNegative():
    k_n, k_s = 8, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 91)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["eigenvalues"].shape == (1, k_n)
    ev = out["eigenvalues"][0]
    assert np.all(ev >= -1e-6)
    assert np.all(np.diff(ev) >= -1e-9)
    assert ev[-1] > 10.0 * ev[-2]


def test_RankDeficientAutoFallsBackToOneSource():
    k_n, k_s = 8, 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 12)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        max_sources=4,
    )
    assert int(out["source_count"][0]) == 1


def test_RejectsNumSourcesTooLarge():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n, 1)
    with pytest.raises(ValueError):
        run_root(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=k_n,
        )


def test_RejectsInvalidMaxSources():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n, 1)
    with pytest.raises(ValueError):
        run_root(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            max_sources=0,
        )


def test_RejectsInvalidSourceCountPfa():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n, 1)
    with pytest.raises(ValueError):
        run_root(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            source_count_pfa=0.0,
        )


def test_EmptyItemsProducesEmptyOutputs():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    out = run_root(
        np.empty((0, k_n, 4), dtype=np.complex128),
        array_geometry=geom,
        angle_convention=conv,
        max_sources=3,
    )
    assert out["aoa"].shape == (0, 2)
    assert out["doa_multi"].shape == (0, 3, 2)
    assert out["source_count"].shape == (0,)
    assert out["eigenvalues"].shape == (0, k_n)


def test_RejectsChannelMismatch():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.ones((1, k_n - 1, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        run_root(
            snap,
            array_geometry=geom,
            angle_convention=conv,
        )


def test_RejectsSnapshotDtypeMismatch():
    # run_root() always casts the input to match op_cls, so a genuine mismatch
    # (complex64 data into an F64 op) needs the lower-level harness directly.
    k_n = 4
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    conv = radar_boresight_convention()
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap_f = to_device_array(a.reshape(1, k_n, 1), dtype=np.complex64)

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator_multi(
            make_op,
            single_map(snap_f, "snapshot"),
            input_port="snapshot",
            output_ports=["aoa_estimates"],
            drain_ports=["doa_multi", "source_count", "eigenvalues"],
        )


def test_RejectsSnapshotRank2():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.05, 0.0)
    snap = a.reshape(1, k_n)
    with pytest.raises(ValueError):
        run_root(
            snap,
            array_geometry=geom,
            angle_convention=conv,
        )


def test_FloatInstantiationSmoke():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.06, 0.0, k_s, 0.02, 11)
    snap = block.reshape(1, k_n, k_s).astype(np.complex64)
    out = run_root(
        snap,
        op_cls=RootMUSICEstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(0.06, abs=0.03)


def test_FloatDoubleAgreeOnDoa():
    k_n, k_s = 8, 32
    truth_az = 0.09
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.05, 5)
    kwargs = {
        "array_geometry": geom,
        "angle_convention": conv,
        "num_sources": 1,
    }
    out_f = run_root(
        block.reshape(1, k_n, k_s).astype(np.complex64),
        op_cls=RootMUSICEstimatorOpF32,
        **kwargs,
    )
    out_d = run_root(block.reshape(1, k_n, k_s), **kwargs)
    assert out_f["aoa"][0, 0] == pytest.approx(float(out_d["aoa"][0, 0]), abs=1e-2)


def test_DeterministicAcrossRuns():
    k_n, k_s = 8, 64
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.09, 0.0, k_s, 0.02, 171)
    kwargs = {
        "array_geometry": geom,
        "angle_convention": conv,
        "num_sources": 1,
        "max_sources": 2,
    }
    first = run_root(block.reshape(1, k_n, k_s), **kwargs)
    second = run_root(block.reshape(1, k_n, k_s), **kwargs)
    for port in first:
        np.testing.assert_array_equal(first[port], second[port])


def test_DirectionCosineConventionMapsUlaProjection():
    k_n, k_s = 8, 64
    truth_u = 0.23
    conv = direction_cosine_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_u, 0.0, k_s, 0.01, 76)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"][0, 0] == pytest.approx(truth_u, abs=0.01)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_DirectionCosineNearEndfirePreservesSign():
    k_n, k_s = 16, 128
    u0, u1 = -0.92, 0.92
    conv = direction_cosine_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    b0 = make_block(geom, conv, u0, 0.0, k_s, 0.005, 79)
    b1 = make_block(geom, conv, u1, 0.0, k_s, 0.005, 80)
    out = run_root(
        np.stack([b0, b1], axis=0),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"][0, 0] == pytest.approx(u0, abs=0.01)
    assert out["aoa"][1, 0] == pytest.approx(u1, abs=0.01)


def test_UnusedDoaRowsAreNan():
    k_n, k_s = 8, 64
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.01, 77)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
        max_sources=3,
    )
    assert int(out["source_count"][0]) == 1
    assert np.all(np.isfinite(out["doa_multi"][0, 0]))
    assert np.all(np.isnan(out["doa_multi"][0, 1:]))


def test_IsValidMaskExcludesWrongSnapshots():
    k_n, k_s = 8, 32
    truth_az, wrong_az = 0.1, -0.2
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)

    good = make_block(geom, conv, truth_az, 0.0, k_s // 2, 0.02, 31)
    bad = make_block(geom, conv, wrong_az, 0.0, k_s // 2, 0.02, 32)
    block = np.empty((k_n, k_s), dtype=np.complex128)
    block[:, : k_s // 2] = good
    block[:, k_s // 2 :] = bad
    mask = np.zeros((1, k_s), dtype=np.uint8)
    mask[0, : k_s // 2] = 1

    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    mask_dev = to_device_array(mask, dtype=np.uint8)

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            max_sources=2,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        single_map(snap, "snapshot"),
        input_port="snapshot",
        output_ports=["aoa_estimates"],
        extra_inputs={"is_valid": [single_map(mask_dev, "is_valid")]},
        drain_ports=["doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["aoa_estimates"]) == 1
    assert outs["aoa_estimates"][0][0, 0] == pytest.approx(truth_az, abs=0.03)


def test_AllInvalidMaskBlanksDoa():
    k_n, k_s = 8, 16
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 33)
    mask = np.zeros((1, k_s), dtype=np.uint8)

    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    mask_dev = to_device_array(mask, dtype=np.uint8)

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            max_sources=2,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        single_map(snap, "snapshot"),
        input_port="snapshot",
        output_ports=["aoa_estimates", "doa_multi", "source_count"],
        extra_inputs={"is_valid": [single_map(mask_dev, "is_valid")]},
        drain_ports=["eigenvalues"],
    )

    aoa = outs["aoa_estimates"][0]
    assert np.isnan(aoa[0, 0])
    assert np.isnan(aoa[0, 1])
    assert int(outs["source_count"][0][0]) == 0
    assert np.all(np.isnan(outs["doa_multi"][0]))


def test_LiveNumSourcesOverrideAppliesPerTick():
    k_n, k_s = 12, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, k_s, 0.02, 111)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            max_sources=4,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["source_count"],
        overrides={"num_sources": [1, 2]},
        drain_ports=["aoa_estimates", "doa_multi", "eigenvalues"],
    )
    assert len(outs["source_count"]) == 2
    assert int(outs["source_count"][0][0]) == 1
    assert int(outs["source_count"][1][0]) == 2


def test_LiveArrayGeometryOverrideAppliesPerTick():
    k_n, k_s = 4, 16
    conv = radar_boresight_convention()
    geom_a = make_ula_geometry(k_n, 0.5, 1.0)
    geom_b = make_ula_geometry(k_n, 0.5, 1.0)
    geom_b.wavelength = 0.8
    block = make_block(geom_a, conv, 0.05, 0.0, k_s, 0.05, 61)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom_a,
            angle_convention=conv,
            num_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["aoa_estimates"],
        overrides={"array_geometry": [geom_a, geom_b, geom_a]},
        drain_ports=["doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["aoa_estimates"]) == 3
    assert outs["aoa_estimates"][0][0, 0] != pytest.approx(
        float(outs["aoa_estimates"][1][0, 0]), abs=1e-12
    )
    np.testing.assert_array_equal(outs["aoa_estimates"][0], outs["aoa_estimates"][2])


def test_LiveSpatialSmoothingOverrideAppliesPerTick():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 141)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            spatial_smoothing=[],
            num_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["eigenvalues"],
        overrides={"spatial_smoothing": [[], [2]]},
        drain_ports=["aoa_estimates", "doa_multi", "source_count"],
    )
    assert len(outs["eigenvalues"]) == 3
    assert outs["eigenvalues"][0].shape == (1, k_n)
    assert outs["eigenvalues"][1].shape == (1, k_n - 1)
    # Tick 2 reverts to the Params smoothing (off), as on tick 0.
    assert outs["eigenvalues"][2].shape == (1, k_n)


def test_LiveSourceCountPfaOverrideAppliesPerTick():
    k_n, k_s = 8, 40
    truth_az = 0.08
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.45, 191)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=-1,
            source_count_pfa=1e-4,
            max_sources=4,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["source_count"],
        overrides={"source_count_pfa": [1e-4, 0.5]},
        drain_ports=["aoa_estimates", "doa_multi", "eigenvalues"],
    )
    assert len(outs["source_count"]) == 3
    tight = int(outs["source_count"][0][0])
    loose = int(outs["source_count"][1][0])
    assert tight <= loose
    # Tick 2 reverts to the Params Pfa of 1e-4, as on tick 0.
    assert int(outs["source_count"][2][0]) == tight


def test_LiveMaxSourcesOverrideAppliesPerTick():
    k_n, k_s = 12, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, k_s, 0.02, 101)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return RootMUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=2,
            max_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["doa_multi"],
        overrides={"max_sources": [1, 3]},
        drain_ports=["aoa_estimates", "source_count", "eigenvalues"],
    )
    assert len(outs["doa_multi"]) == 2
    assert outs["doa_multi"][0].shape == (1, 1, 2)
    assert outs["doa_multi"][1].shape == (1, 3, 2)


def test_MultipleItemsRecoverIndependentDoa():
    k_n, k_s = 8, 48
    az0, az1 = 0.11, -0.07
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    b0 = make_block(geom, conv, az0, 0.0, k_s, 0.02, 21)
    b1 = make_block(geom, conv, az1, 0.0, k_s, 0.02, 22)
    snap = np.stack([b0, b1], axis=0)
    out = run_root(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (2, 2)
    assert out["aoa"][0, 0] == pytest.approx(az0, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)
    assert out["aoa"][1, 0] == pytest.approx(az1, abs=0.02)
    assert out["aoa"][1, 1] == pytest.approx(0.0, abs=1e-9)


def test_SpatialSmoothing1DRecoversDoa():
    k_n, k_s = 10, 64
    az0, az1 = 0.30, -0.30
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two_coherent(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 70)
    out = run_root(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        spatial_smoothing=[4],
        forward_backward=True,
        num_sources=2,
        max_sources=2,
    )
    assert out["eigenvalues"].shape == (1, 7)
    assert int(out["source_count"][0]) == 2
    a = float(out["doa_multi"][0, 0, 0])
    b = float(out["doa_multi"][0, 1, 0])
    matched = (abs(a - az0) < 0.04 and abs(b - az1) < 0.04) or (
        abs(a - az1) < 0.04 and abs(b - az0) < 0.04
    )
    assert matched, f"a={a} b={b}"
