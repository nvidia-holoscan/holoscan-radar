# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_music_estimator.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.angle_helpers import linspace  # noqa: E402
from utils.angle_helpers import (
    direction_cosine_convention,
    make_planar_ura_geometry,
    make_steering_snapshot,
    make_ula_geometry,
    radar_boresight_convention,
)
from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    BeamscanEstimatorOpF64,
    MUSICEstimatorOpF32,
    MUSICEstimatorOpF64,
)

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for MUSICEstimatorOp tests: {exc}")


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
        g = complex(np.cos(ps), np.sin(ps))  # shared gain -> coherent pair
        for ch in range(c):
            v = (a0[ch] + a1[ch]) * g
            v += complex(rng.normal(0, noise), rng.normal(0, noise))
            x[ch, s] = v
    return x


def run_music(
    snapshot: np.ndarray, *, op_cls=MUSICEstimatorOpF64, **kwargs
) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is MUSICEstimatorOpF32 else np.complex128
    snap = to_device_array(np.asarray(snapshot, dtype=dtype), dtype=dtype)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="snapshot",
        output_ports=["aoa_estimates", "angle_spectrum"],
        drain_ports=["doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["aoa_estimates"]) == 1
    return {"aoa": outs["aoa_estimates"][0], "spectrum": outs["angle_spectrum"][0]}


def run_music_full(
    snapshot: np.ndarray, *, op_cls=MUSICEstimatorOpF64, **kwargs
) -> dict[str, np.ndarray]:
    dtype = np.complex64 if op_cls is MUSICEstimatorOpF32 else np.complex128
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
            "eigenvalues",
        ],
    )
    return {
        "aoa": outs["aoa_estimates"][0],
        "spectrum": outs["angle_spectrum"][0],
        "doa_multi": outs["doa_multi"][0],
        "source_count": outs["source_count"][0],
        "eigenvalues": outs["eigenvalues"][0],
    }


def test_RecoversKnownDoaUlaAzOnlyFixedOneSource():
    k_n, k_s = 8, 32
    truth_az = 0.08
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.02, 3)
    snap = block.reshape(1, k_n, k_s)
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
        num_sources=1,
    )
    assert out["spectrum"].shape == (1, 61, 1)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_SharedSparseSlotInterchangesWithBeamscan():
    k_n = 8
    truth_az = 0.08
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    look = make_steering_snapshot(geom, conv, truth_az, 0.0)
    snapshot = look.reshape(1, k_n, 1)
    frame = single_map(to_device_array(snapshot, dtype=np.complex128), "snapshot")
    kwargs = {
        "array_geometry": geom,
        "angle_convention": conv,
        "az_angles": linspace(-0.3, 0.3, 61),
        "el_angles": [0.0],
    }

    def make_beamscan(app):
        return BeamscanEstimatorOpF64(app, name="op_under_test", **kwargs)

    beamscan = run_operator_multi(
        make_beamscan,
        frame,
        input_port="snapshot",
        output_ports=["aoa_estimates"],
        drain_ports=["angle_spectrum"],
    )
    music = run_music(snapshot, num_sources=1, **kwargs)

    assert beamscan["aoa_estimates"][0][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert music["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)


def test_RecoversKnownDoaUraFixedOneSource():
    k_ny, k_nz, k_s = 4, 4, 48
    truth_az, truth_el = 0.12, 0.05
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.02, 7)
    snap = block.reshape(1, geom.num_elements(), k_s)
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 101),
        el_angles=linspace(-0.3, 0.3, 61),
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.02)


def test_AutoNumSourcesRecoversOneSource():
    k_n, k_s = 8, 64
    truth_az = 0.1
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, 0.0, k_s, 0.02, 9)
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
        max_sources=4,
    )
    assert int(out["source_count"][0]) == 1
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.02)


def test_MultiSourceRecoversTwoDoa():
    k_n, k_s = 12, 256
    az0, az1 = 0.20, -0.15
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 71)
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 201),
        el_angles=[0.0],
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
    assert float(out["doa_multi"][0, 0, 1]) == pytest.approx(float(out["aoa"][0, 1]), abs=1e-9)


def test_MaxSourcesCapsBelowNumSourcesNanFillsRest():
    k_n, k_s = 12, 256
    az0, az1 = 0.20, -0.15
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 72)
    # num_sources = 2 (structural), but max_sources = 1 caps the report.
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.5, 0.5, 201),
        el_angles=[0.0],
        num_sources=2,
        max_sources=1,
    )
    assert out["doa_multi"].shape == (1, 1, 2)
    assert int(out["source_count"][0]) == 1


def test_SpatialSmoothingResolvesCoherentSourcesUla():
    k_n, k_s = 10, 64
    az0, az1 = 0.30, -0.30
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two_coherent(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.02, 70)
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.6, 0.6, 241),
        el_angles=[0.0],
        spatial_smoothing=[4],
        forward_backward=True,
        num_sources=2,
        max_sources=2,
    )
    # Smoothing reduces the estimation aperture to L = 7 channels.
    assert out["eigenvalues"].shape == (1, 7)
    assert int(out["source_count"][0]) == 2
    a = float(out["doa_multi"][0, 0, 0])
    b = float(out["doa_multi"][0, 1, 0])
    matched = (abs(a - az0) < 0.04 and abs(b - az1) < 0.04) or (
        abs(a - az1) < 0.04 and abs(b - az0) < 0.04
    )
    assert matched, f"a={a} b={b}"


def test_EigenvaluesAscendingAndNonNegative():
    k_n, k_s = 8, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 91)
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 31),
        el_angles=[0.0],
        num_sources=1,
    )
    assert out["eigenvalues"].shape == (1, k_n)
    ev = out["eigenvalues"][0]
    assert np.all(ev >= -1e-6)
    assert np.all(np.diff(ev) >= -1e-9)
    assert ev[-1] > 10.0 * ev[-2]


def test_RankDeficientAutoFallsBackToOneSource():
    # K = 4 valid snapshots < num_channels = 8: covariance is rank-deficient,
    # so num_sources: auto must fall back to 1 rather than run EFT.
    k_n, k_s = 8, 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 12)
    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 31),
        el_angles=[0.0],
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
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            num_sources=k_n,
        )


def test_RejectsInvalidMaxSources():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n, 1)
    with pytest.raises(ValueError):
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            max_sources=0,
        )


def test_RejectsInvalidSourceCountPfa():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, k_n, 1)
    with pytest.raises(ValueError):
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            source_count_pfa=0.0,
        )


def test_EmptyItemsProducesEmptyOutputs():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.empty((0, k_n, 4), dtype=np.complex128)
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=[-0.1, 0.0, 0.1],
        el_angles=[0.0],
    )
    assert out["aoa"].shape == (0, 2)
    assert out["spectrum"].shape == (0, 3, 1)


def test_FloatInstantiationSmoke():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.06, 0.0, k_s, 0.02, 11)
    snap = block.reshape(1, k_n, k_s).astype(np.complex64)
    out = run_music(
        snap,
        op_cls=MUSICEstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
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
        "az_angles": linspace(-0.3, 0.3, 61),
        "el_angles": [0.0],
        "num_sources": 1,
    }
    out_f = run_music(
        block.reshape(1, k_n, k_s).astype(np.complex64),
        op_cls=MUSICEstimatorOpF32,
        **kwargs,
    )
    out_d = run_music(block.reshape(1, k_n, k_s), **kwargs)
    assert out_f["aoa"][0, 0] == pytest.approx(float(out_d["aoa"][0, 0]), abs=1e-2)


def test_DeterministicAcrossRuns():
    k_n, k_s = 8, 64
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.09, 0.0, k_s, 0.02, 171)
    kwargs = {
        "array_geometry": geom,
        "angle_convention": conv,
        "az_angles": linspace(-0.3, 0.3, 61),
        "el_angles": [0.0],
        "num_sources": 1,
        "max_sources": 2,
    }
    first = run_music_full(block.reshape(1, k_n, k_s), **kwargs)
    second = run_music_full(block.reshape(1, k_n, k_s), **kwargs)
    for port in first:
        np.testing.assert_array_equal(first[port], second[port])


def test_LiveNumSourcesStickyOverride():
    k_n, k_s = 12, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, k_s, 0.02, 111)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            name="op_under_test",
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.5, 0.5, 101),
            el_angles=[0.0],
            num_sources=1,
            max_sources=4,
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["source_count"],
        overrides={"num_sources": [1, 2]},
        drain_ports=["aoa_estimates", "angle_spectrum", "doa_multi", "eigenvalues"],
    )
    assert len(outs["source_count"]) == 2
    assert int(outs["source_count"][0][0]) == 1
    assert int(outs["source_count"][1][0]) == 2


# --- TESTING.md-mandated super-resolution comparison ------------------------


def count_prominent_local_maxima_1d(spec: np.ndarray, rel_threshold: float = 0.3) -> int:
    """Prominent strict local maxima - mirrors the C++ test's helper.

    A finite-aperture beampattern's ever-present sidelobe ripple is
    technically a sequence of local maxima too, but is not a "resolved
    source"; counting every ripple (no threshold) over-counts.
    """
    global_max = float(np.max(spec)) if spec.size else 0.0
    if not (global_max > 0.0):
        return 0
    floor = rel_threshold * global_max
    count = 0
    n = spec.shape[0]
    for i in range(n):
        p = spec[i]
        if p < floor:
            continue
        left_ok = (i == 0) or spec[i - 1] < p
        right_ok = (i == n - 1) or spec[i + 1] < p
        if left_ok and right_ok:
            count += 1
    return count


def test_ResolvesCloselySpacedSourcesBeamscanCannot():
    # N=16 ULA: Beamscan's Rayleigh resolution is ~2/N = 0.125 rad in u near
    # boresight. Two sources 0.05 rad apart sit well below that limit -
    # Beamscan (fed the deterministic in-phase combination a(az0)+a(az1) -
    # the textbook two-point-source resolution setup, not an arbitrary-
    # relative-phase snapshot, which is a periodogram of one interference
    # realization and not a stable resolution measure) shows one blended
    # lobe; MUSIC, given a good covariance estimate, resolves both.
    k_n, k_s = 16, 256
    az0, az1 = 0.025, -0.025
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.01, 200)
    az_grid = linspace(-0.3, 0.3, 241)

    a0 = make_steering_snapshot(geom, conv, az0, 0.0)
    a1 = make_steering_snapshot(geom, conv, az1, 0.0)
    single_look = (a0 + a1).reshape(1, k_n)
    beamscan_snap = to_device_array(single_look, dtype=np.complex128)

    beamscan_out = run_operator_multi(
        lambda app: BeamscanEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=az_grid,
            el_angles=[0.0],
            name="op_under_test",
        ),
        single_map(beamscan_snap, "snapshot"),
        input_port="snapshot",
        output_ports=["angle_spectrum"],
        drain_ports=["aoa_estimates"],
    )
    beamscan_spec = beamscan_out["angle_spectrum"][0][0, :, 0]
    beamscan_peaks = count_prominent_local_maxima_1d(beamscan_spec)
    assert beamscan_peaks == 1, "Beamscan should show a single blended lobe"

    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az_grid,
        el_angles=[0.0],
        num_sources=2,
        max_sources=2,
    )
    music_spec = out["spectrum"][0, :, 0]
    music_peaks = count_prominent_local_maxima_1d(music_spec)
    assert music_peaks >= 2, "MUSIC should resolve two lobes where Beamscan cannot"
    assert int(out["source_count"][0]) == 2
    dm = out["doa_multi"]
    matched = (abs(dm[0, 0, 0] - az0) < 0.015 and abs(dm[0, 1, 0] - az1) < 0.015) or (
        abs(dm[0, 0, 0] - az1) < 0.015 and abs(dm[0, 1, 0] - az0) < 0.015
    )
    assert matched, f"dm[0,0,0]={dm[0, 0, 0]} dm[0,1,0]={dm[0, 1, 0]}"


def test_AutoNumSourcesResolvesTwoCloselySpacedSources():
    k_n, k_s = 16, 256
    az0, az1 = 0.025, -0.025
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, az0, 0.0, az1, 0.0, k_s, 0.01, 201)
    az_grid = linspace(-0.3, 0.3, 241)

    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=az_grid,
        el_angles=[0.0],
        max_sources=2,
    )
    assert int(out["source_count"][0]) == 2
    music_spec = out["spectrum"][0, :, 0]
    assert count_prominent_local_maxima_1d(music_spec) >= 2
    dm = out["doa_multi"]
    matched = (abs(dm[0, 0, 0] - az0) < 0.015 and abs(dm[0, 1, 0] - az1) < 0.015) or (
        abs(dm[0, 0, 0] - az1) < 0.015 and abs(dm[0, 1, 0] - az0) < 0.015
    )
    assert matched, f"dm[0,0,0]={dm[0, 0, 0]} dm[0,1,0]={dm[0, 1, 0]}"


# --- Multi-item batch (one-block-per-item kernel indexing) ------------------


def test_MultipleItemsRecoverIndependentDoa():
    k_ny, k_nz, k_s = 4, 4, 48
    az0, el0 = 0.11, 0.04
    az1, el1 = -0.07, -0.02
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    b0 = make_block(geom, conv, az0, el0, k_s, 0.02, 21)
    b1 = make_block(geom, conv, az1, el1, k_s, 0.02, 22)
    snap = np.stack([b0, b1], axis=0)
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.4, 0.4, 81),
        el_angles=linspace(-0.2, 0.2, 41),
        num_sources=1,
    )
    assert out["aoa"].shape == (2, 2)
    assert out["aoa"][0, 0] == pytest.approx(az0, abs=0.02)
    assert out["aoa"][0, 1] == pytest.approx(el0, abs=0.02)
    assert out["aoa"][1, 0] == pytest.approx(az1, abs=0.02)
    assert out["aoa"][1, 1] == pytest.approx(el1, abs=0.02)


# --- 2-D single-snapshot input acceptance -----------------------------------


def test_Accepts2DSingleSnapshot():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    a = make_steering_snapshot(geom, conv, 0.05, 0.0)
    snap = a.reshape(1, k_n)
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.2, 0.2, 21),
        el_angles=[0.0],
        num_sources=1,
    )
    assert out["spectrum"].shape[0] == 1
    assert out["aoa"][0, 0] == pytest.approx(0.05, abs=0.03)


# --- Validation gaps ---------------------------------------------------------


def test_RejectsEmptyScanGrid():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.full((1, k_n, 4), 1.0 + 0.0j, dtype=np.complex128)
    with pytest.raises(ValueError):
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[],
            el_angles=[0.0],
        )


def test_RejectsInvalidSnapshotRank():
    k_n = 4
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    snap = np.full((k_n,), 1.0 + 0.0j, dtype=np.complex128)
    with pytest.raises(ValueError):
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
        )


# --- 2-D (URA) and forward-only spatial smoothing ---------------------------


def test_SpatialSmoothing2DUraRecoversDoa():
    k_ny, k_nz, k_s = 4, 4, 32
    truth_az, truth_el = 0.10, 0.06
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.05, 44)
    snap = block.reshape(1, geom.num_elements(), k_s)

    # 2-D smoothing: 2 sub-arrays per axis -> 3x3 reduced sub-array (L = 9).
    out = run_music(
        snap,
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.4, 0.4, 81),
        el_angles=linspace(-0.3, 0.3, 61),
        spatial_smoothing=[2, 2],
        forward_backward=True,
        num_sources=1,
    )
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.03)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.03)


def test_SpatialSmoothingForwardOnlyReducesEigenvalueChannels():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.05, 45)

    out = run_music_full(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-0.3, 0.3, 61),
        el_angles=[0.0],
        spatial_smoothing=[3],
        forward_backward=False,
        num_sources=1,
    )
    # 3 sub-arrays on 8 channels -> L = 6 reduced channels.
    assert out["eigenvalues"].shape == (1, 6)
    assert out["aoa"][0, 0] == pytest.approx(0.1, abs=0.03)


# --- Live overrides ---------------------------------------------------------


def test_LiveSourceCountPfaStickyOverride():
    k_n = k_s = 6
    eigenvalues = np.asarray(
        [0.8930540251, 0.9468277841, 1.3517214181, 1.6732266229, 9.0798471606, 14.3477610683]
    )
    block = np.zeros((k_n, k_s), dtype=np.complex128)
    for c in range(k_n):
        block[c, c] = np.sqrt(float(k_s) * eigenvalues[c])
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    conv = radar_boresight_convention()

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.5, 0.5, 51),
            el_angles=[0.0],
            num_sources=-1,
            source_count_pfa=1e-4,
            max_sources=4,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["angle_spectrum"],
        overrides={"source_count_pfa": [1e-4, 0.5]},
        drain_ports=["aoa_estimates", "doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["angle_spectrum"]) == 3
    tight = float(np.max(outs["angle_spectrum"][0]))
    loose = float(np.max(outs["angle_spectrum"][1]))
    assert tight != loose
    np.testing.assert_array_equal(outs["angle_spectrum"][1], outs["angle_spectrum"][2])


def test_LiveSpatialSmoothingStickyOverride():
    k_n, k_s = 8, 32
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 141)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.3, 0.3, 31),
            el_angles=[0.0],
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
        drain_ports=["aoa_estimates", "angle_spectrum", "doa_multi", "source_count"],
    )
    assert len(outs["eigenvalues"]) == 3
    assert outs["eigenvalues"][0].shape == (1, k_n)
    assert outs["eigenvalues"][1].shape == (1, k_n - 1)
    assert outs["eigenvalues"][2].shape == outs["eigenvalues"][1].shape


def test_LiveAzGridStickyOverride():
    k_n, k_s = 4, 16
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.05, 0.0, k_s, 0.05, 41)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[-0.1, 0.0, 0.1],
            el_angles=[0.0],
            num_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["angle_spectrum"],
        overrides={"az_angles": [[-0.1, 0.0, 0.1], linspace(-0.2, 0.2, 11)]},
        drain_ports=["aoa_estimates", "doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["angle_spectrum"]) == 2
    assert outs["angle_spectrum"][0].shape[1] == 3
    assert outs["angle_spectrum"][1].shape[1] == 11


def test_LiveElGridStickyOverride():
    k_ny, k_nz, k_s = 4, 4, 48
    truth_el = 0.06
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, truth_el, k_s, 0.02, 151)
    snap = to_device_array(block.reshape(1, geom.num_elements(), k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.3, 0.3, 31),
            el_angles=[0.0],
            num_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["angle_spectrum"],
        overrides={"el_angles": [[0.0], linspace(-0.2, 0.2, 11)]},
        drain_ports=["aoa_estimates", "doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["angle_spectrum"]) == 2
    assert outs["angle_spectrum"][0].shape == (1, 31, 1)
    assert outs["angle_spectrum"][1].shape == (1, 31, 11)


def test_LiveArrayGeometryStickyOverride():
    k_n, k_s = 4, 16
    conv = radar_boresight_convention()
    geom_a = make_ula_geometry(k_n, 0.5, 1.0)
    geom_b = make_ula_geometry(k_n, 0.5, 1.0)
    geom_b.wavelength = 0.8
    block = make_block(geom_a, conv, 0.05, 0.0, k_s, 0.05, 61)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom_a,
            angle_convention=conv,
            az_angles=linspace(-0.2, 0.2, 11),
            el_angles=[0.0],
            num_sources=1,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["angle_spectrum"],
        overrides={"array_geometry": [geom_a, geom_b, geom_a]},
        drain_ports=["aoa_estimates", "doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["angle_spectrum"]) == 3
    assert outs["angle_spectrum"][0].shape == outs["angle_spectrum"][1].shape
    assert float(np.max(outs["angle_spectrum"][0])) != float(np.max(outs["angle_spectrum"][1]))
    np.testing.assert_array_equal(outs["angle_spectrum"][0], outs["angle_spectrum"][2])


def test_LiveMaxSourcesStickyOverrideReallocates():
    k_n, k_s = 12, 128
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block_two(geom, conv, 0.2, 0.0, -0.15, 0.0, k_s, 0.02, 101)
    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    frame = single_map(snap, "snapshot")
    max_sources_overrides = [1, 3]

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.3, 0.3, 61),
            el_angles=[0.0],
            num_sources=2,
            max_sources=max_sources_overrides[0],
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame],
        input_port="snapshot",
        output_ports=["doa_multi"],
        overrides={"max_sources": max_sources_overrides},
        drain_ports=["aoa_estimates", "angle_spectrum", "source_count", "eigenvalues"],
    )
    assert len(outs["doa_multi"]) == 2
    assert outs["doa_multi"][0].shape == (1, 1, 2)
    assert outs["doa_multi"][1].shape == (1, 3, 2)


# --- Remaining C++-only cases, attempted for Python parity ------------------


def test_SpatialSmoothingRejectsNonUniformGeometry():
    from holoscan.radar import ArrayGeometry

    geom = ArrayGeometry()
    geom.wavelength = 1.0
    ys = [0.0, 0.5, 1.0, 2.0]
    positions: list[float] = []
    for y in ys:
        positions.extend([0.0, y, 0.0])
    geom.element_positions = positions
    conv = radar_boresight_convention()
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap = a.reshape(1, geom.num_elements(), 1)

    with pytest.raises(ValueError):
        run_music(
            snap,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            spatial_smoothing=[2],
        )


def test_RejectsChannelMismatch():
    k_n = 4
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    conv = radar_boresight_convention()
    host = np.full((1, k_n - 1, 4), 1.0 + 0.0j, dtype=np.complex128)
    with pytest.raises(ValueError):
        run_music(
            host,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
        )


def test_RejectsSnapshotDtypeMismatch():
    # run_music()/run_music_full() always cast the input to match op_cls, so
    # a genuine mismatch (complex64 data into an F64 op) needs the lower-
    # level harness directly - the whole point here is bypassing that cast.
    k_n = 4
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    conv = radar_boresight_convention()
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    snap_f = to_device_array(a.reshape(1, k_n, 1), dtype=np.complex64)

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator_multi(
            make_op,
            single_map(snap_f, "snapshot"),
            input_port="snapshot",
            output_ports=["aoa_estimates"],
            drain_ports=["angle_spectrum", "doa_multi", "source_count", "eigenvalues"],
        )


def test_RejectsPageableHostSnapshot():
    k_n = 4
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    conv = radar_boresight_convention()
    a = make_steering_snapshot(geom, conv, 0.0, 0.0)
    # Bypass to_device_array's CuPy path deliberately: as_tensor() on a plain
    # numpy array wraps pageable host memory, which the operator must reject.
    snap_cpu = holoscan.as_tensor(a.reshape(1, k_n, 1))
    frame = single_map(snap_cpu, "snapshot")

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=[0.0],
            el_angles=[0.0],
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator_multi(
            make_op,
            frame,
            input_port="snapshot",
            output_ports=["aoa_estimates"],
            drain_ports=["angle_spectrum", "doa_multi", "source_count", "eigenvalues"],
        )


def test_DirectionCosineMasksInvisibleRegion():
    k_n, k_s = 4, 16
    conv = direction_cosine_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.0, 0.0, k_s, 0.02, 2)
    out = run_music(
        block.reshape(1, k_n, k_s),
        array_geometry=geom,
        angle_convention=conv,
        az_angles=linspace(-1.2, 1.2, 5),
        el_angles=linspace(-1.2, 1.2, 5),
        num_sources=1,
    )
    # corner (u,v)=(-1.2,-1.2): invisible.
    assert out["spectrum"][0, 0, 0] == pytest.approx(0.0, abs=1e-3)


def test_IsValidMaskExcludesWrongDoaSnapshots():
    k_n, k_s = 8, 32
    truth_az, wrong_az = 0.1, -0.2
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)

    good = make_block(geom, conv, truth_az, 0.0, k_s // 2, 0.02, 31)
    bad = make_block(geom, conv, wrong_az, 0.0, k_s // 2, 0.02, 32)
    block = np.empty((k_n, k_s), dtype=np.complex128)
    block[:, : k_s // 2] = good
    block[:, k_s // 2 :] = bad
    # uint8, not bool: the operator accepts either (dtype bool or uint8);
    # uint8 sidesteps HSDK 4.3's bool-DLPack opacity, which is only proven
    # to bite on the *output*/host-conversion side (NullSinkOp/drain_ports)
    # - untested here whether it would also bite as an input, so avoid it.
    mask = np.zeros((1, k_s), dtype=np.uint8)
    mask[0, : k_s // 2] = 1

    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    mask_dev = to_device_array(mask, dtype=np.uint8)

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.4, 0.4, 81),
            el_angles=[0.0],
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
        drain_ports=["angle_spectrum", "doa_multi", "source_count", "eigenvalues"],
    )
    assert len(outs["aoa_estimates"]) == 1
    assert outs["aoa_estimates"][0][0, 0] == pytest.approx(truth_az, abs=0.03)


def test_AllInvalidMaskBlanksDoaAndSpectrum():
    k_n, k_s = 8, 16
    conv = radar_boresight_convention()
    geom = make_ula_geometry(k_n, 0.5, 1.0)
    block = make_block(geom, conv, 0.1, 0.0, k_s, 0.02, 33)
    mask = np.zeros((1, k_s), dtype=np.uint8)

    snap = to_device_array(block.reshape(1, k_n, k_s), dtype=np.complex128)
    mask_dev = to_device_array(mask, dtype=np.uint8)

    def make_op(app):
        return MUSICEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            az_angles=linspace(-0.4, 0.4, 41),
            el_angles=[0.0],
            num_sources=1,
            max_sources=2,
            name="op_under_test",
        )

    outs = run_operator_multi(
        make_op,
        single_map(snap, "snapshot"),
        input_port="snapshot",
        output_ports=["aoa_estimates", "angle_spectrum", "doa_multi", "source_count"],
        extra_inputs={"is_valid": [single_map(mask_dev, "is_valid")]},
        drain_ports=["eigenvalues"],
    )
    aoa = outs["aoa_estimates"][0]
    assert np.isnan(aoa[0, 0])
    assert np.isnan(aoa[0, 1])

    spec = outs["angle_spectrum"][0]
    assert np.all(spec == 0.0)

    assert int(outs["source_count"][0][0]) == 0
    dm = outs["doa_multi"][0]
    assert np.all(np.isnan(dm))
