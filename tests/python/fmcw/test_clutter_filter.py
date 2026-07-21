# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/fmcw/test_clutter_filter.cpp.

C++-parity coverage for ClutterFilterOp via the tests/python/utils harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import ClutterFilterOp  # noqa: E402
from holoscan.radar import ClutterFilterMethod, ClutterFilterOpF32, ClutterFilterOpF64

pytestmark = pytest.mark.requires_gpu

K_CHIRPS = 16
K_BINS = 64
K_PI = math.pi
K_C = 299792458.0


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for ClutterFilterOp tests: {exc}")


# ---------------------------------------------------------------------------
# Analytic point-target simulator (subset of radar_common/sim/point_target_sim)
# ---------------------------------------------------------------------------


@dataclass
class _SimWaveform:
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
class _PointTarget:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0 + 0.0j


@dataclass
class _SimScene:
    targets: list[_PointTarget] = field(default_factory=list)
    waveform: _SimWaveform = field(default_factory=_SimWaveform)
    num_chirps: int = 64
    samples_per_chirp: int = 256
    tx_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )
    propagation_speed: float = K_C
    dechirped: bool = True


@dataclass
class _SimResult:
    cube: np.ndarray  # complex128, row-major
    shape: tuple[int, ...]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def _simulate(scene: _SimScene) -> _SimResult:
    """Minimal SISO/SIMO port of point_target_sim.hpp::simulate."""
    num_rx = len(scene.rx_positions)
    nc = scene.num_chirps
    ns = scene.samples_per_chirp
    c = scene.propagation_speed
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    pri = scene.waveform.chirp_repetition_interval

    cube = np.zeros((num_rx, nc, ns), dtype=np.complex128)
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
        for rx, rx_pos in enumerate(scene.rx_positions):
            aperture = _dot(scene.tx_position, look) + _dot(rx_pos, look)
            for k in range(nc):
                range_k = rng + v_r * k * pri
                tau = (2.0 * range_k - aperture) / c
                beat = slope * tau
                t = np.arange(ns, dtype=np.float64) / fs
                phase = 2.0 * K_PI * (beat * t) - 2.0 * K_PI * f_c * tau
                cube[rx, k, :] += target.amplitude * np.exp(1j * phase)

    if not scene.dechirped:
        t = np.arange(ns, dtype=np.float64) / fs
        ref = np.exp(1j * K_PI * slope * t * t)
        cube *= ref[np.newaxis, np.newaxis, :]

    if num_rx == 1:
        flat = cube[0]
        return _SimResult(cube=flat.reshape(-1), shape=flat.shape)
    return _SimResult(cube=cube.reshape(-1), shape=cube.shape)


def _moving_target(waveform: _SimWaveform, f: float) -> _PointTarget:
    f_d = f / waveform.chirp_repetition_interval
    v_r = -f_d * waveform.wavelength() / 2.0
    return _PointTarget(position=(45.0, 0.0, 0.0), velocity=(v_r, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Fixtures matching the C++ helpers
# ---------------------------------------------------------------------------


def doppler_tone(num_chirps: int, num_bins: int, f: float, dtype=np.complex64):
    """Per-chirp complex exponential x[k,r] = a_r * exp(j*2*pi*f*k)."""
    k = np.arange(num_chirps, dtype=np.float64)[:, None]
    r = np.arange(num_bins, dtype=np.float64)[None, :]
    amp = 1.0 + r / float(num_bins)
    phase = 2.0 * K_PI * f * k
    return (amp * np.exp(1j * phase)).astype(dtype)


def _run_single(
    host: np.ndarray,
    shape: tuple[int, ...] | list[int],
    method: ClutterFilterMethod | None = None,
    *,
    key: str = "range_profiles",
    op_cls=ClutterFilterOpF32,
) -> np.ndarray:
    arr = np.asarray(host).reshape(shape)
    frame = single_map(to_device_array(arr), key)

    def make_op(app):
        if method is None:
            return op_cls(app, name="op_under_test")
        return op_cls(app, method=method, name="op_under_test")

    outs = run_operator(make_op, frame)
    assert len(outs) == 1
    return outs[0]


def _max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def _tensor_energy(arr: np.ndarray) -> float:
    return float(np.sum(np.abs(arr) ** 2))


# ---------------------------------------------------------------------------
# mean_subtraction
# ---------------------------------------------------------------------------


def test_MeanSubtractionNullsStaticTarget():
    scene = _SimScene(
        num_chirps=K_CHIRPS,
        samples_per_chirp=K_BINS,
        targets=[_PointTarget(position=(30.0, 0.0, 0.0))],
    )
    result = _simulate(scene)
    host = result.cube.astype(np.complex64)
    peak_in = float(np.max(np.abs(result.cube)))

    out = _run_single(host, result.shape, None)
    assert out.shape == result.shape
    assert _max_abs(out) < 1e-4 * peak_in


def test_MeanSubtractionPreservesMovingTarget():
    scene = _SimScene(num_chirps=K_CHIRPS, samples_per_chirp=K_BINS)
    moving = _moving_target(scene.waveform, 4.0 / 16.0)
    stationary = _PointTarget(position=(30.0, 0.0, 0.0))

    scene.targets = [moving]
    moving_only = _simulate(scene)
    scene.targets = [stationary, moving]
    both = _simulate(scene)

    host = both.cube.astype(np.complex64)
    out = _run_single(host, both.shape, ClutterFilterMethod.MEAN_SUBTRACTION)
    assert out.shape == both.shape
    e_out = _tensor_energy(out)
    e_moving = float(np.sum(np.abs(moving_only.cube) ** 2))
    assert e_out / e_moving == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# delay cancellers
# ---------------------------------------------------------------------------


def test_SingleDelayShapeAndResponse():
    f = 0.19
    tone = doppler_tone(K_CHIRPS, K_BINS, f)
    out = _run_single(tone, (K_CHIRPS, K_BINS), ClutterFilterMethod.SINGLE_DELAY)
    assert out.shape == (K_CHIRPS - 1, K_BINS)

    expected_gain = abs(1.0 - np.exp(1j * 2.0 * K_PI * f))
    for k in range(K_CHIRPS - 1):
        for r in range(K_BINS):
            amp = abs(tone[k, r])
            gain = abs(out[k, r]) / amp
            assert gain == pytest.approx(
                expected_gain, abs=1e-4 * expected_gain
            ), f"cell ({k}, {r})"


def test_MethodConfigurableFromYamlNode(tmp_path):
    """YAML via Fragment.from_config must configure method (C++ Arg parity).

    Holoscan Python apps pass ``self.from_config("clutter_filter")`` as a
    positional Arg/ArgList; the binding trampoline forwards it so
    ``YAML::convert<ClutterFilterMethod>`` runs at initialize().
    """
    cfg = tmp_path / "clutter_filter.yaml"
    cfg.write_text("%YAML 1.2\n---\nclutter_filter:\n  method: single_delay\n")
    tone = doppler_tone(K_CHIRPS, K_BINS, 0.19)
    arr = np.asarray(tone).reshape((K_CHIRPS, K_BINS))
    frame = single_map(to_device_array(arr), "range_profiles")

    def make_op(app):
        app.config(str(cfg))
        return ClutterFilterOpF32(app, app.from_config("clutter_filter"), name="op_under_test")

    outs = run_operator(make_op, frame)
    assert len(outs) == 1
    # single_delay drops one chirp — proof the YAML value took effect (the
    # default, mean_subtraction, preserves the chirp count).
    assert outs[0].shape == (K_CHIRPS - 1, K_BINS)


def test_DoubleDelayShapeAndResponse():
    f = 0.19
    tone = doppler_tone(K_CHIRPS, K_BINS, f)
    out = _run_single(tone, (K_CHIRPS, K_BINS), ClutterFilterMethod.DOUBLE_DELAY)
    assert out.shape == (K_CHIRPS - 2, K_BINS)

    base = abs(1.0 - np.exp(1j * 2.0 * K_PI * f))
    expected_gain = base * base
    for k in range(K_CHIRPS - 2):
        for r in range(K_BINS):
            amp = abs(tone[k, r])
            gain = abs(out[k, r]) / amp
            assert gain == pytest.approx(
                expected_gain, abs=1e-4 * expected_gain
            ), f"cell ({k}, {r})"


def test_DelayCancellersNotchDC():
    dc = doppler_tone(K_CHIRPS, K_BINS, 0.0)
    for method in (
        ClutterFilterMethod.SINGLE_DELAY,
        ClutterFilterMethod.DOUBLE_DELAY,
    ):
        out = _run_single(dc, (K_CHIRPS, K_BINS), method)
        assert _max_abs(out) < 1e-5, f"method {method}"


# ---------------------------------------------------------------------------
# leading batch axes
# ---------------------------------------------------------------------------


def test_LeadingBatchAxesPreservedAndIndependent():
    batch, nc, nb = 2, 8, 16
    slice0 = doppler_tone(nc, nb, 0.25)
    slice1 = doppler_tone(nc, nb, 0.125) + np.complex64(2.0 - 1.0j)
    host = np.stack([slice0, slice1], axis=0)

    out = _run_single(host, (batch, nc, nb), ClutterFilterMethod.MEAN_SUBTRACTION, key="adc")
    assert out.shape == (batch, nc, nb)

    for b in range(batch):
        for r in range(nb):
            mean = np.mean(host[b, :, r])
            expected = host[b, :, r] - mean
            got = out[b, :, r]
            assert np.max(np.abs(got - expected)) < 1e-5, f"cell batch={b} bin={r}"

    delayed = _run_single(host, (batch, nc, nb), ClutterFilterMethod.SINGLE_DELAY)
    assert delayed.shape == (batch, nc - 1, nb)


# ---------------------------------------------------------------------------
# live method override
# ---------------------------------------------------------------------------


def test_LiveMethodOverrideMidStreamAndSticky():
    """Sticky [live] method override via typed Python→C++ emit."""
    tone = doppler_tone(K_CHIRPS, K_BINS, 0.19)
    frame = single_map(to_device_array(tone), "range_profiles")
    frames = [frame, frame, frame]

    def make_op(app):
        return ClutterFilterOpF32(app, name="op_under_test")

    outs = run_operator(
        make_op,
        frames,
        overrides={
            "method": [
                ClutterFilterMethod.SINGLE_DELAY,
                ClutterFilterMethod.MEAN_SUBTRACTION,
            ]
        },
    )
    assert len(outs) == 3
    assert outs[0].shape == (K_CHIRPS - 1, K_BINS)
    assert outs[1].shape == (K_CHIRPS, K_BINS)
    assert outs[2].shape == (K_CHIRPS, K_BINS)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_RejectsMethodNone():
    tone = doppler_tone(K_CHIRPS, K_BINS, 0.19)
    with pytest.raises(ValueError, match="none"):
        _run_single(tone, (K_CHIRPS, K_BINS), ClutterFilterMethod.NONE)


def test_RejectsRealInput():
    real_cube = np.ones((K_CHIRPS, K_BINS), dtype=np.float32)
    frame = single_map(to_device_array(real_cube), "range_profiles")

    def make_op(app):
        return ClutterFilterOp(app, name="op_under_test")

    with pytest.raises(ValueError, match="range_profiles"):
        run_operator(make_op, frame)


def test_RejectsTooFewChirps():
    one_chirp = doppler_tone(1, K_BINS, 0.0)
    with pytest.raises(ValueError):
        _run_single(one_chirp, (1, K_BINS), ClutterFilterMethod.SINGLE_DELAY)

    two_chirps = doppler_tone(2, K_BINS, 0.0)
    with pytest.raises(ValueError):
        _run_single(two_chirps, (2, K_BINS), ClutterFilterMethod.DOUBLE_DELAY)


def test_RejectsMultiEntryTensorMap():
    tone = doppler_tone(K_CHIRPS, K_BINS, 0.19)
    device = to_device_array(tone)
    frame = {"a": device, "b": to_device_array(tone.copy())}

    def make_op(app):
        return ClutterFilterOp(app, name="op_under_test")

    with pytest.raises(ValueError):
        run_operator(make_op, frame)


# ---------------------------------------------------------------------------
# precision agreement
# ---------------------------------------------------------------------------


def test_FloatAndDoubleAgree():
    tone_d = doppler_tone(K_CHIRPS, K_BINS, 0.15, dtype=np.complex128)
    tone_f = tone_d.astype(np.complex64)

    for method in (
        ClutterFilterMethod.MEAN_SUBTRACTION,
        ClutterFilterMethod.SINGLE_DELAY,
        ClutterFilterMethod.DOUBLE_DELAY,
    ):
        out_f = _run_single(tone_f, (K_CHIRPS, K_BINS), method, op_cls=ClutterFilterOpF32)
        out_d = _run_single(tone_d, (K_CHIRPS, K_BINS), method, op_cls=ClutterFilterOpF64)
        assert out_f.shape == out_d.shape
        scale = max(_max_abs(out_d), 1.0)
        diff = np.abs(out_f.astype(np.complex128) - out_d)
        assert np.max(diff) < 1e-5 * scale, f"method {method}"
