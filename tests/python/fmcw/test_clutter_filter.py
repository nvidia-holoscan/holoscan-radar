# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.radar import K_PI  # noqa: E402
from holoscan.radar import ClutterFilterOp  # noqa: E402
from holoscan.radar import K_SPEED_OF_LIGHT as K_C  # noqa: E402
from holoscan.radar import (
    ClutterFilterMethod,
    ClutterFilterOpF32,
    ClutterFilterOpF64,
    WindowSpec,
    WindowType,
)

pytestmark = pytest.mark.requires_gpu

K_CHIRPS = 16
K_BINS = 64


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
    doppler_window: WindowSpec | None = None,
) -> np.ndarray:
    arr = np.asarray(host).reshape(shape)
    frame = single_map(to_device_array(arr), key)

    def make_op(app):
        kwargs = {"name": "op_under_test"}
        if method is not None:
            kwargs["method"] = method
        if doppler_window is not None:
            kwargs["doppler_window"] = doppler_window
        return op_cls(app, **kwargs)

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


def _window(kind: WindowType, length: int) -> tuple[WindowSpec, np.ndarray]:
    spec = WindowSpec()
    spec.type = kind
    k = np.arange(length, dtype=np.float64)
    phase = 2.0 * np.pi * k / (length - 1)
    if kind == WindowType.HANN:
        taps = 0.5 - 0.5 * np.cos(phase)
    elif kind == WindowType.HAMMING:
        taps = 0.54 - 0.46 * np.cos(phase)
    else:
        raise AssertionError(kind)
    return spec, taps


@pytest.mark.parametrize(
    "dtype,op_cls", [(np.complex64, ClutterFilterOpF32), (np.complex128, ClutterFilterOpF64)]
)
@pytest.mark.parametrize("kind", [WindowType.HANN, WindowType.HAMMING])
def test_WeightedMeanMatchesIndependentReference(dtype, op_cls, kind):
    spec, taps = _window(kind, K_CHIRPS)
    host = doppler_tone(K_CHIRPS, 5, 0.173, dtype=dtype)
    host += np.asarray(2.0 - 0.5j, dtype=dtype)
    expected = host - np.sum(taps[:, None] * host, axis=0) / np.sum(taps)
    output = _run_single(
        host,
        host.shape,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        op_cls=op_cls,
        doppler_window=spec,
    )
    tolerance = 2e-5 if dtype == np.complex64 else 2e-12
    np.testing.assert_allclose(output, expected, rtol=0, atol=tolerance)


def test_CustomUnnormalizedWeightedMeanAndZeroSumRejection():
    taps = np.linspace(0.2, 1.7, K_CHIRPS)
    spec = WindowSpec()
    spec.type = WindowType.CUSTOM
    spec.coefficients = taps.tolist()
    spec.coherent_gain_normalization = False
    host = doppler_tone(K_CHIRPS, 3, 0.173)
    expected = host - np.sum(taps[:, None] * host, axis=0) / np.sum(taps)
    output = _run_single(
        host,
        host.shape,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        doppler_window=spec,
    )
    np.testing.assert_allclose(output, expected, rtol=0, atol=2e-5)

    spec.coefficients = ([1.0] * (K_CHIRPS - 1)) + [-(K_CHIRPS - 1)]
    with pytest.raises(ValueError):
        _run_single(
            host,
            host.shape,
            ClutterFilterMethod.MEAN_SUBTRACTION,
            doppler_window=spec,
        )


def test_MatchingWindowMakesDcZeroWhereUniformProjectionLeavesResidual():
    spec, taps = _window(WindowType.HANN, K_CHIRPS)
    host = doppler_tone(K_CHIRPS, 1, 0.137, dtype=np.complex128)
    weighted = _run_single(
        host,
        host.shape,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        op_cls=ClutterFilterOpF64,
        doppler_window=spec,
    )
    uniform = _run_single(
        host,
        host.shape,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        op_cls=ClutterFilterOpF64,
    )
    assert abs(np.sum(taps[:, None] * weighted)) < 2e-12
    assert abs(np.sum(taps[:, None] * uniform)) > 1e-2


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


def test_DopplerWindowConfigurableFromYamlNode(tmp_path):
    cfg = tmp_path / "clutter_filter_window.yaml"
    cfg.write_text(
        "%YAML 1.2\n---\nclutter_filter:\n" "  method: mean_subtraction\n  doppler_window: hann\n"
    )
    tone = doppler_tone(K_CHIRPS, 3, 0.173)
    frame = single_map(to_device_array(tone), "range_profiles")

    def make_op(app):
        app.config(str(cfg))
        return ClutterFilterOpF32(app, app.from_config("clutter_filter"), name="op_under_test")

    output = run_operator(make_op, frame)[0]
    _, taps = _window(WindowType.HANN, K_CHIRPS)
    expected = tone - np.sum(taps[:, None] * tone, axis=0) / np.sum(taps)
    np.testing.assert_allclose(output, expected, rtol=0, atol=2e-5)


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


def test_LiveMethodOverrideAppliesPerTick():
    """Single-tick [live] method override via typed Python→C++ emit."""
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
    # Tick 2 has no override, so the Params default (mean_subtraction) applies.
    assert outs[2].shape == (K_CHIRPS, K_BINS)


def test_LiveDopplerWindowOverrideAppliesOneTickThenReverts():
    tone = doppler_tone(K_CHIRPS, 3, 0.173)
    frame = single_map(to_device_array(tone), "range_profiles")
    hann, _ = _window(WindowType.HANN, K_CHIRPS)

    def make_op(app):
        return ClutterFilterOpF32(app, name="op_under_test")

    outputs = run_operator(
        make_op,
        [frame, frame, frame],
        overrides={"doppler_window": [None, hann, None]},
    )
    assert len(outputs) == 3
    np.testing.assert_array_equal(outputs[2], outputs[0])
    assert _max_abs(outputs[1] - outputs[0]) > 1e-3


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "op_cls"),
    [(np.complex64, ClutterFilterOpF32), (np.complex128, ClutterFilterOpF64)],
)
def test_MethodNonePassesThrough(dtype, op_cls):
    tone = doppler_tone(K_CHIRPS, K_BINS, 0.19, dtype=dtype)
    output = _run_single(
        tone,
        (K_CHIRPS, K_BINS),
        ClutterFilterMethod.NONE,
        op_cls=op_cls,
    )
    np.testing.assert_array_equal(output, tone)


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
