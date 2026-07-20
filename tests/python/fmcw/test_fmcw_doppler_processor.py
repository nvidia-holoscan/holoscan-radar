# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/fmcw/test_fmcw_doppler_processor.cpp."""

from __future__ import annotations

import math

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.bin_model import (  # noqa: E402
    expected_doppler_bin,
    expected_range_bin,
    expected_velocity_axis,
)
from utils.harness import (  # noqa: E402
    run_operator,
    run_operator_multi,
    to_device_array,
)
from utils.point_target_sim import PointTarget, SimScene, simulate  # noqa: E402

from holoscan.radar import (  # noqa: E402
    FMCWDopplerProcessorOp,
    FMCWDopplerProcessorOpF32,
    FMCWDopplerProcessorOpF64,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    WindowSpec,
    WindowType,
)

pytestmark = pytest.mark.requires_gpu

K_PI = math.pi


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for FMCWDopplerProcessorOp tests: {exc}")


def _test_waveform() -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77e9
    spec.bandwidth = 150e6
    spec.chirp_duration = 20e-6
    spec.sample_rate = 12.8e6
    spec.chirp_repetition_interval = 25e-6
    return FMCWWaveformDescriptor(spec)


def _tone_profiles(
    num_chirps: int,
    num_bins: int,
    range_bin: int,
    doppler_hz: float,
    pri: float,
) -> np.ndarray:
    data = np.zeros((num_chirps, num_bins), dtype=np.complex128)
    k = np.arange(num_chirps, dtype=np.float64)
    phase = 2.0 * K_PI * doppler_hz * pri * k
    data[:, range_bin] = np.exp(1j * phase)
    return data


def _profiles_frame(data: np.ndarray, key: str = "range_profiles", *, op_cls=FMCWDopplerProcessorOpF32) -> dict:
    ctype = np.complex128 if op_cls is FMCWDopplerProcessorOpF64 else np.complex64
    return {key: to_device_array(data.astype(ctype))}


def _cell_mag(rd_map: np.ndarray, batch: int, num_doppler: int, num_range: int, d: int, r: int) -> float:
    arr = np.asarray(rd_map)
    if arr.ndim == 2:
        return float(np.abs(arr[d, r]))
    return float(np.abs(arr[batch, d, r]))


def _doppler_argmax(rd_map: np.ndarray, batch: int, num_doppler: int, num_range: int, range_bin: int) -> int:
    best = 0
    best_mag = -1.0
    for d in range(num_doppler):
        mag = _cell_mag(rd_map, batch, num_doppler, num_range, d, range_bin)
        if mag > best_mag:
            best_mag = mag
            best = d
    return best


def _run_rd(
    frame: dict,
    *,
    op_cls=FMCWDopplerProcessorOpF32,
    capture_velocity: bool = False,
    **op_kwargs,
):
    def make_op(app):
        return op_cls(app, name="op_under_test", **op_kwargs)

    if capture_velocity:
        captured = run_operator_multi(
            make_op,
            frame,
            input_port="range_profiles",
            output_ports=("rd_maps", "velocity_bins"),
        )
        return captured["rd_maps"][0], captured["velocity_bins"][0]
    outs = run_operator(
        make_op, frame, input_port="range_profiles", output_port="rd_maps"
    )
    return outs[0], None


def _max_doppler_sidelobe(rd_map: np.ndarray, num_chirps: int, num_bins: int, range_bin: int) -> float:
    worst = 0.0
    for d in range(num_chirps):
        if 1 <= d <= 8:
            continue
        worst = max(
            worst,
            _cell_mag(rd_map, 0, num_chirps, num_bins, d, range_bin),
        )
    return worst


def test_VelocityPeakMatchesTruthCentered():
    num_chirps = 32
    num_bins = 16
    range_bin = 5
    radial_velocity = 5.0
    waveform = _test_waveform()
    pri = waveform.chirp_repetition_interval()
    wavelength = waveform.wavelength()
    doppler_hz = -2.0 * radial_velocity / wavelength

    frame = _profiles_frame(
        _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri),
        "profiles",
    )
    rd_map, bins = _run_rd(frame, waveform=waveform, capture_velocity=True)

    assert rd_map.shape == (num_chirps, num_bins)
    peak = _doppler_argmax(rd_map, 0, num_chirps, num_bins, range_bin)
    assert peak == expected_doppler_bin(
        radial_velocity, wavelength, pri, num_chirps, True
    )

    assert bins.shape == (num_chirps,)
    expected = expected_velocity_axis(num_chirps, wavelength, pri, True)
    np.testing.assert_allclose(
        bins.astype(np.float32), np.asarray(expected, dtype=np.float32), rtol=0, atol=0
    )

    bin_width = wavelength / 2.0 / pri / float(num_chirps)
    assert bins[peak] == pytest.approx(radial_velocity, abs=bin_width + 1e-9)


def test_VelocityAxisUncentered():
    num_chirps = 32
    num_bins = 8
    range_bin = 3
    radial_velocity = -4.0
    waveform = _test_waveform()
    pri = waveform.chirp_repetition_interval()
    wavelength = waveform.wavelength()
    doppler_hz = -2.0 * radial_velocity / wavelength

    frame = _profiles_frame(
        _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    )
    rd_map, bins = _run_rd(
        frame,
        waveform=waveform,
        center_zero_doppler=False,
        capture_velocity=True,
    )

    peak = _doppler_argmax(rd_map, 0, num_chirps, num_bins, range_bin)
    assert peak == expected_doppler_bin(
        radial_velocity, wavelength, pri, num_chirps, False
    )

    expected = expected_velocity_axis(num_chirps, wavelength, pri, False)
    np.testing.assert_allclose(
        bins.astype(np.float32), np.asarray(expected, dtype=np.float32), rtol=0, atol=0
    )
    assert bins[0] == pytest.approx(0.0)

    bin_width = wavelength / 2.0 / pri / float(num_chirps)
    assert bins[peak] == pytest.approx(radial_velocity, abs=bin_width + 1e-9)


def test_CenterZeroDopplerPlacesDc():
    num_chirps = 32
    num_bins = 4
    range_bin = 1
    dc_tone = _tone_profiles(num_chirps, num_bins, range_bin, 0.0, 25e-6)
    frame = _profiles_frame(dc_tone)

    rd_centered = _run_rd(frame)[0]
    peak = _doppler_argmax(rd_centered, 0, num_chirps, num_bins, range_bin)
    assert peak == num_chirps // 2
    assert _cell_mag(rd_centered, 0, num_chirps, num_bins, peak, range_bin) == pytest.approx(
        float(num_chirps), abs=1e-3
    )

    rd_uncentered = _run_rd(frame, center_zero_doppler=False)[0]
    assert _doppler_argmax(rd_uncentered, 0, num_chirps, num_bins, range_bin) == 0


def test_OddChirpCountFftshiftMatchesNumpy():
    num_chirps = 31
    num_bins = 4
    range_bin = 2
    waveform = _test_waveform()
    pri = waveform.chirp_repetition_interval()
    wavelength = waveform.wavelength()
    doppler_hz = 3.0 / (pri * float(num_chirps))

    frame = _profiles_frame(
        _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    )
    rd_map, bins = _run_rd(frame, waveform=waveform, capture_velocity=True)

    peak = _doppler_argmax(rd_map, 0, num_chirps, num_bins, range_bin)
    assert peak == (3 + num_chirps // 2) % num_chirps

    axis = bins.astype(np.float64)
    assert axis[num_chirps // 2] == pytest.approx(0.0)
    expected = expected_velocity_axis(num_chirps, wavelength, pri, True)
    np.testing.assert_allclose(
        axis.astype(np.float32), np.asarray(expected, dtype=np.float32), rtol=0, atol=0
    )


def test_HannWindowPreservesOnBinPeak():
    num_chirps = 32
    num_bins = 4
    range_bin = 1
    pri = 25e-6
    doppler_hz = 4.0 / (pri * float(num_chirps))
    tone = _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    frame = _profiles_frame(tone)

    def peak_magnitude(window: WindowSpec) -> float:
        rd_map = _run_rd(
            frame,
            doppler_window=window,
            center_zero_doppler=False,
        )[0]
        peak = _doppler_argmax(rd_map, 0, num_chirps, num_bins, range_bin)
        assert peak == 4
        return _cell_mag(rd_map, 0, num_chirps, num_bins, peak, range_bin)

    hann = WindowSpec()
    hann.type = WindowType.HANN
    rect_peak = peak_magnitude(WindowSpec())
    hann_peak = peak_magnitude(hann)
    assert rect_peak == pytest.approx(float(num_chirps), abs=1e-3)
    assert hann_peak == pytest.approx(rect_peak, abs=0.03 * rect_peak)


def test_HannWindowLowersDopplerSidelobes():
    num_chirps = 32
    num_bins = 4
    range_bin = 2
    pri = 25e-6
    doppler_hz = 4.5 / (pri * float(num_chirps))
    tone = _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    frame = _profiles_frame(tone)

    def max_sidelobe(window: WindowSpec) -> float:
        rd_map = _run_rd(
            frame,
            doppler_window=window,
            center_zero_doppler=False,
        )[0]
        return _max_doppler_sidelobe(rd_map, num_chirps, num_bins, range_bin)

    hann = WindowSpec()
    hann.type = WindowType.HANN
    rect_sidelobe = max_sidelobe(WindowSpec())
    hann_sidelobe = max_sidelobe(hann)
    assert rect_sidelobe > 0.0
    assert hann_sidelobe < 0.4 * rect_sidelobe


def test_NfftInterpolatesVelocityGrid():
    num_chirps = 32
    nfft = 64
    num_bins = 8
    range_bin = 6
    radial_velocity = 7.0
    waveform = _test_waveform()
    pri = waveform.chirp_repetition_interval()
    wavelength = waveform.wavelength()
    doppler_hz = -2.0 * radial_velocity / wavelength

    frame = _profiles_frame(
        _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    )
    rd_map, bins = _run_rd(frame, waveform=waveform, nfft=nfft, capture_velocity=True)

    assert rd_map.shape == (nfft, num_bins)
    axis = bins.astype(np.float64)
    assert axis.shape == (nfft,)
    expected = expected_velocity_axis(nfft, wavelength, pri, True)
    np.testing.assert_allclose(
        axis.astype(np.float32), np.asarray(expected, dtype=np.float32), rtol=0, atol=0
    )

    original_bin_width = wavelength / 2.0 / pri / float(num_chirps)
    assert abs(axis[1] - axis[0]) == pytest.approx(
        original_bin_width / 2.0, abs=1e-5 * original_bin_width
    )
    peak = _doppler_argmax(rd_map, 0, nfft, num_bins, range_bin)
    assert axis[peak] == pytest.approx(radial_velocity, abs=original_bin_width + 1e-9)


def test_LeadingBatchAxesProcessedIndependently():
    batch = 2
    num_chirps = 32
    num_bins = 8
    pri = 25e-6
    f0 = 3.0 / (pri * float(num_chirps))
    f1 = 10.0 / (pri * float(num_chirps))
    slice0 = _tone_profiles(num_chirps, num_bins, 2, f0, pri)
    slice1 = _tone_profiles(num_chirps, num_bins, 6, f1, pri)
    cube = np.stack([slice0, slice1], axis=0)
    frame = _profiles_frame(cube)

    rd_map = _run_rd(frame, center_zero_doppler=False)[0]
    assert rd_map.shape == (batch, num_chirps, num_bins)
    assert _doppler_argmax(rd_map, 0, num_chirps, num_bins, 2) == 3
    assert _doppler_argmax(rd_map, 1, num_chirps, num_bins, 6) == 10
    assert _cell_mag(rd_map, 0, num_chirps, num_bins, 10, 6) < 1e-3
    assert _cell_mag(rd_map, 1, num_chirps, num_bins, 3, 2) < 1e-3


def test_LiveOverrideCenterZeroDoppler():
    num_chirps = 32
    num_bins = 4
    range_bin = 1
    waveform = _test_waveform()
    pri = waveform.chirp_repetition_interval()
    wavelength = waveform.wavelength()
    tone = _tone_profiles(num_chirps, num_bins, range_bin, 0.0, pri)
    frame = _profiles_frame(tone)
    frames = [frame, frame]

    rd_default = _run_rd(frame, waveform=waveform)[0]
    assert _doppler_argmax(rd_default, 0, num_chirps, num_bins, range_bin) == (
        num_chirps // 2
    )

    def make_op(app):
        return FMCWDopplerProcessorOpF32(
            app, waveform=waveform, name="op_under_test"
        )

    captured = run_operator_multi(
        make_op,
        frames,
        input_port="range_profiles",
        output_ports=("rd_maps", "velocity_bins"),
        overrides={"center_zero_doppler": [False]},
    )
    assert len(captured["rd_maps"]) == 2
    assert _doppler_argmax(captured["rd_maps"][0], 0, num_chirps, num_bins, range_bin) == 0
    assert _doppler_argmax(captured["rd_maps"][1], 0, num_chirps, num_bins, range_bin) == 0
    expected = expected_velocity_axis(num_chirps, wavelength, pri, False)
    assert np.allclose(captured["velocity_bins"][1], expected, rtol=0, atol=1e-5)


def test_LiveOverrideDopplerWindow():
    num_chirps = 32
    num_bins = 4
    range_bin = 2
    pri = 25e-6
    doppler_hz = 4.5 / (pri * float(num_chirps))
    tone = _tone_profiles(num_chirps, num_bins, range_bin, doppler_hz, pri)
    frame = _profiles_frame(tone)
    frames = [frame, frame]

    rd_rect = _run_rd(frame, center_zero_doppler=False)[0]
    rect_sidelobe = _max_doppler_sidelobe(rd_rect, num_chirps, num_bins, range_bin)

    hann = WindowSpec()
    hann.type = WindowType.HANN

    def make_op(app):
        return FMCWDopplerProcessorOpF32(
            app, center_zero_doppler=False, name="op_under_test"
        )

    outs = run_operator(
        make_op,
        frames,
        input_port="range_profiles",
        output_port="rd_maps",
        overrides={"doppler_window": [hann]},
    )
    assert len(outs) == 2
    assert rect_sidelobe > 0.0
    for tick, rd in enumerate(outs):
        hann_sidelobe = _max_doppler_sidelobe(rd, num_chirps, num_bins, range_bin)
        assert hann_sidelobe < 0.4 * rect_sidelobe, f"tick {tick}"


def test_VelocityBinsOnlyEmittedWithWaveform():
    num_chirps = 32
    num_bins = 4
    frame = _profiles_frame(
        _tone_profiles(num_chirps, num_bins, 1, 0.0, 25e-6)
    )

    rd_map = _run_rd(frame)[0]
    assert rd_map.shape == (num_chirps, num_bins)

    rd_map2, bins = _run_rd(frame, waveform=_test_waveform(), capture_velocity=True)
    assert rd_map2.shape == (num_chirps, num_bins)
    assert bins.shape == (num_chirps,)


def test_FloatAndDoubleAgree():
    num_chirps = 32
    num_bins = 8
    pri = 25e-6
    doppler_hz = 6.3 / (pri * float(num_chirps))
    tone = _tone_profiles(num_chirps, num_bins, 4, doppler_hz, pri)
    frame = _profiles_frame(tone)

    rd_f = _run_rd(frame, op_cls=FMCWDopplerProcessorOpF32)[0]
    rd_d = _run_rd(
        _profiles_frame(tone, op_cls=FMCWDopplerProcessorOpF64),
        op_cls=FMCWDopplerProcessorOpF64,
    )[0]
    assert rd_f.shape == rd_d.shape
    for d in range(num_chirps):
        for r in range(num_bins):
            mag_f = _cell_mag(rd_f, 0, num_chirps, num_bins, d, r)
            mag_d = _cell_mag(rd_d, 0, num_chirps, num_bins, d, r)
            assert mag_f == pytest.approx(mag_d, abs=1e-3 * float(num_chirps)), (
                f"(doppler={d}, range={r})"
            )


def test_RealInputRejected():
    num_chirps = 8
    num_bins = 4
    real = np.ones((num_chirps, num_bins), dtype=np.float32)
    frame = {"range_profiles": to_device_array(real)}

    def make_op(app):
        return FMCWDopplerProcessorOp(app, name="op_under_test")

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="range_profiles", output_port="rd_maps")


def test_MultiEntryInputRejected():
    tone = _tone_profiles(8, 4, 1, 0.0, 25e-6).astype(np.complex64)
    frame = {"a": to_device_array(tone), "b": to_device_array(tone.copy())}

    def make_op(app):
        return FMCWDopplerProcessorOp(app, name="op_under_test")

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="range_profiles", output_port="rd_maps")


def test_NfftSmallerThanChirpCountRejected():
    tone = _tone_profiles(32, 4, 1, 0.0, 25e-6)
    frame = _profiles_frame(tone)

    def make_op(app):
        return FMCWDopplerProcessorOp(app, nfft=16, name="op_under_test")

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="range_profiles", output_port="rd_maps")


def _derive_radial_velocity(target: PointTarget) -> float:
    rng = math.sqrt(sum(x * x for x in target.position))
    look = tuple(x / rng for x in target.position)
    return sum(v * lk for v, lk in zip(target.velocity, look, strict=True))


def test_SimulatorTargetMapsToTruthVelocity():
    scene = SimScene(
        num_chirps=32,
        samples_per_chirp=64,
        targets=[PointTarget(position=(20.0, 0.0, 0.0), velocity=(6.0, 0.0, 0.0))],
    )
    sim = simulate(scene)
    radial_velocity = _derive_radial_velocity(scene.targets[0])
    target_range = math.sqrt(20.0 * 20.0)

    num_chirps = scene.num_chirps
    samples = scene.samples_per_chirp
    cube = sim.cube.reshape(sim.shape)
    profiles = np.zeros((num_chirps, samples), dtype=np.complex128)
    for k in range(num_chirps):
        for b in range(samples):
            acc = 0.0 + 0.0j
            for s in range(samples):
                phase = -2.0 * K_PI * float(b) * float(s) / float(samples)
                acc += cube[k, s] * np.exp(1j * phase)
            profiles[k, b] = acc

    spec = FMCWWaveformSpec()
    spec.carrier_frequency = scene.waveform.carrier_frequency
    spec.bandwidth = scene.waveform.bandwidth
    spec.chirp_duration = scene.waveform.chirp_duration
    spec.sample_rate = scene.waveform.sample_rate
    spec.chirp_repetition_interval = scene.waveform.chirp_repetition_interval
    waveform = FMCWWaveformDescriptor(spec)

    frame = _profiles_frame(profiles)
    rd_map, bins = _run_rd(frame, waveform=waveform, capture_velocity=True)

    range_bin = expected_range_bin(
        target_range,
        scene.waveform.slope(),
        scene.waveform.sample_rate,
        samples,
    )
    peak = _doppler_argmax(rd_map, 0, num_chirps, samples, range_bin)
    assert peak == expected_doppler_bin(
        radial_velocity,
        scene.waveform.wavelength(),
        scene.waveform.chirp_repetition_interval,
        num_chirps,
        True,
    )

    bin_width = (
        scene.waveform.wavelength()
        / 2.0
        / scene.waveform.chirp_repetition_interval
        / float(num_chirps)
    )
    assert bins[peak] == pytest.approx(radial_velocity, abs=bin_width + 1e-9)
