# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/fmcw/test_fmcw_range_processor.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.bin_model import (  # noqa: E402
    K_C,
    expected_range_axis,
    range_of_bin,
)
from utils.harness import (  # noqa: E402
    run_operator,
    run_operator_multi,
    single_map,
    to_device_array,
)
from utils.point_target_sim import (  # noqa: E402
    PointTarget,
    SimScene,
    SimWaveform,
    simulate,
)

from holoscan.radar import (  # noqa: E402
    FMCWRangeProcessorOp,
    FMCWRangeProcessorOpF32,
    FMCWRangeProcessorOpF64,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    WindowSpec,
    WindowType,
)

pytestmark = pytest.mark.requires_gpu

K_CHIRPS = 32
K_SAMPLES = 128


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for FMCWRangeProcessorOp tests: {exc}")


def _bin_width(nfft: int) -> float:
    waveform = SimWaveform()
    return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft, K_C)


def _range_at_bin(bin_idx: float) -> float:
    return bin_idx * _bin_width(K_SAMPLES)


def _to_descriptor(waveform: SimWaveform) -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = waveform.carrier_frequency
    spec.bandwidth = waveform.bandwidth
    spec.chirp_duration = waveform.chirp_duration
    spec.sample_rate = waveform.sample_rate
    spec.chirp_repetition_interval = waveform.chirp_repetition_interval
    return FMCWWaveformDescriptor(spec)


def _make_scene(range_m: float, *, dechirped: bool = True) -> SimScene:
    return SimScene(
        num_chirps=K_CHIRPS,
        samples_per_chirp=K_SAMPLES,
        dechirped=dechirped,
        targets=[PointTarget(position=(range_m, 0.0, 0.0))],
    )


def _to_real_if(cube: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(cube).reshape(shape).real.astype(np.float32)


def _slice_abs(profiles: np.ndarray, batch: int, chirp: int) -> np.ndarray:
    arr = np.asarray(profiles)
    if arr.ndim == 2:
        row = arr[chirp]
    else:
        row = arr[batch, chirp]
    return np.abs(row)


def _argmax(values: np.ndarray) -> int:
    return int(np.argmax(values))


def _run_complex(
    sim,
    *,
    op_cls=FMCWRangeProcessorOpF32,
    output_ports=("range_profiles", "range_bins"),
    **op_kwargs,
):
    ctype = np.complex128 if op_cls is FMCWRangeProcessorOpF64 else np.complex64
    host = sim.cube.astype(ctype)
    frame = single_map(to_device_array(host.reshape(sim.shape)), "adc_cube")

    def make_op(app):
        return op_cls(app, name="op_under_test", **op_kwargs)

    if len(output_ports) == 1:
        outs = run_operator(
            make_op, frame, input_port="adc_cube", output_port=output_ports[0]
        )
        return {output_ports[0]: outs[0]}
    captured = run_operator_multi(
        make_op,
        frame,
        input_port="adc_cube",
        output_ports=output_ports,
    )
    return {port: frames[0] for port, frames in captured.items()}


def _run_real(sim, *, op_cls=FMCWRangeProcessorOpF32, **op_kwargs):
    real = _to_real_if(sim.cube, sim.shape)
    frame = single_map(to_device_array(real), "adc_cube")

    def make_op(app):
        return op_cls(app, name="op_under_test", **op_kwargs)

    captured = run_operator_multi(
        make_op,
        frame,
        input_port="adc_cube",
        output_ports=("range_profiles", "range_bins"),
    )
    return {port: frames[0] for port, frames in captured.items()}


def test_SingleTargetComplexIfPeakAndRangeAxis():
    waveform = SimWaveform()
    target_range = _range_at_bin(30.0)
    sim = simulate(_make_scene(target_range))
    out = _run_complex(sim, waveform=_to_descriptor(waveform))

    profiles = out["range_profiles"]
    bins = out["range_bins"]
    assert profiles.shape == (K_CHIRPS, K_SAMPLES)
    assert bins.shape == (K_SAMPLES,)

    magnitude = _slice_abs(profiles, 0, 0)
    peak = _argmax(magnitude)
    assert bins[peak] == pytest.approx(target_range, abs=_bin_width(K_SAMPLES))

    expected = expected_range_axis(
        K_SAMPLES, waveform.slope(), waveform.sample_rate, K_SAMPLES, K_C
    )
    for i, exp in enumerate(expected):
        assert bins[i] == pytest.approx(exp, abs=1e-3 + 1e-5 * abs(exp)), (
            f"bin {i}"
        )


def test_RealIfPathPeakAtRange():
    waveform = SimWaveform()
    target_range = _range_at_bin(30.0)
    sim = simulate(_make_scene(target_range))
    out = _run_real(sim, waveform=_to_descriptor(waveform))

    profiles = out["range_profiles"]
    bins = out["range_bins"]
    assert profiles.shape == (K_CHIRPS, K_SAMPLES // 2 + 1)
    assert bins.shape == (K_SAMPLES // 2 + 1,)

    magnitude = _slice_abs(profiles, 0, 0)
    peak = _argmax(magnitude)
    assert bins[peak] == pytest.approx(target_range, abs=_bin_width(K_SAMPLES))


def test_NoRangeBinsWithoutWaveform():
    sim = simulate(_make_scene(_range_at_bin(30.0)))
    out = _run_complex(sim, output_ports=("range_profiles",))
    profiles = out["range_profiles"]
    assert profiles.shape == (K_CHIRPS, K_SAMPLES)


def test_ApplyDechirpMatchesHardwareDechirp():
    waveform = SimWaveform()
    target_range = _range_at_bin(30.5)
    hw = simulate(_make_scene(target_range, dechirped=True))
    pre_mixer = simulate(_make_scene(target_range, dechirped=False))
    desc = _to_descriptor(waveform)

    hw_out = _run_complex(hw, waveform=desc)
    sw_out = _run_complex(pre_mixer, waveform=desc, apply_dechirp=True)

    a = hw_out["range_profiles"]
    b = sw_out["range_profiles"]
    assert a.shape == b.shape
    peak_mag = float(np.max(np.abs(a)))
    assert peak_mag > 0.0
    diff = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    assert np.max(diff) <= 2e-3 * peak_mag


def test_ApplyDechirpWithoutWaveformErrors():
    sim = simulate(_make_scene(_range_at_bin(30.0), dechirped=False))
    frame = single_map(
        to_device_array(sim.cube.astype(np.complex64).reshape(sim.shape)),
        "adc_cube",
    )

    def make_op(app):
        return FMCWRangeProcessorOp(app, apply_dechirp=True, name="op_under_test")

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="range_profiles")


def test_ApplyDechirpWithRealInputErrors():
    waveform = SimWaveform()
    sim = simulate(_make_scene(_range_at_bin(30.0)))
    real = _to_real_if(sim.cube, sim.shape)
    frame = single_map(to_device_array(real), "adc_cube")

    def make_op(app):
        return FMCWRangeProcessorOp(
            app,
            apply_dechirp=True,
            waveform=_to_descriptor(waveform),
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="range_profiles")


def test_HannWindowLowersSidelobesKeepsPeak():
    waveform = SimWaveform()
    desc = _to_descriptor(waveform)
    hann = WindowSpec()
    hann.type = WindowType.HANN

    off_bin = simulate(_make_scene(_range_at_bin(30.5)))
    rect = _run_complex(off_bin, waveform=desc)
    hann_run = _run_complex(off_bin, waveform=desc, range_window=hann)

    rect_mag = _slice_abs(rect["range_profiles"], 0, 0)
    hann_mag = _slice_abs(hann_run["range_profiles"], 0, 0)
    peak = _argmax(rect_mag)
    assert abs(_argmax(hann_mag) - peak) <= 1

    rect_sidelobe = float(np.max(rect_mag[peak + 4 : min(peak + 16, K_SAMPLES)]))
    hann_sidelobe = float(np.max(hann_mag[peak + 4 : min(peak + 16, K_SAMPLES)]))
    assert hann_sidelobe < 0.5 * rect_sidelobe

    on_bin = simulate(_make_scene(_range_at_bin(30.0)))
    rect_on = _run_complex(on_bin, waveform=desc)
    hann_on = _run_complex(on_bin, waveform=desc, range_window=hann)
    rect_on_mag = _slice_abs(rect_on["range_profiles"], 0, 0)
    hann_on_mag = _slice_abs(hann_on["range_profiles"], 0, 0)
    rect_peak = rect_on_mag[30]
    hann_peak = hann_on_mag[30]
    assert hann_peak == pytest.approx(rect_peak, abs=0.03 * rect_peak)


def test_NfftOversamplingInterpolatesGrid():
    waveform = SimWaveform()
    target_range = _range_at_bin(30.0)
    sim = simulate(_make_scene(target_range))
    desc = _to_descriptor(waveform)
    nfft = 2 * K_SAMPLES

    base = _run_complex(sim, waveform=desc)
    oversampled = _run_complex(sim, waveform=desc, nfft=nfft)

    assert oversampled["range_profiles"].shape == (K_CHIRPS, nfft)
    assert oversampled["range_bins"].shape == (nfft,)

    magnitude = _slice_abs(oversampled["range_profiles"], 0, 0)
    peak = _argmax(magnitude)
    assert oversampled["range_bins"][peak] == pytest.approx(
        target_range, abs=_bin_width(K_SAMPLES)
    )
    assert oversampled["range_bins"][1] == pytest.approx(
        0.5 * base["range_bins"][1], abs=1e-4 * base["range_bins"][1]
    )


def test_LeadingBatchAxesRoundTrip():
    waveform = SimWaveform()
    target_range = _range_at_bin(30.0)
    scene = _make_scene(target_range)
    scene.rx_positions = [(0.0, 0.0, 0.0), (0.0, 0.0019, 0.0)]
    sim = simulate(scene)
    assert sim.shape == (2, K_CHIRPS, K_SAMPLES)

    out = _run_complex(sim, waveform=_to_descriptor(waveform))
    profiles = out["range_profiles"]
    assert profiles.shape == (2, K_CHIRPS, K_SAMPLES)

    for chirp in range(K_CHIRPS):
        rx0 = _slice_abs(profiles, 0, chirp)
        rx1 = _slice_abs(profiles, 1, chirp)
        assert np.max(np.abs(rx0 - rx1)) < 1e-3, f"chirp {chirp}"


def test_LiveWindowOverrideAppliesPerTick():
    waveform = SimWaveform()
    desc = _to_descriptor(waveform)
    sim = simulate(_make_scene(_range_at_bin(30.5)))
    host = sim.cube.astype(np.complex64).reshape(sim.shape)
    frame = single_map(to_device_array(host), "adc_cube")
    frames = [frame, frame, frame]

    reference = _run_complex(sim, waveform=desc)
    rect_ref = _slice_abs(reference["range_profiles"], 0, 0)

    hann = WindowSpec()
    hann.type = WindowType.HANN

    def make_op(app):
        return FMCWRangeProcessorOpF32(app, waveform=desc, name="op_under_test")

    captured = run_operator_multi(
        make_op,
        frames,
        input_port="adc_cube",
        output_ports=("range_profiles", "range_bins"),
        overrides={"range_window": [hann, WindowSpec()]},
    )
    assert len(captured["range_profiles"]) == 3
    assert len(captured["range_bins"]) == 3

    tick0 = _slice_abs(captured["range_profiles"][0], 0, 0)
    tick1 = _slice_abs(captured["range_profiles"][1], 0, 0)
    tick2 = _slice_abs(captured["range_profiles"][2], 0, 0)
    peak = float(rect_ref[_argmax(rect_ref)])
    diff_ref0 = float(np.max(np.abs(rect_ref - tick0)))
    diff_ref1 = float(np.max(np.abs(rect_ref - tick1)))
    diff12 = float(np.max(np.abs(tick1 - tick2)))
    assert diff_ref0 > 0.01 * peak
    assert diff_ref1 <= 1e-5 * peak
    assert diff12 <= 1e-5 * peak


def test_FloatAndDoubleInstantiationsAgree():
    waveform = SimWaveform()
    sim = simulate(_make_scene(_range_at_bin(30.0)))
    desc = _to_descriptor(waveform)

    run_f = _run_complex(sim, waveform=desc, op_cls=FMCWRangeProcessorOpF32)
    run_d = _run_complex(sim, waveform=desc, op_cls=FMCWRangeProcessorOpF64)

    mag_f = _slice_abs(run_f["range_profiles"], 0, 0)
    mag_d = _slice_abs(run_d["range_profiles"], 0, 0)
    peak = float(mag_d[_argmax(mag_d)])
    assert _argmax(mag_f) == _argmax(mag_d)
    for i in range(K_SAMPLES):
        assert mag_f[i] == pytest.approx(mag_d[i], abs=1e-4 * peak), f"bin {i}"
