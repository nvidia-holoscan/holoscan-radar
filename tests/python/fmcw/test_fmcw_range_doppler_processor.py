# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/fmcw/test_fmcw_range_doppler_processor.cpp.

Peak/axis checks and float-tolerance split-chain parity; C++ owns bit-exact
fused-vs-split memcmp coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.bin_model import (  # noqa: E402
    K_C,
    expected_doppler_bin,
    expected_range_bin,
    range_of_bin,
)
from utils.harness import (  # noqa: E402
    run_operator,
    run_operator_multi,
    single_map,
    to_device_array,
)
from utils.point_target_sim import PointTarget, SimScene, SimWaveform, simulate  # noqa: E402

from holoscan.radar import (  # noqa: E402
    ClutterFilterMethod,
    ClutterFilterOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeDopplerProcessorOp,
    FMCWRangeDopplerProcessorOpF32,
    FMCWRangeDopplerProcessorOpF64,
    FMCWRangeProcessorOpF32,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    WindowSpec,
)

pytestmark = pytest.mark.requires_gpu

K_CHIRPS = 32
K_SAMPLES = 128


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for FMCWRangeDopplerProcessorOp tests: {exc}")


def _range_bin_width(nfft: int) -> float:
    waveform = SimWaveform()
    return range_of_bin(1, waveform.slope(), waveform.sample_rate, nfft, K_C)


def _range_at_bin(bin_idx: float) -> float:
    return bin_idx * _range_bin_width(K_SAMPLES)


def _to_descriptor(waveform: SimWaveform) -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = waveform.carrier_frequency
    spec.bandwidth = waveform.bandwidth
    spec.chirp_duration = waveform.chirp_duration
    spec.sample_rate = waveform.sample_rate
    spec.chirp_repetition_interval = waveform.chirp_repetition_interval
    return FMCWWaveformDescriptor(spec)


def _parity_scene(*, dechirped: bool = True) -> SimScene:
    return SimScene(
        num_chirps=K_CHIRPS,
        samples_per_chirp=K_SAMPLES,
        dechirped=dechirped,
        targets=[
            PointTarget(position=(_range_at_bin(20.0), 0.0, 0.0)),
            PointTarget(
                position=(_range_at_bin(40.0), 0.0, 0.0),
                velocity=(5.0, 0.0, 0.0),
            ),
        ],
    )


_PARITY_SIM = simulate(_parity_scene())
_PARITY_SIM_PRE_MIXER = simulate(_parity_scene(dechirped=False))


def _to_real_if(cube: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(cube).reshape(shape).real.astype(np.float32)


def _derive_radial_velocity(target: PointTarget) -> float:
    rng = math.sqrt(sum(x * x for x in target.position))
    look = tuple(x / rng for x in target.position)
    return sum(v * lk for v, lk in zip(target.velocity, look, strict=True))


@dataclass
class ParityConfig:
    clutter: ClutterFilterMethod = ClutterFilterMethod.NONE
    nfft_range: int | None = None
    nfft_doppler: int | None = None
    range_window: WindowSpec | None = None
    doppler_window: WindowSpec | None = None
    apply_dechirp: bool = False


def _adc_frame(sim, *, real_if: bool = False, op_cls=FMCWRangeDopplerProcessorOpF32) -> dict:
    if real_if:
        host = _to_real_if(sim.cube, sim.shape)
    else:
        ctype = np.complex128 if op_cls is FMCWRangeDopplerProcessorOpF64 else np.complex64
        host = sim.cube.astype(ctype).reshape(sim.shape)
    return single_map(to_device_array(host), "adc_cube")


def _profiles_frame(profiles: np.ndarray) -> dict:
    return single_map(to_device_array(profiles.astype(np.complex64)), "range_profiles")


def _run_fused(
    frame: dict,
    cfg: ParityConfig,
    waveform: FMCWWaveformDescriptor,
    *,
    op_cls=FMCWRangeDopplerProcessorOpF32,
):
    kwargs = {
        "apply_dechirp": cfg.apply_dechirp,
        "clutter_filter": cfg.clutter,
        "waveform": waveform,
        "name": "op_under_test",
    }
    if cfg.range_window is not None:
        kwargs["range_window"] = cfg.range_window
    if cfg.doppler_window is not None:
        kwargs["doppler_window"] = cfg.doppler_window
    if cfg.nfft_range is not None:
        kwargs["nfft_range"] = cfg.nfft_range
    if cfg.nfft_doppler is not None:
        kwargs["nfft_doppler"] = cfg.nfft_doppler

    def make_op(app):
        return op_cls(app, **kwargs)

    captured = run_operator_multi(
        make_op,
        frame,
        input_port="adc_cube",
        output_ports=("rd_maps", "range_bins", "velocity_bins"),
    )
    return {port: frames[0] for port, frames in captured.items()}


def _run_split(
    frame: dict,
    cfg: ParityConfig,
    waveform: FMCWWaveformDescriptor,
):
    range_kwargs = {
        "waveform": waveform,
        "apply_dechirp": cfg.apply_dechirp,
        "name": "range_op",
    }
    if cfg.range_window is not None:
        range_kwargs["range_window"] = cfg.range_window
    if cfg.nfft_range is not None:
        range_kwargs["nfft"] = cfg.nfft_range

    def make_range(app):
        return FMCWRangeProcessorOpF32(app, **range_kwargs)

    range_out = run_operator_multi(
        make_range,
        frame,
        input_port="adc_cube",
        output_ports=("range_profiles", "range_bins"),
    )
    profiles = range_out["range_profiles"][0]
    range_bins = range_out["range_bins"][0]

    if cfg.clutter != ClutterFilterMethod.NONE:
        prof_frame = _profiles_frame(profiles)

        def make_clutter(app):
            return ClutterFilterOpF32(
                app, method=cfg.clutter, name="clutter_op"
            )

        profiles = run_operator(
            make_clutter,
            prof_frame,
            input_port="range_profiles",
            output_port="range_profiles",
        )[0]

    doppler_kwargs = {"waveform": waveform, "name": "doppler_op"}
    if cfg.doppler_window is not None:
        doppler_kwargs["doppler_window"] = cfg.doppler_window
    if cfg.nfft_doppler is not None:
        doppler_kwargs["nfft"] = cfg.nfft_doppler

    def make_doppler(app):
        return FMCWDopplerProcessorOpF32(app, **doppler_kwargs)

    doppler_out = run_operator_multi(
        make_doppler,
        _profiles_frame(profiles),
        input_port="range_profiles",
        output_ports=("rd_maps", "velocity_bins"),
    )
    return {
        "rd_maps": doppler_out["rd_maps"][0],
        "range_bins": range_bins,
        "velocity_bins": doppler_out["velocity_bins"][0],
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    return float(np.max(diff))


def _max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def test_PeakMapsToTruthRangeAndVelocity():
    scene = SimScene(
        num_chirps=K_CHIRPS,
        samples_per_chirp=K_SAMPLES,
        targets=[
            PointTarget(
                position=(_range_at_bin(40.0), 0.0, 0.0),
                velocity=(5.0, 0.0, 0.0),
            )
        ],
    )
    sim = simulate(scene)
    target = scene.targets[0]
    target_range = math.sqrt(sum(x * x for x in target.position))
    radial_velocity = _derive_radial_velocity(target)

    waveform = _to_descriptor(scene.waveform)
    fused = _run_fused(_adc_frame(sim), ParityConfig(), waveform)

    rd = fused["rd_maps"]
    assert rd.shape == (K_CHIRPS, K_SAMPLES)

    flat = np.abs(rd.reshape(-1))
    peak_idx = int(np.argmax(flat))
    peak_doppler = peak_idx // K_SAMPLES
    peak_range = peak_idx % K_SAMPLES

    assert peak_range == expected_range_bin(
        target_range,
        scene.waveform.slope(),
        scene.waveform.sample_rate,
        K_SAMPLES,
    )
    assert peak_doppler == expected_doppler_bin(
        radial_velocity,
        scene.waveform.wavelength(),
        scene.waveform.chirp_repetition_interval,
        K_CHIRPS,
        True,
    )

    range_axis = fused["range_bins"]
    velocity_axis = fused["velocity_bins"]
    assert range_axis[peak_range] == pytest.approx(
        target_range, abs=_range_bin_width(K_SAMPLES) + 1e-9
    )
    velocity_bin_width = (
        scene.waveform.wavelength()
        / 2.0
        / scene.waveform.chirp_repetition_interval
        / float(K_CHIRPS)
    )
    assert velocity_axis[peak_doppler] == pytest.approx(
        radial_velocity, abs=velocity_bin_width + 1e-9
    )


def test_InlineClutterReducesDopplerLength():
    waveform = _to_descriptor(SimWaveform())
    frame = _adc_frame(_PARITY_SIM)
    cases = [
        (ClutterFilterMethod.MEAN_SUBTRACTION, K_CHIRPS),
        (ClutterFilterMethod.SINGLE_DELAY, K_CHIRPS - 1),
        (ClutterFilterMethod.DOUBLE_DELAY, K_CHIRPS - 2),
    ]
    for method, expected_doppler in cases:
        fused = _run_fused(frame, ParityConfig(clutter=method), waveform)
        assert fused["rd_maps"].shape == (expected_doppler, K_SAMPLES), method
        assert fused["velocity_bins"].shape == (expected_doppler,), method


def test_OptionalOutputsOnlyWithWaveform():
    frame = _adc_frame(_PARITY_SIM)

    def make_op(app):
        return FMCWRangeDopplerProcessorOp(app, name="op_under_test")

    rd = run_operator(
        make_op, frame, input_port="adc_cube", output_port="rd_maps"
    )[0]
    assert rd.shape == (K_CHIRPS, K_SAMPLES)


def test_ParityMeanSubtractionSplitChain():
    waveform = _to_descriptor(SimWaveform())
    cfg = ParityConfig(clutter=ClutterFilterMethod.MEAN_SUBTRACTION)
    frame = _adc_frame(_PARITY_SIM)
    fused = _run_fused(frame, cfg, waveform)
    split = _run_split(frame, cfg, waveform)

    reference = _max_abs(split["rd_maps"])
    assert reference > 0.0
    rel = _max_abs_diff(fused["rd_maps"], split["rd_maps"]) / reference
    assert rel < 1e-5
    np.testing.assert_array_equal(fused["range_bins"], split["range_bins"])
    np.testing.assert_allclose(
        fused["velocity_bins"], split["velocity_bins"], rtol=0, atol=0
    )


def test_ParityApplyDechirp():
    waveform = _to_descriptor(SimWaveform())
    cfg = ParityConfig(apply_dechirp=True)
    frame = _adc_frame(_PARITY_SIM_PRE_MIXER)
    fused = _run_fused(frame, cfg, waveform)
    split = _run_split(frame, cfg, waveform)
    reference = _max_abs(split["rd_maps"])
    rel = _max_abs_diff(fused["rd_maps"], split["rd_maps"]) / reference
    assert rel < 1e-5


def test_ApplyDechirpWithoutWaveformErrors():
    frame = _adc_frame(_PARITY_SIM_PRE_MIXER)

    def make_op(app):
        return FMCWRangeDopplerProcessorOp(
            app, apply_dechirp=True, name="op_under_test"
        )

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="rd_maps")


def test_ApplyDechirpWithRealInputErrors():
    frame = _adc_frame(_PARITY_SIM, real_if=True)

    def make_op(app):
        return FMCWRangeDopplerProcessorOp(
            app,
            apply_dechirp=True,
            waveform=_to_descriptor(SimWaveform()),
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="rd_maps")


def test_NfftRangeTooSmallErrors():
    frame = _adc_frame(_PARITY_SIM)

    def make_op(app):
        return FMCWRangeDopplerProcessorOp(
            app, nfft_range=K_SAMPLES // 2, name="op_under_test"
        )

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="rd_maps")


def test_NfftDopplerTooSmallErrors():
    frame = _adc_frame(_PARITY_SIM)

    def make_op(app):
        return FMCWRangeDopplerProcessorOp(
            app, nfft_doppler=K_CHIRPS // 2, name="op_under_test"
        )

    with pytest.raises(ValueError):
        run_operator(make_op, frame, input_port="adc_cube", output_port="rd_maps")

    def make_op_delay(app):
        return FMCWRangeDopplerProcessorOp(
            app,
            clutter_filter=ClutterFilterMethod.SINGLE_DELAY,
            nfft_doppler=K_CHIRPS - 2,
            name="op_under_test",
        )

    with pytest.raises(ValueError):
        run_operator(make_op_delay, frame, input_port="adc_cube", output_port="rd_maps")


def test_LiveClutterOverrideChangesShape():
    waveform = _to_descriptor(SimWaveform())
    frame = _adc_frame(_PARITY_SIM)
    frames = [frame, frame, frame]

    def make_op(app):
        return FMCWRangeDopplerProcessorOpF32(
            app, waveform=waveform, name="op_under_test"
        )

    captured = run_operator_multi(
        make_op,
        frames,
        input_port="adc_cube",
        output_ports=("rd_maps", "velocity_bins"),
        overrides={
            "clutter_filter": [
                ClutterFilterMethod.NONE,
                ClutterFilterMethod.SINGLE_DELAY,
            ]
        },
    )
    assert len(captured["rd_maps"]) == 3
    assert captured["rd_maps"][0].shape == (K_CHIRPS, K_SAMPLES)
    assert captured["rd_maps"][1].shape == (K_CHIRPS - 1, K_SAMPLES)
    assert captured["rd_maps"][2].shape == (K_CHIRPS - 1, K_SAMPLES)
    assert captured["velocity_bins"][0].shape == (K_CHIRPS,)
    assert captured["velocity_bins"][1].shape == (K_CHIRPS - 1,)
    assert captured["velocity_bins"][2].shape == (K_CHIRPS - 1,)


def test_FloatAndDoubleAgree():
    waveform = _to_descriptor(SimWaveform())
    cfg = ParityConfig(clutter=ClutterFilterMethod.MEAN_SUBTRACTION)
    frame = _adc_frame(_PARITY_SIM)
    fused_f = _run_fused(frame, cfg, waveform, op_cls=FMCWRangeDopplerProcessorOpF32)
    fused_d = _run_fused(
        _adc_frame(_PARITY_SIM, op_cls=FMCWRangeDopplerProcessorOpF64),
        cfg,
        waveform,
        op_cls=FMCWRangeDopplerProcessorOpF64,
    )
    assert fused_f["rd_maps"].shape == fused_d["rd_maps"].shape
    peak = _max_abs(fused_d["rd_maps"])
    mag_f = np.abs(fused_f["rd_maps"])
    mag_d = np.abs(fused_d["rd_maps"])
    assert np.max(np.abs(mag_f - mag_d)) < 1e-4 * peak
