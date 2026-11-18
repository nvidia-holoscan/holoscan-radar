# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public SAR/I/O factories across the native GPU boundary."""

import os
from pathlib import Path

import cupy as cp
import holoscan_radar as radar
import numpy as np
import pytest

import holoscan as hs
from holoscan.testing import OperatorTestHarness, TestRunOptions


class _Construction:
    def op(self, type_name, name, **params):
        self.type_name, self.params = type_name, params
        return self


def _configure(operator, **params):
    captured = _Construction()
    operator(captured, "test", **params)
    return captured


def _tensor_inputs(harness, arrays):
    tensors = {}
    for port, array in arrays.items():
        if not isinstance(array, np.ndarray):
            tensors[port] = array
            continue
        tensors[port] = hs.Tensor.as_tensor(cp.asarray(array))
        code = (
            5
            if array.dtype.kind == "c"
            else 2 if array.dtype.kind == "f" else 1 if array.dtype.kind == "u" else 0
        )
        harness.set_input_source_tensor_spec(
            port,
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(code, array.dtype.itemsize * 8, 1),
                rank=array.ndim,
                bounds=hs.tensor.TensorBounds(max_byte_span=array.nbytes),
            ),
        )
        harness.set_input_source_device(port, hs.execution.DeviceId(0))
    return tensors


def _capture(captured, arrays, output, *, outputs=None, extra_devices=()):
    with OperatorTestHarness(
        captured.type_name, params=captured.params, outputs=outputs or output
    ) as harness:
        tensors = _tensor_inputs(harness, arrays)
        for name in (*(outputs or (output,)), *extra_devices):
            harness.set_output_device(name, hs.execution.DeviceId(0))
        harness.push_frame(tensors, capture_timestamp_ns=100)
        with harness.run_async(TestRunOptions(expect_stop=True)) as running:
            observed = all(
                running.wait_for_output(name, 1, timeout_ms=10_000)
                for name in (outputs or (output,))
            )
            running.request_stop()
            result = running.finish()
        assert observed, result.describe_failure()
        return result


@pytest.mark.parametrize("suffix,dtype", [("F32", np.complex64), ("F64", np.complex128)])
def test_range_compression_centers_constant_spectrum(suffix, dtype):
    captured = _configure(
        getattr(radar, f"SARRangeCompressionOp{suffix}"),
        extents={"max_pulses": 1, "max_samples_per_pulse": 8, "max_range_bins": 8},
    )
    result = _capture(
        captured,
        {"phase_history": np.ones((1, 8), dtype=dtype)},
        "range_profiles",
        extra_devices=("range_bins",),
    )
    snapshot = result.snapshot("range_profiles")
    image = np.frombuffer(snapshot.bytes, dtype=dtype).reshape(snapshot.shape)
    expected = np.zeros((1, 8), dtype=dtype)
    expected[0, 4] = 1
    np.testing.assert_allclose(image, expected, atol=1e-7)


@pytest.mark.parametrize(
    "suffix,real,complex_type",
    [("F32", np.float32, np.complex64), ("F64", np.float64, np.complex128)],
)
@pytest.mark.parametrize("sense,sign", [("add", 1), ("subtract", -1)])
def test_phase_correction_preserves_sign(suffix, real, complex_type, sense, sign):
    phase = np.array([0, np.pi / 2], dtype=real)
    captured = _configure(
        getattr(radar, f"PhaseCorrectionOp{suffix}"),
        extents={"max_pulses": 2, "max_range_bins": 3},
        sense=sense,
    )
    result = _capture(
        captured,
        {"range_profiles": np.ones((2, 3), dtype=complex_type), "phase_correction": phase},
        "range_profiles_out",
    )
    np.testing.assert_allclose(
        np.frombuffer(result.snapshot("range_profiles_out").bytes, dtype=complex_type).reshape(
            2, 3
        ),
        np.repeat(np.exp(sign * 1j * phase)[:, None], 3, axis=1),
        atol=1e-6,
    )


_PAIRS = [
    ("F32GeomF32", np.complex64, np.float32),
    ("F32GeomF64", np.complex64, np.float64),
    ("F64GeomF64", np.complex128, np.float64),
]


@pytest.mark.parametrize("suffix,samples,geometry", _PAIRS)
def test_backprojection_point_target_uses_explicit_geometry_precision(suffix, samples, geometry):
    impulse = np.zeros((1, 64), dtype=samples)
    impulse[0, 32] = 1
    captured = _configure(
        getattr(radar, f"BackprojectionOp{suffix}"),
        extents={"max_pulses": 1, "max_range_bins": 64, "max_image_v": 5, "max_image_u": 5},
        coordinate_frame="scene_local",
        sample_geometry={
            "origin": [-2, -2, 0],
            "axis_u": [1, 0, 0],
            "axis_v": [0, 1, 0],
            "spacing": [1, 1],
            "counts": [5, 5],
        },
        backend="nvradar_fp64_reference",
        range_bin_spacing=0.25,
        range_bin_origin=-8,
        reference_frequency=9.6e9,
    )
    result = _capture(
        captured,
        {
            "control": np.array([1], dtype=np.uint8),
            "range_profiles": impulse,
            "positions": np.array([[0, 0, 100]], dtype=geometry),
            "reference_position": np.zeros((1, 3), dtype=geometry),
        },
        "image",
    )
    snapshot = result.snapshot("image")
    image = np.frombuffer(snapshot.bytes, dtype=samples).reshape(snapshot.shape)
    assert image.shape == (5, 5)
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image[2, 2], 1, atol=1e-5)


@pytest.mark.parametrize("suffix,samples,geometry", _PAIRS)
def test_aperture_manager_preserves_tensor_precisions_and_closes_stream(suffix, samples, geometry):
    captured = _configure(
        getattr(radar, f"ApertureManagerOp{suffix}"),
        extents={"max_pulses": 2, "max_range_bins": 8, "max_image_v": 2, "max_image_u": 2},
        emit_mode="reset",
    )
    profiles = np.arange(16).reshape(2, 8).astype(samples)
    positions = np.arange(6).reshape(2, 3).astype(geometry)
    result = _capture(
        captured,
        {
            "range_profiles": profiles,
            "positions": positions,
            "reference_position": positions,
            "stream_boundary": 2,
        },
        "range_profiles_out",
        outputs=("range_profiles_out", "positions_out", "reference_position_out", "control_out"),
        extra_devices=("weight", "sample_points_out"),
    )
    for port, expected in [
        ("range_profiles_out", profiles),
        ("positions_out", positions),
        ("reference_position_out", positions),
    ]:
        snapshot = result.snapshot(port)
        np.testing.assert_array_equal(
            np.frombuffer(snapshot.bytes, dtype=expected.dtype).reshape(snapshot.shape), expected
        )
    assert np.frombuffer(result.snapshot("control_out").bytes, dtype=np.uint8)[0] == 1


@pytest.mark.parametrize("suffix,samples,geometry", _PAIRS)
def test_cphd_source_exposes_both_precision_axes(suffix, samples, geometry):
    path = Path(os.environ["HOLOSCAN_RADAR_TEST_BUILD_DIR"]) / "tests/cpp/generated_sar_smoke.cphd"
    captured = _configure(
        getattr(radar, f"CPHDReaderOp{suffix}"),
        extents={"max_pulses": 3, "max_samples_per_pulse": 64},
        path=str(path),
        coordinate_frame="ecef",
        pulses_per_tick=3,
        io_chunk_bytes=8,
    )
    with OperatorTestHarness(
        captured.type_name,
        params=captured.params,
        outputs=["phase_history", "positions", "reference_position", "stream_boundary"],
    ) as harness:
        for port in (
            "phase_history",
            "positions",
            "reference_position",
            "sample_axis",
            "pulse_times",
            "velocities",
        ):
            harness.set_output_device(port, hs.execution.DeviceId(0))
        with harness.run_async(TestRunOptions(expect_stop=True)) as running:
            observed = all(
                running.wait_for_output(port, 2, timeout_ms=10_000)
                for port in ("phase_history", "positions", "reference_position", "stream_boundary")
            )
            running.request_stop()
            result = running.finish()
        assert observed, result.describe_failure()
    phase, position = result.snapshots("phase_history")[0], result.snapshots("positions")[0]
    assert phase.dtype.bits == np.dtype(samples).itemsize * 8
    assert position.dtype.bits == np.dtype(geometry).itemsize * 8
    assert result.values("stream_boundary") == [0, 2]


@pytest.mark.parametrize("suffix,dtype", [("F32", np.complex64), ("F64", np.complex128)])
def test_autofocus_flat_image_reports_refusal(suffix, dtype):
    captured = _configure(
        getattr(radar, f"AutofocusOp{suffix}"),
        extents={"max_image_v": 4, "max_image_u": 8},
        azimuth_axis="u",
        num_iterations=2,
        min_scatterers=32,
    )
    image = np.ones((4, 8), dtype=dtype)
    outputs = ("image_out", "phase_correction", "sharpness", "termination", "iterations")
    result = _capture(captured, {"image": image}, "iterations", outputs=outputs)
    np.testing.assert_array_equal(
        np.frombuffer(result.snapshot("image_out").bytes, dtype=dtype).reshape(image.shape), image
    )
    assert np.frombuffer(result.snapshot("iterations").bytes, dtype=np.int32)[0] == 0


@pytest.mark.parametrize("suffix,dtype", [("F32", np.complex64), ("F64", np.complex128)])
def test_sar_live_nfft_is_tick_local(suffix, dtype):
    from holoscan.testing import FrameRelease

    captured = _configure(
        getattr(radar, f"SARRangeCompressionOp{suffix}"),
        extents={"max_pulses": 1, "max_samples_per_pulse": 8, "max_range_bins": 16},
        nfft=8,
    )
    with OperatorTestHarness(
        captured.type_name, params=captured.params, outputs="range_profiles"
    ) as harness:
        tensors = _tensor_inputs(harness, {"phase_history": np.ones((1, 8), dtype=dtype)})
        harness.set_output_device("range_profiles", hs.execution.DeviceId(0))
        harness.set_output_device("range_bins", hs.execution.DeviceId(0))
        harness.push_frame(tensors, capture_timestamp_ns=100)
        harness.push_frame([("nfft", 16), *tensors.items()], capture_timestamp_ns=200)
        harness.push_frame(tensors, capture_timestamp_ns=300)
        with harness.run_async(
            TestRunOptions(frames=FrameRelease.MANUAL, expect_stop=True)
        ) as running:
            for frame in range(1, 4):
                running.release_frame(timeout_ms=10_000)
                assert running.wait_for_output("range_profiles", frame, timeout_ms=10_000)
            running.request_stop()
            result = running.finish()
    assert [snapshot.shape for snapshot in result.snapshots("range_profiles")] == [
        [1, 8],
        [1, 16],
        [1, 8],
    ]


def test_sar_enum_ports_remain_unavailable_but_static_values_decode():
    range_type = hs.registry.resolve("holoscan.radar.SARRangeCompressionOpF32")
    bp_type = hs.registry.resolve("holoscan.radar.BackprojectionOpF32GeomF64")
    assert "fft_norm" not in range_type.ports
    assert "nfft" in range_type.ports
    assert "interpolation" not in bp_type.ports
    assert "sample_geometry" in bp_type.ports
    graph = hs.Graph("static_sar_enums")
    radar.SARRangeCompressionOpF32(
        graph,
        "range",
        extents={"max_pulses": 1, "max_samples_per_pulse": 8, "max_range_bins": 8},
        fft_norm="ortho",
    )
    with pytest.raises(hs.errors.InvalidArgumentError):
        radar.BackprojectionOpF32GeomF64(
            graph,
            "bad_geometry",
            extents={"max_pulses": 1, "max_range_bins": 8, "max_image_u": 2, "max_image_v": 2},
            sample_geometry={"counts": [-1, 2]},
        )
