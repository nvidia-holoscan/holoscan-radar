# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Python facade configuration and native GPU Tensor transport together."""

import cupy as cp
import holoscan_radar as radar
import numpy as np
import pytest

import holoscan as hs
from holoscan.testing import FrameRelease, OperatorTestHarness, TestRunOptions


class _CapturedOperator:
    """Capture the public facade's native registration and serialized parameters."""

    def op(self, type_name, name, **params):
        self.type_name = type_name
        self.name = name
        self.params = params
        return self


def _run_until_output(harness, port, count):
    with harness.run_async(TestRunOptions(expect_stop=True)) as running:
        assert running.wait_for_output(port, count, timeout_ms=10_000)
        running.request_stop()
        return running.finish()


def _run_ordered_frames(harness, port, count):
    with harness.run_async(TestRunOptions(frames=FrameRelease.MANUAL, expect_stop=True)) as running:
        for frame in range(1, count + 1):
            running.release_frame(timeout_ms=10_000)
            if not running.wait_for_output(port, frame, timeout_ms=10_000):
                running.request_stop()
                failed = running.finish(check=False)
                raise AssertionError(failed.describe_failure())
        running.request_stop()
        return running.finish()


@pytest.mark.parametrize(
    "operator,dtype,bits",
    [
        (radar.ClutterFilterOpF32, np.complex64, 64),
        (radar.ClutterFilterOpF64, np.complex128, 128),
    ],
    ids=["float32", "float64"],
)
def test_python_binding_moves_gpu_tensors_and_applies_params(operator, dtype, bits):
    graph = _CapturedOperator()
    handle = operator(
        graph,
        "clutter",
        extents={"max_chirps": 4, "max_range_bins": 2},
        method="single_delay",
    )
    assert handle.name == "clutter"

    samples = np.arange(8, dtype=np.float64).reshape(4, 2).astype(dtype)
    samples += 1j
    tensor = hs.Tensor.as_tensor(cp.asarray(samples))
    with OperatorTestHarness(graph.type_name, params=graph.params) as harness:
        harness.set_input_source_tensor_spec(
            "range_profiles",
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(5, bits, 1),
                rank=2,
                bounds=hs.tensor.TensorBounds(max_byte_span=samples.nbytes),
            ),
        )
        harness.set_input_source_device("range_profiles", hs.execution.DeviceId(0))
        harness.set_output_device("range_profiles_out", hs.execution.DeviceId(0))
        harness.push("range_profiles", tensor)
        result = _run_until_output(harness, "range_profiles_out", 1)

    output = result.snapshot("range_profiles_out")
    assert output.shape == [3, 2]
    assert output.source_device.device_id == 0
    np.testing.assert_allclose(
        np.frombuffer(output.bytes, dtype=dtype).reshape(output.shape),
        np.diff(samples, axis=0),
    )


@pytest.mark.parametrize("cache_budget_bytes", [0, 768, 67108864, 4294967296])
def test_python_binding_live_nfft_applies_for_one_tick(cache_budget_bytes):
    graph = _CapturedOperator()
    radar.FMCWDopplerProcessorOpF32(
        graph,
        "doppler",
        extents={"max_chirps": 4, "max_range_bins": 2, "max_doppler_bins": 8},
        nfft=4,
        cache_budget_bytes=cache_budget_bytes,
    )
    samples = np.ones((4, 2), dtype=np.complex64)
    tensor = hs.Tensor.as_tensor(cp.asarray(samples))
    with OperatorTestHarness(graph.type_name, params=graph.params, outputs="rd_maps") as harness:
        harness.set_input_source_tensor_spec(
            "range_profiles",
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(5, 64, 1),
                rank=2,
                bounds=hs.tensor.TensorBounds(max_byte_span=samples.nbytes),
            ),
        )
        harness.set_input_source_device("range_profiles", hs.execution.DeviceId(0))
        harness.set_output_device("rd_maps", hs.execution.DeviceId(0))
        harness.set_output_device("velocity_bins", hs.execution.DeviceId(0))
        harness.capture_snapshot("rd_maps")
        harness.push_frame({"range_profiles": tensor}, capture_timestamp_ns=100)
        harness.push_frame([("nfft", 8), ("range_profiles", tensor)], capture_timestamp_ns=200)
        harness.push_frame({"range_profiles": tensor}, capture_timestamp_ns=300)
        result = _run_ordered_frames(harness, "rd_maps", 3)

    assert [snapshot.shape for snapshot in result.snapshots("rd_maps")] == [
        [4, 2],
        [8, 2],
        [4, 2],
    ]
