# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Radar-flavored Holoscan Application helpers for operator unit tests.

Mirrors ``tests/cpp/utils/sdk_harness.hpp`` and ``tensor_fixtures.hpp``.
Holoscan SDK 4.3 ships ``OperatorTestHarness`` for C++ only, so Python tests
compose tiny ``Application`` graphs with source / capture operators instead.

Device tensors prefer CuPy when available (zero-copy emit into C++ operators
via DLPack). Without CuPy, arrays are wrapped with ``holoscan.as_tensor``
(host memory — operators that call ``require_cuda_accessible`` will reject
them).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - exercised only in CPU-only envs
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

from holoscan import as_tensor
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

# ---------------------------------------------------------------------------
# Tensor factories (tensor_fixtures.hpp)
# ---------------------------------------------------------------------------


def to_device_array(host: Any, dtype: np.dtype | type | None = None) -> Any:
    """Upload host samples as a device array (CuPy) or holoscan Tensor.

    ``host`` may be a numpy array or any array-like. Complex dtypes must be
    specified explicitly when casting (e.g. ``np.complex64``).

    Zero-size arrays use ``make_device_tensor`` so the result is tagged
    ``kDLCUDA`` (CuPy→Python TensorMap path mislabels empty buffers as CPU).
    """
    arr = np.asarray(host)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if arr.size == 0:
        # Import lazily so harness can load before the radar package is built.
        from holoscan.radar import make_device_tensor

        return make_device_tensor(list(arr.shape), arr.dtype)
    if _HAS_CUPY:
        return cp.asarray(arr)
    return as_tensor(arr)


def _numpy_dtype_from_dlpack(dt: Any) -> np.dtype:
    """Map a Holoscan / DLPack dtype descriptor to numpy."""
    code = int(getattr(dt, "code", 2))
    bits = int(getattr(dt, "bits", 32))
    if code == 0:  # kDLInt
        return np.dtype(f"int{bits}")
    if code == 1:  # kDLUInt
        return np.dtype(f"uint{bits}")
    if code == 6:  # kDLBool
        return np.bool_
    if code == 5:  # kDLComplex
        return np.dtype(f"complex{bits * 2}")
    return np.dtype(f"float{bits}")


def to_host_array(tensor_like: Any) -> np.ndarray:
    """Copy a received tensor / CuPy / numpy object to a contiguous numpy array."""
    shape = getattr(tensor_like, "shape", None)
    if shape is not None:
        try:
            if int(np.prod(tuple(shape))) == 0:
                dt = getattr(tensor_like, "dtype", None)
                np_dtype = (
                    _numpy_dtype_from_dlpack(dt)
                    if dt is not None
                    else np.float64
                )
                return np.empty(tuple(shape), dtype=np_dtype)
        except (TypeError, ValueError):
            pass

    if hasattr(tensor_like, "__dlpack__"):
        try:
            return np.ascontiguousarray(np.from_dlpack(tensor_like))
        except (TypeError, RuntimeError, BufferError, ValueError):
            pass
    if _HAS_CUPY:
        try:
            return np.ascontiguousarray(cp.asnumpy(cp.asarray(tensor_like)))
        except (TypeError, RuntimeError, ValueError, BufferError):
            pass
    return np.ascontiguousarray(np.asarray(tensor_like))


def single_map(tensor: Any, key: str) -> dict[str, Any]:
    """Single-entry TensorMap (the module's port payload convention)."""
    return {key: tensor}


# ---------------------------------------------------------------------------
# Capture store (sdk_harness.hpp TensorMapCapture)
# ---------------------------------------------------------------------------


class TensorMapCapture:
    """Host-side store for TensorMap output ports.

    Each received single-entry map contributes one numpy ndarray to
    ``frames`` (the map's sole tensor, copied to host).
    """

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.keys: list[str] = []

    def record(self, tensor_map: Mapping[str, Any]) -> None:
        if len(tensor_map) != 1:
            raise RuntimeError(
                f"TensorMapCapture: expected a single-entry TensorMap, "
                f"got {len(tensor_map)} entries"
            )
        key, value = next(iter(tensor_map.items()))
        self.keys.append(key)
        self.frames.append(to_host_array(value))


# ---------------------------------------------------------------------------
# Source / sink operators
# ---------------------------------------------------------------------------


class _TensorMapSourceOp(Operator):
    """Emit one TensorMap (dict) per tick; CountCondition = frame count."""

    def __init__(
        self, fragment, *args, frames: Sequence[Mapping[str, Any]], **kwargs
    ):
        self._frames = list(frames)
        self._index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("output")

    def compute(self, op_input, op_output, context) -> None:
        frame = self._frames[self._index]
        self._index += 1
        op_output.emit(dict(frame), "output")


def infer_emit_type_name(value: Any) -> str | None:
    """Map a Python override value to a Holoscan EmitterReceiverRegistry name.

    Built-ins use the SDK's registered C++ spellings (``bool``, ``int64_t``,
    ``std::vector<double>``, …). Radar enums/descriptors use the names from
    ``radar_common.register_types``.
    """
    # bool is a subclass of int — check first.
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64_t"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "std::string"
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], bool):
            return "std::vector<bool>"
        if isinstance(value[0], int):
            return "std::vector<int64_t>"
        if isinstance(value[0], float):
            return "std::vector<double>"
        if isinstance(value[0], str):
            return "std::vector<std::string>"
        return None
    # pybind-bound radar types: class name matches register_types labels.
    name = type(value).__name__
    return name or None


class _ValueSourceOp(Operator):
    """Emit one typed value per tick (live override sources)."""

    def __init__(
        self,
        fragment,
        *args,
        values: Sequence[Any],
        emit_type: str | None = None,
        **kwargs,
    ):
        self._values = list(values)
        self._index = 0
        if emit_type is not None:
            self._emit_type = emit_type
        elif self._values:
            self._emit_type = None
            for v in self._values:
                inferred = infer_emit_type_name(v)
                if inferred is not None:
                    self._emit_type = inferred
                    break
        else:
            self._emit_type = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("output")

    def compute(self, op_input, op_output, context) -> None:
        value = self._values[self._index]
        self._index += 1
        if self._emit_type is not None:
            # 3rd arg selects EmitterReceiverRegistry entry so C++ ops
            # receive<T>() get a real T, not a Python-object shared_ptr.
            op_output.emit(value, "output", self._emit_type)
        else:
            op_output.emit(value, "output")


class _TensorMapCaptureOp(Operator):
    """Drain a TensorMap output port into a TensorMapCapture."""

    def __init__(self, fragment, *args, capture: TensorMapCapture, **kwargs):
        self._capture = capture
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        message = op_input.receive("input")
        self._capture.record(message)


# ---------------------------------------------------------------------------
# Wiring helpers (sdk_harness.hpp)
# ---------------------------------------------------------------------------


def wire_data_source(
    app: Application,
    op: Operator,
    port: str,
    frames: Sequence[Mapping[str, Any]],
) -> Operator:
    """Wire a TensorMap data source (CountCondition = frame count)."""
    source = _TensorMapSourceOp(
        app,
        CountCondition(app, count=len(frames)),
        frames=frames,
        name=f"{port}_source",
    )
    app.add_flow(source, op, {("output", port)})
    return source


def wire_leading_override(
    app: Application,
    op: Operator,
    port: str,
    values: Sequence[Any],
    *,
    emit_type: str | None = None,
) -> Operator:
    """Wire an override source that emits only on the leading ticks.

    CountCondition equals ``len(values)``, then the source stays silent while
    the data source keeps ticking — sticky-override semantics.

    Values are emitted as registered C++ types (``emit`` 3rd arg) so wrapped
    C++ operators' ``receive<T>`` override ports apply them.
    """
    source = _ValueSourceOp(
        app,
        CountCondition(app, count=len(values)),
        values=values,
        emit_type=emit_type,
        name=f"{port}_source",
    )
    app.add_flow(source, op, {("output", port)})
    return source

def wire_capture_sink(
    app: Application,
    op: Operator,
    port: str,
    capture: TensorMapCapture,
) -> Operator:
    """Wire a capturing sink onto an output port."""
    sink = _TensorMapCaptureOp(
        app, capture=capture, name=f"{port}_sink"
    )
    app.add_flow(op, sink, {(port, "input")})
    return sink


# ---------------------------------------------------------------------------
# One-shot / multi-tick runner
# ---------------------------------------------------------------------------


def wire_null_sink(app: Application, op: Operator, port: str) -> Operator:
    """Drain an output port without host conversion (bool masks, unused ports).

    Holoscan SDK 4.3 rejects NumPy/PyTensor conversion of bool tensors; use
    this (or ``radar_app.NullSinkOp``) instead of ``wire_capture_sink`` for
    ``detection_mask`` / ``is_valid``.
    """
    from holoscan.radar import NullSinkOp

    sink = NullSinkOp(app, name=f"{port}_null_sink")
    app.add_flow(op, sink, {(port, "input")})
    return sink


def run_operator(
    make_op: Callable[[Application], Operator],
    frames: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    input_port: str = "range_profiles",
    output_port: str = "range_profiles",
    overrides: Mapping[str, Sequence[Any]] | None = None,
    extra_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    drain_ports: Sequence[str] = (),
) -> list[np.ndarray]:
    """Run ``make_op`` over one or more TensorMap frames; return host outputs.

    Parameters
    ----------
    make_op:
        ``make_op(app) -> Operator`` under test (already constructed with
        parameters).
    frames:
        A single TensorMap dict, or a sequence of them (one per tick).
    input_port / output_port:
        Data-plane port names on the operator under test.
    overrides:
        Optional map of override-port name → value sequence. Each sequence
        may be shorter than ``frames`` (sticky override / leading-only).
    extra_inputs:
        Additional TensorMap input ports → frame sequences (same length as
        ``frames``, or broadcast a length-1 sequence).
    drain_ports:
        Output ports to NullSink (no capture; for bool / unused ports).
    """
    if isinstance(frames, Mapping):
        frame_list: list[Mapping[str, Any]] = [frames]
    else:
        frame_list = list(frames)
    if not frame_list:
        raise ValueError("run_operator: need at least one input frame")

    capture = TensorMapCapture()

    class _App(Application):
        def compose(self) -> None:
            op = make_op(self)
            wire_data_source(self, op, input_port, frame_list)
            if extra_inputs:
                for port, port_frames in extra_inputs.items():
                    pf = list(port_frames)
                    if len(pf) == 1 and len(frame_list) > 1:
                        pf = pf * len(frame_list)
                    wire_data_source(self, op, port, pf)
            if overrides:
                for port, values in overrides.items():
                    wire_leading_override(self, op, port, values)
            wire_capture_sink(self, op, output_port, capture)
            for port in drain_ports:
                wire_null_sink(self, op, port)

    _App().run()
    return capture.frames


def run_operator_multi(
    make_op: Callable[[Application], Operator],
    frames: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    input_port: str,
    output_ports: Sequence[str],
    overrides: Mapping[str, Sequence[Any]] | None = None,
    extra_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    drain_ports: Sequence[str] = (),
) -> dict[str, list[np.ndarray]]:
    """Like ``run_operator`` but capture several TensorMap output ports."""
    if isinstance(frames, Mapping):
        frame_list: list[Mapping[str, Any]] = [frames]
    else:
        frame_list = list(frames)
    if not frame_list:
        raise ValueError("run_operator_multi: need at least one input frame")
    if not output_ports:
        raise ValueError("run_operator_multi: need at least one output port")

    captures = {port: TensorMapCapture() for port in output_ports}

    class _App(Application):
        def compose(self) -> None:
            op = make_op(self)
            wire_data_source(self, op, input_port, frame_list)
            if extra_inputs:
                for port, port_frames in extra_inputs.items():
                    pf = list(port_frames)
                    if len(pf) == 1 and len(frame_list) > 1:
                        pf = pf * len(frame_list)
                    wire_data_source(self, op, port, pf)
            if overrides:
                for port, values in overrides.items():
                    wire_leading_override(self, op, port, values)
            for port, capture in captures.items():
                wire_capture_sink(self, op, port, capture)
            for port in drain_ports:
                wire_null_sink(self, op, port)

    _App().run()
    return {port: cap.frames for port, cap in captures.items()}
