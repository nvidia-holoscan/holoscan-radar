# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamping source operators for the Python performance harness."""

from __future__ import annotations

from typing import Any

from holoscan.core import Operator, OperatorSpec

from utils.perf_app import TimingCollector, now


class PerfTensorSourceOp(Operator):
    """Emit the same preallocated TensorMap every tick and record start times."""

    def __init__(
        self,
        fragment,
        *args,
        output_port: str,
        frame: dict[str, Any],
        collector: TimingCollector,
        **kwargs,
    ):
        self._output_port = output_port
        self._frame = frame
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output(self._output_port)

    def compute(self, op_input, op_output, context) -> None:
        # Start the frame clock *after* emit so CuPy→Holoscan wrap cost is not
        # attributed to the operator under test (see performance/OPEN_ITEMS.md).
        if not self._collector.first_tick_set:
            self._collector.t_first_tick = now()
            self._collector.first_tick_set = True
        op_output.emit(dict(self._frame), self._output_port)
        self._collector.starts.append(now())


class PerfDataSourceOp(Operator):
    """Emit the same preallocated rd_maps TensorMap every tick."""

    def __init__(
        self,
        fragment,
        *args,
        frame: dict[str, Any],
        collector: TimingCollector,
        **kwargs,
    ):
        self._frame = frame
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("rd_maps")

    def compute(self, op_input, op_output, context) -> None:
        if not self._collector.first_tick_set:
            self._collector.t_first_tick = now()
            self._collector.first_tick_set = True
        op_output.emit(dict(self._frame), "rd_maps")
        self._collector.starts.append(now())


class PerfOverrideSourceOp(Operator):
    """Emit one override value on a chosen tick, then stay silent."""

    def __init__(
        self,
        fragment,
        *args,
        target_tick: int,
        value: Any,
        **kwargs,
    ):
        self._target_tick = target_tick
        self._value = value
        self._tick = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("output")

    def compute(self, op_input, op_output, context) -> None:
        if self._tick == self._target_tick:
            op_output.emit(self._value, "output")
        self._tick += 1
