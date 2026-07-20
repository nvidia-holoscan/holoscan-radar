# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal and null sinks for the Python performance harness."""

from __future__ import annotations

import cupy as cp

from holoscan.core import Operator, OperatorSpec
from utils.perf_app import TimingCollector, now


class PerfTerminalSinkOp(Operator):
    """Drain one output port and record end time after a device sync."""

    def __init__(
        self,
        fragment,
        *args,
        collector: TimingCollector,
        **kwargs,
    ):
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("input")
        self._collector.ends_receive.append(now())
        cp.cuda.Device().synchronize()
        self._collector.ends.append(now())


def make_null_sink(app, name: str):
    """Entity drain for dense/bool outputs (``radar_app.NullSinkOp``).

    Must be called after ``bootstrap_radar_package()`` so ``holoscan.radar``
    resolves from the build tree.
    """
    from holoscan.radar import NullSinkOp

    return NullSinkOp(app, name=name)
