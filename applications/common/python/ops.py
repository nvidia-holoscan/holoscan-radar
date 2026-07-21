# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, rank-agnostic Holoscan operators for the FMCW example apps.

``SimAdcSourceOp`` mirrors ``radar_app/app/source.hpp``'s C++ operator of the
same name. ``TensorMapDiscardOp`` is a Python-only addition: some Python-bound
output ports (e.g. ``BeamscanEstimatorOp.angle_spectrum``) require a connected
consumer where the C++ port sets ``.condition(kNone)`` and can be left
unconnected; the C++ ``radar_app::TensorMapDiscardOp`` (``report_ops.hpp``)
that would otherwise cover this is not yet exposed to Python (a binding gap,
not fixed here) — so the demos keep a small local drain operator instead.
"""

from __future__ import annotations

# holoscan.radar must load before common.python.sim (which imports cupy) -
# MatX CUDA cache / atexit ordering, same rule the app entry points follow.
# isort: off
from holoscan.core import Operator, OperatorSpec
from holoscan.radar import CadenceMeter

from common.python.sim import SimScene, advance_scene, simulate

# isort: on


class SimAdcSourceOp(Operator):
    """Emit one synthetic complex-IF ADC cube per tick.

    Rank-agnostic (mirrors the C++ SimAdcSourceOp): the emitted cube follows
    the simulator's shape verbatim - rank-2 for a single RX element (SISO),
    rank-3 for multiple (SIMO / dense). This source does not pre-validate
    rank; a shape mismatch surfaces as a clear error from the first
    downstream operator that rejects it (rank-agnostic batching - see
    AGENTS.md design invariants).
    """

    def __init__(
        self,
        fragment,
        *args,
        scene: SimScene,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        result = simulate(self.scene)
        op_output.emit({"adc_cube": result.cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


class TensorMapDiscardOp(Operator):
    """Drain an unused TensorMap port."""

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("input")
