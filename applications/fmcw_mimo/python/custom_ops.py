# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-specific operators for fmcw_mimo:
#   DdmReportVelocityBinsOp — post-DDM-demux velocity axis for reporting
#     (mirrors the C++ app-local op of the same name)
#   MimoTdmAdcSourceOp / MimoDdmAdcSourceOp — TDM/DDM synthetic ADC sources;
#     MIMO's multiplexing needs a bespoke simulator (common.python.mimo_sim),
#     so it keeps its own sources rather than the shared SimAdcSourceOp.
#
# Colocated with the app (only fmcw_mimo needs these). Imported bare
# (``import custom_ops``) from fmcw_mimo.py: see config.py's docstring for
# why this is script-local, not a shared package.

from __future__ import annotations

import cupy as cp
import numpy as np
from common.python.mimo_sim import (
    MimoScene,
    advance_scene,
    simulate_mimo_ddm_coded,
    simulate_mimo_ddm_phase_coded,
    simulate_mimo_tdm_interleaved,
)

from holoscan.core import Operator, OperatorSpec
from holoscan.radar import CadenceMeter, TXSchedule


class DdmReportVelocityBinsOp(Operator):
    """Post-DDM-demux velocity axis for reporting (mirrors C++ app-local op)."""

    def __init__(
        self,
        fragment,
        *args,
        waveform,
        num_doppler: int,
        bin_spacing_hz: float,
        **kwargs,
    ):
        self.waveform = waveform
        self.num_doppler = num_doppler
        self.bin_spacing_hz = bin_spacing_hz
        self._velocity_bins: cp.ndarray | None = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("sync")
        spec.output("velocity_bins")

    def compute(self, op_input, op_output, context) -> None:
        op_input.receive("sync")
        if self._velocity_bins is None:
            if self.num_doppler <= 0:
                raise RuntimeError("DdmReportVelocityBinsOp: num_doppler must be positive")
            if self.bin_spacing_hz <= 0.0:
                raise RuntimeError("DdmReportVelocityBinsOp: bin_spacing_hz must be > 0")
            wavelength = self.waveform.wavelength()
            axis = np.zeros(self.num_doppler, dtype=np.float32)
            for i in range(self.num_doppler):
                f = (i - self.num_doppler // 2) * self.bin_spacing_hz
                axis[i] = -f * wavelength / 2.0
            self._velocity_bins = cp.asarray(axis)
        op_output.emit({"velocity_bins": self._velocity_bins}, "velocity_bins")


class MimoTdmAdcSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        scene: MimoScene,
        tx_schedule: TXSchedule,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.tx_schedule = tx_schedule
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        result = simulate_mimo_tdm_interleaved(self.scene, self.tx_schedule)
        if len(result.shape) != 3:
            raise RuntimeError("MimoTdmAdcSourceOp: expected rank-3 (N_rx, M*L, samples)")
        # Keep a reference so the DLPack buffer outlives this compute().
        self._adc_cube = result.cube
        op_output.emit({"adc_cube": self._adc_cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)


class MimoDdmAdcSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        scene: MimoScene,
        doppler_offsets: list[float] | None = None,
        phase_codes: list[complex] | None = None,
        use_phase_codes: bool = False,
        frame_period: float = 0.1,
        cadence: CadenceMeter | None = None,
        **kwargs,
    ):
        self.scene = scene
        self.doppler_offsets = doppler_offsets or []
        self.phase_codes = phase_codes or []
        self.use_phase_codes = use_phase_codes
        self.frame_period = frame_period
        self.cadence = cadence
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("adc_cube")

    def compute(self, op_input, op_output, context) -> None:
        if self.use_phase_codes:
            result = simulate_mimo_ddm_phase_coded(self.scene, self.phase_codes)
        else:
            result = simulate_mimo_ddm_coded(self.scene, self.doppler_offsets)
        if len(result.shape) != 3:
            raise RuntimeError("MimoDdmAdcSourceOp: expected rank-3 (N_rx, num_chirps, samples)")
        self._adc_cube = result.cube
        op_output.emit({"adc_cube": self._adc_cube}, "adc_cube")
        if self.cadence is not None:
            self.cadence.on_source_emit()
        advance_scene(self.scene, self.frame_period)
