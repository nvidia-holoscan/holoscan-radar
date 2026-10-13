# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python correctness coverage for DDMSparseDemultiplexOp."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("holoscan")
pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    DDMCodeDescriptor,
    DDMSparseDemultiplexOpF32,
    OverlapPolicy,
)

pytestmark = pytest.mark.requires_gpu


def test_default_fold_zero_and_initial_phase_dephasing():
    code = DDMCodeDescriptor()
    code.num_tx = 2
    code.phase_codes = [
        1j,
        1j,
        1j,
        1j,
        1j,
        1j,
        1j,
        1j,
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
    ]
    source = np.zeros((1, 8, 1), np.complex64)
    # For folded center bin 2, the C++ shared table maps TX0/TX1 to bins 0/4.
    source[0, 0, 0] = 2j
    source[0, 4, 0] = -3
    detection = np.array([[2, 0]], np.int64)

    def make_op(app):
        return DDMSparseDemultiplexOpF32(
            app,
            max_tx=2,
            max_rx=1,
            max_doppler_bins=8,
            max_range_bins=1,
            max_detections=1,
            max_virtual_channels=2,
            max_snapshots=1,
            ddm_code=code,
            tx_positions=[0, 0, 0, 1, 0, 0],
            rx_positions=[0, 0, 0],
            overlap_policy=OverlapPolicy.KEEP_ALL,
            wavelength=0.004,
            name="sparse",
        )

    output = run_operator(
        make_op,
        single_map(to_device_array(source), "rd_maps"),
        input_port="rd_maps",
        output_port="snapshot",
        extra_inputs={"detection_idx": [single_map(to_device_array(detection), "detection_idx")]},
        drain_ports=["is_valid", "array_geometry"],
    )[0]
    np.testing.assert_allclose(output[0, :, 0], [2, 3], atol=1e-5)
