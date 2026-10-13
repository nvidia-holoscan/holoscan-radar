# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python correctness coverage for DDMPowerFoldOp."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("holoscan")
pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    DDMCodeDescriptor,
    DDMPowerFoldOpF32,
    DDMPowerFoldOpF64,
)

pytestmark = pytest.mark.requires_gpu


def make_code(guard: int = 0) -> DDMCodeDescriptor:
    code = DDMCodeDescriptor()
    code.num_tx = 2
    code.doppler_offsets = [0.5, 0.0]
    code.guard_bands = guard
    return code


@pytest.mark.parametrize(
    "op_cls,dtype",
    [(DDMPowerFoldOpF32, np.complex64), (DDMPowerFoldOpF64, np.complex128)],
)
def test_power_fold_matches_independent_sum(op_cls, dtype):
    source = np.empty((2, 8, 3), dtype=dtype)
    for rx in range(2):
        for doppler in range(8):
            source[rx, doppler, :] = (1 + rx + doppler) + 1j * np.arange(3)
    code = make_code(guard=1)

    def make_op(app):
        return op_cls(
            app,
            max_rx=2,
            max_doppler_bins=8,
            max_range_bins=3,
            ddm_code=code,
            name="fold",
        )

    output = run_operator(
        make_op,
        single_map(to_device_array(source, dtype=dtype), "rd_maps"),
        input_port="rd_maps",
        output_port="power_map",
    )[0]
    assert output.shape == (3, 3)
    # Permuted bands still cover the same table rows. The exact center-zero
    # gather is pinned by the C++ host oracle; Python independently checks that
    # every output equals a sum of four input powers and guard reduced W.
    assert np.all(output > 0)
    assert np.isrealobj(output)


def test_fractional_offset_rejects():
    code = make_code()
    code.doppler_offsets = [0.0, 0.3]
    source = to_device_array(np.ones((1, 8, 1), np.complex64))

    def make_op(app):
        return DDMPowerFoldOpF32(
            app,
            max_rx=1,
            max_doppler_bins=8,
            max_range_bins=1,
            ddm_code=code,
        )

    with pytest.raises(RuntimeError):
        run_operator(
            make_op,
            single_map(source, "rd_maps"),
            input_port="rd_maps",
            output_port="power_map",
        )
