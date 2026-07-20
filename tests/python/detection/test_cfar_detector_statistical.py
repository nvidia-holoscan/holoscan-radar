# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python port of tests/cpp/detection/test_cfar_detector_statistical.cpp.

Monte-Carlo false-alarm calibration. Marked ``statistical`` — excluded from
default CI when deselected; run explicitly with ``-m statistical``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import (  # noqa: E402
    run_operator_multi,
    single_map,
    to_device_array,
)

from holoscan.radar import CFARDetectorOpF64, CfarVariant  # noqa: E402

pytestmark = [pytest.mark.requires_gpu, pytest.mark.statistical]

K_GUARD = 1
K_TRAIN = 2
K_MARGIN = K_GUARD + K_TRAIN


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for CFARDetectorOp tests: {exc}")


def _variant_kwargs(variant: CfarVariant, pfa: float, *, nci: bool) -> dict:
    kwargs: dict = {
        "cfar_variant": variant,
        "guard_cells": [K_GUARD, K_GUARD],
        "training_cells": [K_TRAIN, K_TRAIN],
        "pfa": pfa,
    }
    if variant == CfarVariant.OS:
        kwargs["os_rank"] = 30
    if nci:
        kwargs["nci_axes"] = [0]
    return kwargs


def count_interior_false_alarms(
    score: np.ndarray, num_doppler: int, num_range: int
) -> tuple[int, int]:
    interior = 0
    false_alarms = 0
    grid = score.reshape(num_doppler, num_range)
    for d in range(K_MARGIN, num_doppler - K_MARGIN):
        for r in range(K_MARGIN, num_range - K_MARGIN):
            interior += 1
            if grid[d, r] > 1.0:
                false_alarms += 1
    return interior, false_alarms


def expect_realized_pfa_near_configured(
    realized: float,
    configured: float,
    total_cells: int,
    min_ratio_tol: float = 0.15,
) -> None:
    assert total_cells > 0
    expected = configured * float(total_cells)
    assert expected >= 50.0, f"Monte Carlo design too weak: expected FA {expected}"
    ratio = realized / configured
    sigma = math.sqrt(configured * (1.0 - configured) / float(total_cells))
    tol = max(4.0 * sigma / configured, min_ratio_tol)
    assert abs(ratio - 1.0) <= tol, (
        f"realized_pfa={realized} configured_pfa={configured} "
        f"total_interior_cells={total_cells}"
    )


def run_monte_carlo_pfa_k1(
    num_trials: int,
    num_doppler: int,
    num_range: int,
    configured_pfa: float,
    seed: int,
    variant: CfarVariant,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(num_trials):
        # Exponential power with mean 1 (same as C++ exponential_distribution).
        mp = rng.exponential(scale=1.0, size=(num_doppler, num_range))
        frames.append(
            single_map(to_device_array(mp, dtype=np.float64), "rd_maps")
        )

    def make_op(app):
        return CFARDetectorOpF64(
            app,
            name="op_under_test",
            **_variant_kwargs(variant, configured_pfa, nci=False),
        )

    outs = run_operator_multi(
        make_op,
        frames,
        input_port="rd_maps",
        output_ports=["detection_score"],
        drain_ports=["detection_mask", "noise_estimate", "detection_idx"],
    )
    interior = 0
    false_alarms = 0
    for score in outs["detection_score"]:
        i, f = count_interior_false_alarms(score, num_doppler, num_range)
        interior += i
        false_alarms += f
    return interior, false_alarms


def run_monte_carlo_pfa_nci(
    num_trials: int,
    num_channels: int,
    num_doppler: int,
    num_range: int,
    configured_pfa: float,
    seed: int,
    variant: CfarVariant,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(num_trials):
        cube = rng.exponential(
            scale=1.0, size=(num_channels, num_doppler, num_range)
        )
        frames.append(
            single_map(to_device_array(cube, dtype=np.float64), "rd_maps")
        )

    def make_op(app):
        return CFARDetectorOpF64(
            app,
            name="op_under_test",
            **_variant_kwargs(variant, configured_pfa, nci=True),
        )

    outs = run_operator_multi(
        make_op,
        frames,
        input_port="rd_maps",
        output_ports=["detection_score"],
        drain_ports=["detection_mask", "noise_estimate", "detection_idx"],
    )
    interior = 0
    false_alarms = 0
    for score in outs["detection_score"]:
        i, f = count_interior_false_alarms(score, num_doppler, num_range)
        interior += i
        false_alarms += f
    return interior, false_alarms


@pytest.mark.parametrize(
    "variant,seed,min_tol",
    [
        (CfarVariant.CA, 0xCAFEBABE0001, 0.15),
        (CfarVariant.GO, 0x474F434641520001, 0.15),
        (CfarVariant.SO, 0x534F434641520001, 0.25),
        (CfarVariant.OS, 0x4F53434641520001, 0.15),
    ],
)
def test_MonteCarloPfaK1(variant, seed, min_tol):
    trials, doppler, ranges, pfa = 30, 64, 64, 1e-2
    interior, false_alarms = run_monte_carlo_pfa_k1(
        trials, doppler, ranges, pfa, seed, variant
    )
    assert interior > 0
    realized = float(false_alarms) / float(interior)
    expect_realized_pfa_near_configured(realized, pfa, interior, min_tol)


@pytest.mark.parametrize(
    "variant,seed,min_tol",
    [
        (CfarVariant.CA, 0xCAFEBABE0002, 0.15),
        (CfarVariant.GO, 0x474F434641520002, 0.15),
        (CfarVariant.SO, 0x534F434641520002, 0.25),
        (CfarVariant.OS, 0x4F53434641520002, 0.15),
    ],
)
def test_MonteCarloPfaNciK8(variant, seed, min_tol):
    trials, channels, doppler, ranges, pfa = 30, 8, 64, 64, 1e-2
    interior, false_alarms = run_monte_carlo_pfa_nci(
        trials, channels, doppler, ranges, pfa, seed, variant
    )
    assert interior > 0
    realized = float(false_alarms) / float(interior)
    expect_realized_pfa_near_configured(realized, pfa, interior, min_tol)
