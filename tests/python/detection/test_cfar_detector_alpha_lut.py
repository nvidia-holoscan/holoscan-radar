# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python port of tests/cpp/detection/test_cfar_detector_alpha_lut.cpp."""

from __future__ import annotations

import math

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import CFARDetectorOpF64  # noqa: E402
from holoscan.radar import (
    CFARDetectorOpF32,
    CfarVariant,
    build_ca_alpha_lut,
    build_go_alpha_lut,
    build_os_alpha_lut,
    build_so_alpha_lut,
)

pytestmark = pytest.mark.requires_gpu

K_DOPPLER = 16
K_RANGE = 32
K_GUARD_D = 1
K_GUARD_R = 1
K_TRAIN_D = 2
K_TRAIN_R = 2


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for CFARDetectorOp tests: {exc}")


def power_map(
    floor: float,
    targets: list[tuple[int, int]],
    peak: float,
) -> np.ndarray:
    arr = np.full(K_DOPPLER * K_RANGE, floor, dtype=np.float64)
    for d, r in targets:
        arr[d * K_RANGE + r] = peak
    return arr


def num_detections(idx: np.ndarray) -> int:
    return 0 if idx.size == 0 else int(idx.shape[0])


def build_variant_lut(variant: CfarVariant, pfa: float, fold: int, os_rank: int) -> list[float]:
    if variant == CfarVariant.CA:
        return list(
            build_ca_alpha_lut(
                pfa,
                fold,
                K_DOPPLER,
                K_RANGE,
                K_GUARD_D,
                K_GUARD_R,
                K_TRAIN_D,
                K_TRAIN_R,
            )
        )
    if variant == CfarVariant.GO:
        return list(
            build_go_alpha_lut(
                pfa,
                fold,
                K_DOPPLER,
                K_RANGE,
                K_GUARD_D,
                K_GUARD_R,
                K_TRAIN_D,
                K_TRAIN_R,
            )
        )
    if variant == CfarVariant.SO:
        return list(
            build_so_alpha_lut(
                pfa,
                fold,
                K_DOPPLER,
                K_RANGE,
                K_GUARD_D,
                K_GUARD_R,
                K_TRAIN_D,
                K_TRAIN_R,
            )
        )
    return list(
        build_os_alpha_lut(
            pfa,
            fold,
            os_rank,
            K_DOPPLER,
            K_RANGE,
            K_GUARD_D,
            K_GUARD_R,
            K_TRAIN_D,
            K_TRAIN_R,
        )
    )


def inflated_lut(lut: list[float], scale: float) -> list[float]:
    out = list(lut)
    for i in range(1, len(out)):
        out[i] *= scale
    return out


def run_cfar_idx(host, *, dtype=np.float64, op_cls=CFARDetectorOpF64, **kwargs):
    arr = np.asarray(host, dtype=dtype).reshape(K_DOPPLER, K_RANGE)
    frame = single_map(to_device_array(arr, dtype=dtype), "rd_maps")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="rd_maps",
        output_ports=["detection_idx"],
        drain_ports=["detection_mask", "detection_score", "noise_estimate"],
    )
    return outs["detection_idx"][0]


def _base_args(variant: CfarVariant, *, os_rank: int | None = None) -> dict:
    args = {
        "cfar_variant": variant,
        "guard_cells": [1, 1],
        "training_cells": [2, 2],
        "pfa": 1e-3,
    }
    if variant == CfarVariant.OS:
        args["os_rank"] = 18 if os_rank is None else os_rank
    return args


@pytest.mark.parametrize(
    "variant,os_rank",
    [
        (CfarVariant.CA, 0),
        (CfarVariant.GO, 0),
        (CfarVariant.SO, 0),
        (CfarVariant.OS, 18),
    ],
)
def test_AlphaLutPrecedenceOverPfa(variant, os_rank):
    mp = power_map(1.0, [(8, 16)], 100.0)
    pfa_lut = build_variant_lut(variant, 1e-3, 1, os_rank)
    huge_lut = inflated_lut(pfa_lut, 1e6)
    args = _base_args(variant, os_rank=os_rank if variant == CfarVariant.OS else None)
    with_pfa = run_cfar_idx(mp, **args, blocking_alpha_lut_rebuild=True)
    assert num_detections(with_pfa) >= 1
    with_lut = run_cfar_idx(mp, **args, alpha_lut=huge_lut)
    assert num_detections(with_lut) == 0


def test_RejectsTooShortAlphaLut():
    mp = power_map(1.0, [], 1.0)
    with pytest.raises(ValueError):
        run_cfar_idx(
            mp,
            cfar_variant=CfarVariant.CA,
            guard_cells=[1, 1],
            training_cells=[2, 2],
            pfa=1e-3,
            alpha_lut=[0.0, 1.0],
        )


def test_RejectsZeroAlphaLutEntry():
    mp = power_map(1.0, [], 1.0)
    lut = build_variant_lut(CfarVariant.CA, 1e-3, 1, 0)
    assert len(lut) > 1
    lut[1] = 0.0
    with pytest.raises(ValueError):
        run_cfar_idx(
            mp,
            cfar_variant=CfarVariant.CA,
            guard_cells=[1, 1],
            training_cells=[2, 2],
            alpha_lut=lut,
        )


def test_RejectsNonFiniteAlphaLutEntry():
    mp = power_map(1.0, [], 1.0)
    lut = build_variant_lut(CfarVariant.CA, 1e-3, 1, 0)
    lut[1] = math.nan
    with pytest.raises(ValueError):
        run_cfar_idx(
            mp,
            cfar_variant=CfarVariant.CA,
            guard_cells=[1, 1],
            training_cells=[2, 2],
            alpha_lut=lut,
        )


def test_FloatDoubleAgreeWithExternalAlphaLut():
    mp = power_map(1.0, [(8, 16)], 100.0)
    lut = build_variant_lut(CfarVariant.CA, 1e-3, 1, 0)
    f_out = run_cfar_idx(
        mp.astype(np.float32),
        dtype=np.float32,
        op_cls=CFARDetectorOpF32,
        cfar_variant=CfarVariant.CA,
        guard_cells=[1, 1],
        training_cells=[2, 2],
        alpha_lut=lut,
    )
    d_out = run_cfar_idx(
        mp,
        cfar_variant=CfarVariant.CA,
        guard_cells=[1, 1],
        training_cells=[2, 2],
        alpha_lut=lut,
    )
    assert num_detections(f_out) == num_detections(d_out)


def test_PfaIgnoredWhileExternalAlphaLutActive():
    mp = power_map(1.0, [(8, 16)], 100.0)
    pfa_lut = build_variant_lut(CfarVariant.CA, 1e-3, 1, 0)
    huge_lut = inflated_lut(pfa_lut, 1e6)
    with_loose = run_cfar_idx(
        mp,
        cfar_variant=CfarVariant.CA,
        guard_cells=[1, 1],
        training_cells=[2, 2],
        pfa=0.5,
        alpha_lut=huge_lut,
    )
    assert num_detections(with_loose) == 0


def test_LiveAlphaLutOverrideStickyAndClearing():
    mp = power_map(1.0, [(8, 16)], 100.0)
    pfa_lut = build_variant_lut(CfarVariant.CA, 1e-3, 1, 0)
    huge_lut = inflated_lut(pfa_lut, 1e6)
    arr = np.asarray(mp, dtype=np.float64).reshape(K_DOPPLER, K_RANGE)
    frame = single_map(to_device_array(arr, dtype=np.float64), "rd_maps")

    def make_op(app):
        return CFARDetectorOpF64(
            app,
            name="op_under_test",
            cfar_variant=CfarVariant.CA,
            guard_cells=[1, 1],
            training_cells=[2, 2],
            pfa=1e-3,
        )

    outs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="rd_maps",
        output_ports=["detection_idx"],
        overrides={"alpha_lut": [huge_lut, huge_lut, []]},
        drain_ports=["detection_mask", "detection_score", "noise_estimate"],
    )
    idx = outs["detection_idx"]
    assert num_detections(idx[0]) == 0
    assert num_detections(idx[1]) == 0
    assert num_detections(idx[2]) >= 1
