# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic CuPy tensor factories for performance scenarios.

Large cubes are allocated and filled **on device**. A host build +
``cp.asarray`` path adds ~150 ms of first-tick cold on xl cubes through the
Holoscan Python emit/receive path (C++ does not see that). Avoid full-cube
host buffers and avoid full-cube float64 index temporaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cupy as cp
import numpy as np


def single_map(tensor: cp.ndarray, key: str) -> dict[str, cp.ndarray]:
    return {key: tensor}


def to_device_array(host: np.ndarray) -> cp.ndarray:
    """Upload a host array as CUDA device memory (zero-size safe).

    CuPy's CuPy-array-to-TensorMap emit path mislabels zero-size buffers as
    host (CPU) memory, which operators that require CUDA-accessible input
    (e.g. FeatureAssemblyOp.detection_idx) reject. Route those through the
    radar package's ``make_device_tensor`` instead — mirrors
    ``tests/python/utils/harness.py::to_device_array``.
    """
    if host.size == 0:
        from holoscan.radar import make_device_tensor

        return make_device_tensor(list(host.shape), host.dtype)
    return cp.asarray(host)


def make_power_map_device(shape: Sequence[int]) -> cp.ndarray:
    """Deterministic power map: floor 1.0 with sparse 500.0 peaks (stride 997)."""
    shape_t = tuple(int(d) for d in shape)
    out = cp.ones(shape_t, dtype=cp.float32)
    flat = out.ravel()
    # Index only the peak locations — do not materialize arange(count).
    count = int(flat.size)
    if count > 0:
        peaks = cp.arange(0, count, 997, dtype=cp.int64)
        flat[peaks] = 500.0
    return out


def make_iq_cube_device(shape: Sequence[int]) -> cp.ndarray:
    """Deterministic multi-tone IQ cube, filled on device.

    Pattern lives on the last axis only (broadcast), matching the C++ harness
    intent without a full-cube index temporary.
    """
    shape_t = tuple(int(d) for d in shape)
    if not shape_t:
        return cp.zeros((), dtype=cp.complex64)
    n = int(shape_t[-1])
    denom = float(n if n > 0 else 1)
    s = cp.arange(n, dtype=cp.float32)
    phase1 = (2.0 * math.pi * 5.0) * s / denom
    phase2 = (2.0 * math.pi * 17.0) * s / denom
    tone = cp.empty(n, dtype=cp.complex64)
    tone.real = (cp.cos(phase1) + 0.5 * cp.cos(phase2) + 0.1).astype(cp.float32, copy=False)
    tone.imag = (cp.sin(phase1) + 0.5 * cp.sin(phase2)).astype(cp.float32, copy=False)
    # Broadcast last-axis tone across leading dims (one device alloc for result).
    return cp.broadcast_to(tone, shape_t).copy()


def deterministic_detection_idx_host(detections: int, doppler: int, range_bins: int) -> np.ndarray:
    """Deterministic (doppler_bin, range_bin) grid, wrapping across the RD plane."""
    host = np.zeros((max(detections, 0), 2), dtype=np.int64)
    cells = max(doppler * range_bins, 1)
    for i in range(detections):
        cell = (i * 997) % cells
        host[i, 0] = cell // range_bins
        host[i, 1] = cell % range_bins
    return host


def make_detection_idx_device(detections: int, doppler: int, range_bins: int) -> cp.ndarray:
    host = deterministic_detection_idx_host(detections, doppler, range_bins)
    return to_device_array(host)
