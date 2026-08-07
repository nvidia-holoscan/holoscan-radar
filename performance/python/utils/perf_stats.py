# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency aggregation for the Python performance harness."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LatencyStats:
    min_ms: float = math.nan
    median_ms: float = math.nan
    p90_ms: float = math.nan
    p99_ms: float = math.nan
    max_ms: float = math.nan
    samples: int = 0


def percentile_sorted(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return math.nan
    rank = q * (len(sorted_samples) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac


def summarize(samples: list[float]) -> LatencyStats:
    stats = LatencyStats(samples=len(samples))
    if not samples:
        return stats
    ordered = sorted(samples)
    stats.min_ms = ordered[0]
    stats.max_ms = ordered[-1]
    stats.median_ms = percentile_sorted(ordered, 0.50)
    stats.p90_ms = percentile_sorted(ordered, 0.90)
    stats.p99_ms = percentile_sorted(ordered, 0.99)
    return stats
