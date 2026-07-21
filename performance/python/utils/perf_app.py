# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared timing collector for the Python performance harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def to_ms(seconds: float) -> float:
    return seconds * 1000.0


@dataclass
class TimingCollector:
    t_construct: float = 0.0
    t_first_tick: float = 0.0
    first_tick_set: bool = False
    starts: list[float] = field(default_factory=list)
    ends_receive: list[float] = field(default_factory=list)
    ends: list[float] = field(default_factory=list)

    def setup_ms(self) -> float:
        return to_ms(self.t_first_tick - self.t_construct)

    def frame_latencies_ms(self) -> list[float]:
        """Full frame: post-source-emit → sink device sync."""
        n = min(len(self.starts), len(self.ends))
        return [to_ms(self.ends[i] - self.starts[i]) for i in range(n)]

    def to_receive_latencies_ms(self) -> list[float]:
        """Post-source-emit → sink receive (before device sync)."""
        n = min(len(self.starts), len(self.ends_receive))
        return [to_ms(self.ends_receive[i] - self.starts[i]) for i in range(n)]

    def sync_wait_latencies_ms(self) -> list[float]:
        """Sink receive → device sync (GPU drain)."""
        n = min(len(self.ends_receive), len(self.ends))
        return [to_ms(self.ends[i] - self.ends_receive[i]) for i in range(n)]


def now() -> float:
    return time.perf_counter()
