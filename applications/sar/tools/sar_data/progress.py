# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-free transfer reporting for the CPHD tools."""

from __future__ import annotations

import sys
import threading
import time


def format_bytes(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class DownloadProgress:
    """One meter, with a heartbeat even during blocking network or disk work.

    Resumed bytes contribute to completion, but never to this run's rate.
    Unknown values are explicit rather than guessed. All output goes to stderr.
    """

    def __init__(self, label: str, total: int | None, stage: str = "Download") -> None:
        self.label = label
        self.total = total
        self.stage = stage
        self.done = 0
        self.initial = 0
        self.started = time.monotonic()
        self.last_draw = float("-inf")
        self.tty = sys.stderr.isatty()
        self._width = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._draw()
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(0.25):
            self._draw()

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.finish()
        else:
            self.abort()

    def phase(self, stage: str, total: int | None = None, initial: int = 0):
        with self._lock:
            self.stage, self.total = stage, total
            self.done = self.initial = initial
            self.started = time.monotonic()
            self._draw(force=True)

    def add(self, num_bytes: int):
        with self._lock:
            self.done += num_bytes
            self._draw()

    def set(self, num_bytes: int):
        with self._lock:
            self.done = num_bytes
            self._draw()

    def message(self, text: str):
        with self._lock:
            self._write(f"{self.stage} {self.label}: {text}", final=True)

    def finish(self):
        self._close("Complete")

    def abort(self):
        self._close("Failed")

    def _close(self, status):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._draw(force=True, status=status)

    def _write(self, line, final=False):
        if self.tty:
            padding = " " * max(0, self._width - len(line))
            print("\r" + line + padding, end="\n" if final else "", file=sys.stderr, flush=True)
            self._width = 0 if final else len(line)
        else:
            print(line, file=sys.stderr, flush=True)

    def _draw(self, force=False, status=None):
        with self._lock:
            now = time.monotonic()
            if not force and now - self.last_draw < (0.25 if self.tty else 2.0):
                return
            self.last_draw = now
            elapsed = max(0, now - self.started)
            rate = (self.done - self.initial) / elapsed if elapsed > 0 else 0
            total = format_bytes(self.total) if self.total is not None else "?"
            pct = (
                f"{100 * self.done / self.total:.1f}%"
                if self.total is not None and self.total > 0
                else "?%"
            )
            eta = "--:--"
            if self.total is not None and self.done >= self.total:
                eta = "00:00"
            elif self.total is not None and rate > 0:
                eta = format_duration((self.total - self.done) / rate)
            line = (
                f"{self.stage} {self.label} | {format_bytes(self.done)} / {total} | {pct}"
                f" | {format_bytes(rate)}/s | elapsed {format_duration(elapsed)} | ETA {eta}"
            )
            if status:
                line += f" | {status}"
            self._write(line, final=status is not None)
