# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort provenance and platform capture for performance reports."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import cupy as cp


@dataclass
class Provenance:
    git_commit: str = "unknown"
    git_dirty: bool = False
    timestamp_utc: str = "unknown"


@dataclass
class Platform:
    gpu_name: str = "unknown"
    gpu_driver: str = "unknown"
    cuda_runtime: str = "unknown"
    holoscan_version: str = "unknown"
    cpu_model: str = "unknown"
    os_name: str = "unknown"


def _run_capture(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def capture_provenance() -> Provenance:
    prov = Provenance()
    commit = _run_capture("git rev-parse --short HEAD 2>/dev/null")
    if commit:
        prov.git_commit = commit
    prov.git_dirty = bool(_run_capture("git status --porcelain 2>/dev/null"))
    prov.timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return prov


def capture_platform() -> Platform:
    plat = Platform()
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        plat.gpu_name = (
            props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        )
    except Exception:
        pass

    try:
        driver = cp.cuda.runtime.driverGetVersion()
        plat.gpu_driver = f"{driver // 1000}.{(driver % 1000) // 10}"
    except Exception:
        pass

    try:
        runtime = cp.cuda.runtime.runtimeGetVersion()
        plat.cuda_runtime = f"{runtime // 1000}.{(runtime % 1000) // 10}"
    except Exception:
        pass

    try:
        import holoscan

        plat.holoscan_version = getattr(holoscan, "__version__", "unknown")
    except Exception:
        pass

    cpu = _run_capture("lscpu 2>/dev/null | grep 'Model name' | head -1")
    if cpu and ":" in cpu:
        plat.cpu_model = cpu.split(":", 1)[1].strip()
    else:
        plat.cpu_model = platform.processor() or "unknown"

    plat.os_name = _run_capture("uname -sr 2>/dev/null") or platform.platform()
    return plat
