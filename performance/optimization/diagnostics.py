# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only environment preflight and scoped Nsight SQLite phase attribution."""

import argparse
import collections
import shutil
import sqlite3
import statistics
import subprocess
from pathlib import Path

from artifacts import digest, write_json


def clock_state():
    configuration, observations = {}, {}
    roots = sorted(Path("/sys/class/devfreq").glob("*")) + sorted(
        Path("/sys/devices/system/cpu/cpufreq").glob("policy*")
    )
    for root in roots:
        for name in (
            "governor",
            "min_freq",
            "max_freq",
            "scaling_governor",
            "scaling_min_freq",
            "scaling_max_freq",
        ):
            try:
                configuration[str(root / name)] = (root / name).read_text().strip()
            except OSError:
                pass
        for name in ("cur_freq", "scaling_cur_freq"):
            try:
                observations[str(root / name)] = (root / name).read_text().strip()
            except OSError:
                pass
    return {
        "configuration": configuration,
        "observations": observations,
        "note": "Read-only snapshot; empty data means unknown, not fixed clocks.",
    }


def preflight(docker_image=None):
    tools = {}
    for name in (
        "holoscan",
        "docker",
        "nsys",
        "ncu",
        "node",
        "pre-commit",
        "nvidia-smi",
        "nvcc",
        "c++",
        "nvpmodel",
    ):
        path = shutil.which(name)
        entry = {"path": path, "available": bool(path)}
        if path:
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=15, check=False
                )
                entry.update(returncode=result.returncode, version=(result.stdout + result.stderr))
            except subprocess.SubprocessError as error:
                entry["error"] = str(error)
        tools[name] = entry
    sampling = Path("/proc/sys/kernel/perf_event_paranoid")
    probes = {}
    for name, argv in {
        "gpu_clocks": ["nvidia-smi", "-q", "-d", "CLOCK,POWER"],
        "power_mode": ["nvpmodel", "-q"],
        "image_identity": ["docker", "image", "inspect", docker_image] if docker_image else [],
    }.items():
        if argv and shutil.which(argv[0]):
            try:
                value = subprocess.run(
                    argv, capture_output=True, text=True, timeout=15, check=False
                )
                probes[name] = {
                    "returncode": value.returncode,
                    "output": value.stdout + value.stderr,
                }
            except subprocess.SubprocessError as error:
                probes[name] = {"error": str(error)}
    release = Path("/etc/nv_tegra_release")
    return {
        "scope": "current execution environment; rerun inside the selected SDK image",
        "tools": tools,
        "clocks": clock_state(),
        "identity_probes": probes,
        "l4t": release.read_text().strip() if release.exists() else None,
        "sdk_revision": "unknown unless verified from image metadata or matching SDK build records",
        "perf_event_paranoid": sampling.read_text().strip() if sampling.exists() else None,
        "limitations": [
            name + " unavailable" for name, entry in tools.items() if not entry["path"]
        ],
        "note": "No provisioning or privilege changes. Probe nsys options and GPU counters separately.",
    }


def union_duration(intervals):
    end, total = None, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def phase_breakdown(
    path,
    started_at,
    source_id,
    range_name="op:compute",
    global_tid=None,
    warmup=0,
    case_id=None,
    repeat=0,
    timed_output=None,
):
    path = Path(path).resolve()
    if not source_id or not path.is_file() or not path.stat().st_size:
        raise ValueError("phase attribution requires source identity and a nonempty SQLite export")
    if path.stat().st_mtime < started_at or warmup < 0:
        raise ValueError("stale phase export or invalid warmup")
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            strings = dict(db.execute("SELECT id, value FROM StringIds"))
            nvtx = [
                (start, end, text or strings.get(text_id, ""), tid)
                for start, end, text, text_id, tid in db.execute(
                    "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS"
                )
            ]
            runtime = [
                (start, end, strings.get(name_id, str(name_id)), tid)
                for start, end, name_id, tid in db.execute(
                    "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
                )
            ]
    except sqlite3.Error as error:
        raise ValueError(f"unsupported or incomplete Nsight SQLite schema: {error}") from error
    ranges = sorted(
        (start, end, tid)
        for start, end, name, tid in nvtx
        if name in {range_name, "holoscan:" + range_name}
        and end is not None
        and end > start
        and (global_tid is None or tid == global_tid)
    )
    if case_id is not None:
        label = f"radar-perf:case:{case_id}:repeat={repeat}"
        sessions = [(a, b) for a, b, name, _ in nvtx if name == label and b is not None]
        if len(sessions) != 1:
            raise ValueError("missing/ambiguous explicit case/repeat session label")
        start, end = sessions[0]
        ranges = [(a, b, tid) for a, b, tid in ranges if start <= a and b <= end]
    if len({tid for _, _, tid in ranges}) > 1 and global_tid is None:
        raise ValueError("ambiguous operator ranges; select --global-tid or a scoped capture")
    ranges = ranges[warmup:]
    if not ranges or not runtime:
        raise ValueError("no scoped NVTX ranges or CUDA runtime events")
    durations, api_time, gaps = [], [], []
    calls, phases = collections.defaultdict(list), collections.defaultdict(list)
    after_emit = []
    for start, end, tid in ranges:
        intervals = []
        frame_calls = collections.Counter()
        for a, b, name, thread in runtime:
            if thread == tid and b is not None and a < end and b > start:
                a, b = max(a, start), min(b, end)
                intervals.append((a, b))
                frame_calls[name] += (b - a) / 1000
        covered = union_duration(intervals) / 1000
        durations.append((end - start) / 1000)
        api_time.append(covered)
        gaps.append((end - start) / 1000 - covered)
        for name, duration in frame_calls.items():
            calls[name].append(duration)
        marks = sorted(
            (a, name)
            for a, b, name, thread in nvtx
            if thread == tid and start <= a < end and b is None and name.startswith("op.")
        )
        points = [(start, "start"), *marks, (end, "end")]
        if timed_output:
            emit_names = {timed_output + ":emit", "op." + timed_output + ":emit"}
            emit_names |= {"holoscan:" + name for name in emit_names}
            emits = [
                a
                for a, _, name, thread in nvtx
                if thread == tid and start <= a <= end and name in emit_names
            ]
            if emits:
                after_emit.append((end - min(emits)) / 1000)
        for (a, first), (b, second) in zip(points, points[1:]):
            phases[first + " -> " + second].append((b - a) / 1000)
    return {
        "source_id": source_id,
        "sqlite": str(path),
        "sha256": digest(path),
        "range": range_name,
        "case_id": case_id,
        "repeat": repeat,
        "after_timed_emit_median_us": statistics.median(after_emit) if after_emit else None,
        "frames": len(ranges),
        "warmup_removed": warmup,
        "compute_median_us": statistics.median(durations),
        "cuda_api_union_median_us": statistics.median(api_time),
        "outside_cuda_api_median_us": statistics.median(gaps),
        "calls": {
            name: {"frames": len(v), "median_us": statistics.median(v)} for name, v in calls.items()
        },
        "phases": {
            name: {"occurrences": len(v), "median_us": statistics.median(v)}
            for name, v in phases.items()
        },
        "scope": "profiled attribution only; outside-API time includes waits, not just host work",
        "warning": "No inferred scenario grouping. Supply a single-case capture; medians are not additive.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--out", required=True)
    pre.add_argument("--docker-image")
    phase = commands.add_parser("phases")
    phase.add_argument("--sqlite", required=True)
    phase.add_argument("--started-at", type=float, required=True)
    phase.add_argument("--source-id", required=True)
    phase.add_argument("--range", default="op:compute")
    phase.add_argument("--global-tid", type=int)
    phase.add_argument("--warmup", type=int, default=0)
    phase.add_argument("--out", required=True)
    phase.add_argument("--case")
    phase.add_argument("--repeat", type=int, default=0)
    phase.add_argument("--timed-output")
    args = parser.parse_args()
    value = (
        preflight(args.docker_image)
        if args.command == "preflight"
        else phase_breakdown(
            args.sqlite,
            args.started_at,
            args.source_id,
            args.range,
            args.global_tid,
            args.warmup,
            args.case,
            args.repeat,
            args.timed_output,
        )
    )
    write_json(args.out, value)


if __name__ == "__main__":
    main()
