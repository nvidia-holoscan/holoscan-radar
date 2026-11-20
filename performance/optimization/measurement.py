# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-three raw timing protocols and serial, resumable adjacent measurements."""

import copy
import fcntl
import hashlib
import math
import os
import signal
import stat
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from artifacts import digest, finite, read_json, write_json
from diagnostics import clock_state

EARLY_METRICS = ("early_median_ms", "early_p90_ms", "early_p99_ms")


def protocol(sampling):
    value = {
        "scope": "overrides",
        "warmup": 5,
        "frames": 50,
        "repeats": 3,
        "process_mode": "isolated",
        "raw_frame_times": True,
        "early_frames": 50,
        "pairing_order": "adjacent_repeat",
        "aggregation": "median",
        "case_overrides": {},
    } | sampling
    allowed = {
        "scope",
        "warmup",
        "frames",
        "repeats",
        "process_mode",
        "raw_frame_times",
        "early_frames",
        "pairing_order",
        "aggregation",
        "case_overrides",
        "sizes",
        "cases",
    }
    if set(sampling) - allowed:
        raise ValueError("unsupported sampling option")
    for key, minimum in (("warmup", 0), ("frames", 50), ("repeats", 1), ("early_frames", 1)):
        if type(value[key]) is not int or value[key] < minimum:
            raise ValueError("invalid sampling option: " + key)
    if value["early_frames"] > value["frames"]:
        raise ValueError("early window exceeds frame count")
    if value["process_mode"] != "isolated" or value["raw_frame_times"] is not True:
        raise ValueError("version 3 requires isolated processes and raw frame times")
    if value["pairing_order"] != "adjacent_repeat" or value["aggregation"] not in {
        "median",
        "mean",
    }:
        raise ValueError("unsupported pairing/aggregation")
    if not isinstance(value["case_overrides"], dict):
        raise ValueError("case overrides must be a mapping")
    for case, override in value["case_overrides"].items():
        if (
            not isinstance(case, str)
            or not isinstance(override, dict)
            or set(override) - {"frames", "repeats"}
        ):
            raise ValueError("invalid case override")
        effective(value, case)
    return value


def effective(sampling, case):
    value = dict(sampling) | sampling.get("case_overrides", {}).get(case, {})
    for name, minimum in (("frames", max(50, sampling["early_frames"])), ("repeats", 1)):
        if type(value[name]) is not int or value[name] < minimum:
            raise ValueError("invalid case sampling: " + case)
    return value


def derive_raw(report, sampling):
    """Recompute frame metrics; never synthesize missing sequences from summaries."""
    from timing import quantile

    if report.get("precision_children"):
        children = []
        for ref in report["precision_children"]:
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("precision child changed")
            child = read_json(ref["path"])
            if child["run_settings"] != report["run_settings"]:
                raise ValueError("precision child sampling differs")
            children.extend(child["results"])
        if children != report["results"]:
            raise ValueError("precision aggregate differs from child reports")
    result = copy.deepcopy(report)
    if result["run_settings"].get("repeats") != 1:
        raise ValueError("raw child must contain one isolated repeat")
    for row in result["results"]:
        cfg = effective(sampling, row["scenario_id"])
        if result["run_settings"]["measured_frames"] != cfg["frames"]:
            raise ValueError("raw child frame count differs from protocol")
        if row["bucket"] != "compute":
            continue
        values = row.get("raw_steady_ms", [])
        if len(values) != cfg["frames"] or row.get("raw_steady_run_offsets") != [0, len(values)]:
            raise ValueError("missing raw frames or invalid repeat boundaries")
        for value in values:
            finite(value, "raw frame", positive=True)
        for metric, probability in (("median_ms", 0.5), ("p90_ms", 0.9), ("p99_ms", 0.99)):
            actual = quantile(values, probability)
            if not math.isclose(actual, row[metric], rel_tol=1e-5):
                raise ValueError("raw sequence disagrees with reported " + metric)
            row[metric] = actual
            row["early_" + metric] = quantile(values[: sampling["early_frames"]], probability)
        midpoint = len(values) // 2
        first, last = quantile(values[:midpoint], 0.5), quantile(values[midpoint:], 0.5)
        mean = statistics.mean(values)
        variance = sum((x - mean) ** 2 for x in values)
        row["frame_diagnostics"] = {
            "half_drift_pct": 100 * (last - first) / first,
            "early_bias_pct": 100 * (row["early_median_ms"] - row["median_ms"]) / row["median_ms"],
            "lag1": (
                sum((a - mean) * (b - mean) for a, b in zip(values, values[1:])) / variance
                if variance
                else 0
            ),
            "stall_frames": sum(x > 3 * row["median_ms"] for x in values),
            "min_ms": min(values),
            "max_ms": max(values),
            "note": "Descriptive; no stationarity or independence claim; no frames excluded.",
        }
    return result


@contextmanager
def device_lock(device):
    # Per-user coordination across working directories. Keep lock inodes after
    # release: unlinking permits a new entrant to bypass an existing waiter.
    uid = os.getuid()
    folder = Path(tempfile.gettempdir()) / f"radar-operator-perf-locks-{uid}"
    try:
        folder.mkdir(mode=0o700)
    except FileExistsError:
        pass
    directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(directory)
        if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("unsafe measurement lock directory ownership or permissions")
        name = hashlib.sha256(device.encode()).hexdigest() + ".lock"
        descriptor = os.open(
            name,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != uid
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise ValueError("unsafe measurement lock file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("measurement device is already in use") from error
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def process_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[19]


def remove_container(name):
    if not name.startswith("radar-perf-"):
        raise ValueError("invalid owned container receipt")
    result = subprocess.run(
        ["docker", "rm", "--force", name], capture_output=True, text=True, check=False, timeout=15
    )
    if result.returncode and "No such container" not in result.stderr:
        raise ValueError("owned container cleanup could not be verified")


def cleanup_owned(receipt):
    """Recover tool-owned leftovers without trusting a reused PID."""
    value = read_json(receipt)
    if value.get("state") != "running" and not value.get("cleanup_unknown"):
        return
    pid = value["pid"]
    if value.get("state") == "running":
        try:
            current = process_ticks(pid)
        except FileNotFoundError:
            current = None
        if current is not None:
            if current != value.get("process_start_ticks") or os.getpgid(pid) != pid:
                raise ValueError("owned PID identity cannot be verified")
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if value.get("container"):
        remove_container(value["container"])
    write_json(
        receipt,
        value | {"state": "interrupted", "cleanup_unknown": False, "recovered_at": time.time()},
    )


@contextmanager
def termination_guard():
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("measurement interrupted")

        signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def execute(argv, cwd, log, receipt, timeout, stdout_path=None):
    """Kill only this invocation's process group and owned container on interruption."""
    if Path(receipt).exists():
        cleanup_owned(receipt)
    container = None
    command = list(argv)
    if Path(command[0]).name == "docker" and len(command) > 1 and command[1] == "run":
        if any(x == "--name" or x.startswith("--name=") for x in command):
            raise ValueError("measurement containers must have tool-owned names")
        container = "radar-perf-" + uuid.uuid4().hex
        command[2:2] = ["--name", container]
    start = time.time()
    with termination_guard(), ExitStack() as stack:
        stream = stack.enter_context(Path(log).open("wb"))
        stdout = stack.enter_context(Path(stdout_path).open("wb")) if stdout_path else stream
        process = subprocess.Popen(
            command, cwd=cwd, stdout=stdout, stderr=stream, start_new_session=True
        )
        try:
            ticks = process_ticks(process.pid)
        except FileNotFoundError:
            ticks = None
        value = {
            "pid": process.pid,
            "process_start_ticks": ticks,
            "container": container,
            "started_at": start,
            "state": "running",
        }
        try:
            write_json(receipt, value)
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            try:
                if container:
                    remove_container(container)
            except (ValueError, subprocess.SubprocessError, OSError):
                value["cleanup_unknown"] = True
                raise
            finally:
                write_json(
                    receipt,
                    value
                    | {
                        "finished_at": time.time(),
                        "returncode": process.returncode,
                        "state": "finished",
                    },
                )


def run_adjacent(
    before, after, output, pairs, timeout_s, sampling, wall_time_s, expected_clocks, device
):
    from timing import aggregate_isolated, compatible, report_rows, verify_build

    sampling = protocol(sampling)
    compatible(before, after)
    if before.get("device_id", "cuda-visible-device-0") != after.get(
        "device_id", "cuda-visible-device-0"
    ):
        raise ValueError("A/B device identity differs")
    if pairs < 2:
        raise ValueError("at least two pairs required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "before": before,
        "after": after,
        "sampling": sampling,
        "requested_pairs": pairs,
        "measurement_policy_version": 3,
        "device": device,
    }
    path = output / "pairs.json"
    index = (
        read_json(path)
        if path.exists()
        else identity
        | {"pairs": [], "clocks": {"expected_configuration": expected_clocks, "snapshots": []}}
    )
    if any(index.get(k) != v for k, v in identity.items()):
        raise ValueError("resume measurement identity changed")
    if expected_clocks is None:
        expected_clocks = (
            index["clocks"]["expected_configuration"] or clock_state()["configuration"]
        )
    if index["clocks"]["expected_configuration"] not in (None, expected_clocks):
        raise ValueError("resume clock configuration changed")
    index["clocks"]["expected_configuration"] = expected_clocks
    deadline = (
        None
        if wall_time_s is None
        else time.monotonic() + finite(wall_time_s, "wall budget", positive=True)
    )

    def remaining():
        duration = timeout_s if deadline is None else min(timeout_s, deadline - time.monotonic())
        if duration <= 0:
            raise TimeoutError("measurement wall budget exhausted")
        return duration

    def clocks():
        snapshot = clock_state()
        if snapshot["configuration"] != expected_clocks:
            raise ValueError("clock/governor configuration changed")
        index["clocks"]["snapshots"].append(snapshot | {"timestamp": time.time()})

    with device_lock(device):
        inventories = []
        for build in (before, after):
            verify_build(build)
            command = list(build["command"]) + ["--list-cases", "--scope", sampling["scope"]]
            for key in ("cases", "sizes"):
                if key in sampling:
                    command += ["--" + key, sampling[key]]
            number = len(inventories)
            inventory_path = output / f"inventory-{number}.txt"
            execute(
                command,
                build["cwd"],
                output / f"inventory-{number}.log",
                output / f"inventory-{number}.process.json",
                remaining(),
                stdout_path=inventory_path,
            )
            inventories.append(
                [
                    line.split("\t")[0]
                    for line in inventory_path.read_text().splitlines()
                    if line.strip()
                ]
            )
        inventory = inventories[0]
        if not inventory or inventory != inventories[1] or len(set(inventory)) != len(inventory):
            raise ValueError("invalid/different case inventory")
        if set(sampling["case_overrides"]) - set(inventory):
            raise ValueError("override names an unmeasured case")
        write_json(path, index)
        for pair in range(pairs):
            records, children = {}, {"before": [], "after": []}
            for number, case in enumerate(inventory):
                cfg = effective(sampling, case)
                for repeat in range(cfg["repeats"]):
                    order = ("before", "after") if (pair + repeat) % 2 == 0 else ("after", "before")
                    for side in order:
                        build = index[side]
                        verify_build(build)
                        child = output / f"{pair:03d}-{side}-{number:03d}-{repeat}.json"
                        done = child.with_suffix(".receipt.json")
                        clocks()
                        if done.exists():
                            completed = read_json(done)
                            if not child.exists() or digest(child) != completed["sha256"]:
                                raise ValueError("completed measurement hash changed")
                        else:
                            if child.exists():
                                raise ValueError(
                                    "unsealed measurement exists; quarantine before retry"
                                )
                            command = list(build["command"]) + [
                                "--scope",
                                sampling["scope"],
                                "--cases",
                                case,
                                "--warmup",
                                str(cfg["warmup"]),
                                "--frames",
                                str(cfg["frames"]),
                                "--repeats",
                                "1",
                                "--raw-frame-times",
                                "--quiet",
                                "--output",
                                str(child),
                            ]
                            execute(
                                command,
                                build["cwd"],
                                child.with_suffix(".log"),
                                child.with_suffix(".process.json"),
                                remaining(),
                            )
                            derive_raw(read_json(child), sampling)
                            write_json(done, {"sha256": digest(child)})
                        raw = derive_raw(read_json(child), sampling)
                        report_rows(raw, build)
                        if {row["scenario_id"] for row in raw["results"]} != {case}:
                            raise ValueError("isolated subprocess returned another case")
                        children[side].append({"path": str(child), "sha256": digest(child)})
                        clocks()
                        write_json(path, index)
            for side in ("before", "after"):
                reports = [read_json(ref["path"]) for ref in children[side]]
                report = aggregate_isolated(reports, sampling, inventory)
                report["isolation"] = {"children": children[side], "inventory": inventory}
                merged = output / f"{pair:03d}-{side}.json"
                if merged.exists() and read_json(merged) != report:
                    raise ValueError("sealed pair aggregate changed")
                if not merged.exists():
                    write_json(merged, report)
                records[side] = {"path": str(merged), "sha256": digest(merged)}
            if pair < len(index["pairs"]):
                if records != index["pairs"][pair]:
                    raise ValueError("sealed pair references changed")
            else:
                index["pairs"].append(records)
            write_json(path, index)
    return index
