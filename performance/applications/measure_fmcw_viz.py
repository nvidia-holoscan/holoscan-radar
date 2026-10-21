# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample matched MIMO TDM GUI runs inside the GPU development container.

Requires an existing DISPLAY (Xvfb is suitable), PyYAML, and nvidia-smi.
100% CPU means one logical core; GPU utilization and power are device-wide.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


def process_sample(pid):
    """Read process CPU seconds and resident bytes, including all its threads."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        rss = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if fields[0] == "Z" or rss == 0:
            return None
        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        return cpu, rss
    except (FileNotFoundError, ProcessLookupError):
        return None


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def measure(args, root, output, enabled, size, repeat):
    width, height = size
    name = f"{args.language}_{'on' if enabled else 'off'}_{width}x{height}_{repeat}"
    cfg = yaml.safe_load((root / "applications/fmcw/mimo_tdm/config/mimo_tdm.yaml").read_text())
    cfg["app"]["run"]["num_frames"] = args.frames
    cfg["app"]["visualization"].update(enabled=enabled, window_width=width, window_height=height)
    path = output / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    if args.language == "cpp":
        command = [str(args.build_dir / "applications/fmcw/mimo_tdm/cpp/mimo_tdm"), str(path)]
    else:
        command = [
            sys.executable,
            "-c",
            "import sys; from holoscan_radar.applications import mimo_tdm; "
            "mimo_tdm.build_graph(sys.argv[1]).run()",
            str(path),
        ]
    samples = []
    previous = None
    with (output / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now - start > args.frames * 0.1 + 60:
                    raise TimeoutError(name)
                sample = process_sample(proc.pid)
                if sample is not None:
                    cpu, rss = sample
                    if previous is not None and now - start >= args.warmup_seconds:
                        elapsed = now - previous[0]
                        samples.append(
                            {
                                "time": now,
                                "cpu_percent": 100 * (cpu - previous[1]) / elapsed,
                                "rss_mib": rss / 2**20,
                            }
                        )
                    previous = now, cpu
                time.sleep(0.2)
            if proc.returncode:
                raise RuntimeError(f"{name} exited {proc.returncode}; see its log")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
    # Exclude the last second to avoid counting Graph shutdown and pool release.
    if samples:
        cutoff = samples[-1]["time"] - 1.0
        samples = [sample for sample in samples if sample["time"] < cutoff]
    if not samples:
        raise RuntimeError(f"{name}: no steady samples; increase --frames")
    return {
        "name": name,
        "enabled": enabled,
        "size": size,
        "repeat": repeat,
        "wall_seconds": time.monotonic() - start,
        "samples": samples,
        "cpu_percent": summarize([s["cpu_percent"] for s in samples]),
        "rss_mib": summarize([s["rss_mib"] for s in samples]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build-gui"))
    parser.add_argument("--output", type=Path, default=Path("build-gui/viz-performance/resources"))
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-seconds", type=float, default=3)
    parser.add_argument("--language", choices=("python", "cpp"), default="python")
    parser.add_argument("--sizes", nargs="+", default=["1280x720"])
    parser.add_argument("--modes", nargs="+", choices=("off", "on"), default=["off", "on"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.build_dir = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(map(int, value.split("x"))) for value in args.sizes]
    (output / "metadata.json").write_text(
        json.dumps(
            {"unix_minus_monotonic": time.time() - time.monotonic(), "settings": vars(args)},
            default=str,
            indent=2,
        )
    )
    runs = []
    # Keep one low-rate NVML monitor for the complete experiment; its own CPU
    # and memory are excluded from the measured application process.
    with (output / "gpu.csv").open("w") as gpu:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=gpu,
        )
        try:
            for size in sizes:
                for repeat in range(args.repeats):
                    # Alternate paired order to reduce warm-cache/order bias.
                    for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                        if ("on" if enabled else "off") not in args.modes:
                            continue
                        run = measure(args, root, output, enabled, size, repeat)
                        runs.append(run)
                        (output / "resources.json").write_text(json.dumps(runs, indent=2))
                        print(run["name"], run["cpu_percent"], run["rss_mib"], flush=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == "__main__":
    main()
