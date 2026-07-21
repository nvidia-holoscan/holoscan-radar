# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nsys runner for the operator performance-optimization harness.

Runs a ``perf_<operator>`` binary two ways and captures the artifacts ``parse``
folds into ``analysis.json``:

* **clean** — the binary alone (no profiler), the authoritative latency source
  (the schema-v2 perf JSON). One steady run per size, plus one override sweep
  across all sizes (the per-param ``override_delta_ms`` source).
* **profiled** — the same binary under ``nsys profile --trace=cuda,nvtx,osrt``,
  the authoritative *attribution* source (kernel / cudaMalloc / memcpy / NVTX
  phase split). nsys perturbs timing, so latency always comes from the clean
  run; attribution always from the profiled run.

Attribution is per-size (the steady path is identical across override params, so
one profiled steady run per size suffices; the per-param deltas ride the clean
override sweep). Everything is written under ``--out-dir`` (inside the mounted
repo tree, so the ``.nsys-rep`` files persist on the host).

Preflight asserts the toolchain is present — this must run inside
``./holohub run-container`` (the host has no ``nsys`` / built binary). Run:

    python performance/agent/runner.py --operator fmcw_range_processor \\
        --build-dir build-perf --sizes xs,md,xl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# nsys stat tables folded by parse.py. Report names verified against Nsight
# Systems 2026.2.1 (`nsys stats --help-reports`); the SQLite export
# (prof.sqlite) plus direct queries is the fallback if a name differs on
# another nsys version.
DEFAULT_REPORTS = (
    "nvtx_pushpop_sum",  # per-phase wall time (our compute() phase ranges)
    "nvtx_gpu_proj_sum",  # per-phase GPU-projected time (true kernel time)
    "cuda_gpu_kern_sum",  # GPU kernel time breakdown
    "cuda_api_sum",  # cudaMalloc / cudaFree / memcpy / sync API time
    "cuda_gpu_mem_time_sum",  # memcpy/memset H2D/D2H by time
)

NSYS_TRACE = "cuda,nvtx,osrt"


@dataclass
class SizeArtifacts:
    """Artifacts captured for one size."""

    size: str
    clean_json: str  # schema-v2 perf JSON (latency baseline)
    nsys_rep: str = ""  # .nsys-rep (empty if --no-nsys)
    stats_json: dict = field(default_factory=dict)  # report -> json path


@dataclass
class RunnerManifest:
    """Index of everything a runner invocation produced (parse.py reads this)."""

    operator: str
    binary: str
    build_dir: str
    out_dir: str
    sizes: list
    reports: list
    trace: str
    clean_override_sweep_json: str
    per_size: list = field(default_factory=list)  # list[SizeArtifacts as dict]
    nsys_version: str = ""
    notes: str = ""


def _log(msg: str) -> None:
    print(f"[perf-runner] {msg}", file=sys.stderr, flush=True)


def preflight(binary: Path, use_nsys: bool) -> str:
    """Assert the in-container toolchain is present. Returns the nsys version.

    The harness runs inside ./holohub run-container; the host has no toolchain.
    A clear failure here beats a confusing one mid-run.
    """
    problems = []
    if not binary.exists():
        problems.append(
            f"perf binary not found: {binary}\n"
            f"  Configure + build it first (inside ./holohub run-container):\n"
            f"    cmake -S . -B {binary.parents[2].name} "
            f"-DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON "
            f"-DHOLOSCAN_RADAR_PERF_NVTX=ON -DBUILD_ALL=ON\n"
            f"    cmake --build {binary.parents[2].name} --target {binary.name}"
        )
    nsys_version = ""
    if use_nsys:
        nsys = shutil.which("nsys")
        if nsys is None:
            problems.append(
                "nsys not on PATH — are you inside ./holohub run-container? "
                "(the host has no Nsight Systems CLI)"
            )
        else:
            try:
                out = subprocess.run(
                    [nsys, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                nsys_version = (
                    out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
                )
            except OSError as exc:  # pragma: no cover - defensive
                problems.append(f"nsys --version failed: {exc}")
    if problems:
        raise SystemExit(
            "[perf-runner] preflight failed:\n  - " + "\n  - ".join(problems)
        )
    return nsys_version


def run_clean(
    binary: Path,
    out_json: Path,
    *,
    sizes=None,
    sweep_overrides: bool = False,
    warmup: int,
    frames: int,
) -> None:
    """Run the perf binary without a profiler and capture its schema-v2 JSON."""
    cmd = [
        str(binary),
        "--quiet",
        "--output",
        str(out_json),
        "--warmup",
        str(warmup),
        "--frames",
        str(frames),
    ]
    if sweep_overrides:
        cmd.append("--sweep-overrides")
    if sizes:
        cmd += ["--sizes", ",".join(sizes)]
    _log("clean: " + " ".join(cmd))
    env = dict(os.environ, HOLOSCAN_LOG_LEVEL="ERROR")
    subprocess.run(cmd, check=True, env=env)


def run_profiled(
    binary: Path,
    rep_base: Path,
    *,
    size: str,
    warmup: int,
    frames: int,
    trace: str = NSYS_TRACE,
) -> Path:
    """Run the perf binary under nsys for one size. Returns the .nsys-rep path."""
    rep = rep_base.with_suffix(".nsys-rep")
    cmd = [
        "nsys",
        "profile",
        "--trace",
        trace,
        "--force-overwrite",
        "true",
        "--stats",
        "false",
        "-o",
        str(rep_base),
        str(binary),
        "--sizes",
        size,
        "--quiet",
        "--warmup",
        str(warmup),
        "--frames",
        str(frames),
    ]
    _log("profiled: " + " ".join(cmd))
    env = dict(os.environ, HOLOSCAN_LOG_LEVEL="ERROR")
    subprocess.run(cmd, check=True, env=env)
    return rep


def export_stats(rep: Path, reports=DEFAULT_REPORTS) -> dict:
    """Export nsys stat tables to clean per-report JSON files.

    Uses ``nsys stats --output <base>`` (not stdout): the file is pure JSON,
    whereas stdout is prefixed with "Generating SQLite / Processing" progress
    lines that themselves contain '[' and would defeat naive parsing.
    """
    base = rep.with_suffix("")  # <dir>/prof_size_md
    cmd = ["nsys", "stats"]
    for r in reports:
        cmd += ["--report", r]
    cmd += [
        "--format",
        "json",
        "--output",
        str(base),
        "--force-export",
        "true",
        str(rep),
    ]
    _log("stats: " + " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = {}
    for r in reports:
        path = base.parent / f"{base.name}_{r}.json"
        if path.exists():
            out[r] = str(path)
        else:
            _log(f"warning: expected stats file missing: {path}")
    return out


def run(
    operator: str,
    build_dir: Path,
    out_dir: Path,
    sizes,
    *,
    use_nsys: bool,
    warmup: int,
    frames: int,
    prof_warmup: int,
    prof_frames: int,
    reports=DEFAULT_REPORTS,
) -> Path:
    """Drive clean + profiled runs for one operator. Returns the manifest path."""
    binary = build_dir / "performance" / "cpp" / f"perf_{operator}"
    nsys_version = preflight(binary, use_nsys)

    op_out = out_dir / operator
    op_out.mkdir(parents=True, exist_ok=True)

    # Clean override sweep across all sizes: the per-param override_delta_ms
    # source (attribution does not need per-param profiles — the steady path is
    # identical across override params). Operators whose perf_<op> does not yet
    # implement --sweep-overrides (exit 2) are handled gracefully: the harness
    # still reports kernel/alloc/memcpy attribution and latency, just without the
    # per-param columns.
    sweep_json = op_out / "clean_override_sweep.json"
    have_sweep = True
    try:
        run_clean(
            binary,
            sweep_json,
            sizes=sizes,
            sweep_overrides=True,
            warmup=warmup,
            frames=frames,
        )
    except subprocess.CalledProcessError:
        have_sweep = False
        _log(
            f"note: {binary.name} does not implement --sweep-overrides; "
            "skipping the override sweep (per-size attribution only)"
        )

    per_size = []
    for size in sizes:
        clean_json = op_out / f"clean_size_{size}.json"
        run_clean(binary, clean_json, sizes=[size], warmup=warmup, frames=frames)
        art = SizeArtifacts(size=size, clean_json=str(clean_json))
        if use_nsys:
            rep = run_profiled(
                binary,
                op_out / f"prof_size_{size}",
                size=size,
                warmup=prof_warmup,
                frames=prof_frames,
            )
            art.nsys_rep = str(rep)
            art.stats_json = export_stats(rep, reports)
        per_size.append(asdict(art))

    manifest = RunnerManifest(
        operator=operator,
        binary=str(binary),
        build_dir=str(build_dir),
        out_dir=str(op_out),
        sizes=list(sizes),
        reports=list(reports),
        trace=NSYS_TRACE,
        clean_override_sweep_json=str(sweep_json) if have_sweep else "",
        per_size=per_size,
        nsys_version=nsys_version,
        notes="clean JSON = latency; profiled nsys tables = attribution; "
        "counts include the binary's discarded process-warmup session",
    )
    manifest_path = op_out / "runner_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    _log(f"wrote manifest: {manifest_path}")
    return manifest_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a perf_<operator> binary clean + under nsys and "
        "export the stat tables (inside ./holohub run-container)."
    )
    p.add_argument(
        "--operator",
        required=True,
        help="operator stem, e.g. fmcw_range_processor " "(binary perf_<operator>)",
    )
    p.add_argument(
        "--build-dir",
        default="build-perf",
        type=Path,
        help="CMake build dir holding performance/cpp/perf_<op>",
    )
    p.add_argument(
        "--out-dir",
        default=Path("performance/agent/runs"),
        type=Path,
        help="artifact output root (in the repo tree)",
    )
    p.add_argument(
        "--sizes",
        default="xs,md,xl",
        help="comma-separated size-ladder ids (small/medium/large)",
    )
    p.add_argument(
        "--warmup", type=int, default=5, help="clean-run warmup frames (default 5)"
    )
    p.add_argument(
        "--frames", type=int, default=20, help="clean-run measured frames (default 20)"
    )
    p.add_argument(
        "--profiled-warmup",
        type=int,
        default=3,
        help="profiled-run warmup frames (default 3)",
    )
    p.add_argument(
        "--profiled-frames",
        type=int,
        default=12,
        help="profiled-run measured frames (default 12; smaller "
        "keeps the trace bounded)",
    )
    p.add_argument(
        "--no-nsys", action="store_true", help="skip profiling (clean latency only)"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sizes = [s for s in args.sizes.split(",") if s]
    manifest = run(
        args.operator,
        args.build_dir,
        args.out_dir,
        sizes,
        use_nsys=not args.no_nsys,
        warmup=args.warmup,
        frames=args.frames,
        prof_warmup=args.profiled_warmup,
        prof_frames=args.profiled_frames,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
