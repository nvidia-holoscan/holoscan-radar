# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Before/after runtime comparison across two build trees.

Runs the same ``perf_<operator>`` binaries out of two different build
directories with byte-identical arguments and diffs their steady-state
latency, scenario by scenario. Intended for "what did this optimization
branch actually buy us" — not for bottleneck attribution (that is
``runner.py`` + ``diagnose.py``, which need nsys).

The premise is that both build trees carry the *same* harness sources, so
the scenario ladders, sync policy and measurement mempool settings match;
only the operator implementations differ. Build the baseline side by
overlaying the current ``performance/`` onto the baseline checkout before
configuring it (see the README). If the two reports disagree on which
scenarios exist, the mismatched ones are reported as ``missing`` rather than
silently dropped.

Each round runs both sides back to back, and one extra round is measured and
discarded up front so the cold-clock / cold-page-cache ramp lands outside the
statistic. The per-scenario figure is the *median* of the per-launch medians
across scored rounds, reported next to the launch-to-launch spread; a delta
inside that spread is flagged as noise rather than presented as a result. See
``compare_operator`` for why this shape, and the README for the identity-run
measurements that forced it.

Usage::

    python performance/agent/compare_builds.py \\
        --before-build build-before/build --after-build build \\
        --operators fmcw_range_processor,cfar_detector --sizes sm,md,xl

Writes ``compare.json`` + ``compare.md`` under ``--out-dir`` and prints a
table to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Scenario rows are emitted once per bucket; steady-state latency lives on the
# "compute" row (setup/teardown rows carry null timings).
_COMPUTE_BUCKET = "compute"

_CHECKSUM_RE = re.compile(r"output_checksum\s+(\d+)\s+([0-9a-fA-F]+)")


# --------------------------------------------------------------------------
# Before-side cache
#
# The "before" tree is pinned to a commit and its binaries never change, so
# re-measuring it every run is pure waste — about half the wall clock. But a
# cached baseline is, by definition, measured in a *different session* from the
# "after" side it is compared against, and cross-session comparison is exactly
# what the interleaving exists to avoid: identity runs showed ~10% bias for a
# fixed order and ~24% for ABBA alternation. Reusing a baseline from a machine
# in a different thermal/clock state silently reintroduces that.
#
# Hence the canary: every cached run re-measures the cheapest operators on both
# sides and checks the fresh baseline still agrees with the cached one. If it
# does not, the machine has moved and the cache is dropped for that run. This
# converts a silent bias into a detected one, at a cost of a few seconds.
#
# The key deliberately hashes the *binary*, not the commit SHA: a dirty tree,
# different cmake flags or a different toolchain all change timings without
# changing the SHA.
# --------------------------------------------------------------------------

_CACHE_VERSION = 2


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(binary: Path, args: list[str], rounds: int, machine: str) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}\0".encode())
    h.update(_file_sha256(binary).encode())
    h.update("\0".join(args).encode())
    h.update(f"\0rounds={rounds}\0{machine}".encode())
    return h.hexdigest()[:20]


def _machine_fingerprint() -> str:
    """GPU name + driver — enough to invalidate across hosts."""
    if not shutil.which("nvidia-smi"):
        return "unknown"
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    out = p.stdout.strip().splitlines()
    return out[0].strip() if out else "unknown"


def _cache_load(cache_dir: Path, op: str, key: str) -> dict | None:
    path = cache_dir / f"{op}.{key}.json"
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if entry.get("cache_version") != _CACHE_VERSION:
        return None
    return entry


def _cache_store(cache_dir: Path, op: str, key: str, entry: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{op}.{key}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=2))
    tmp.replace(path)  # atomic, so a killed run cannot leave a torn entry


def _concurrent_runs() -> list[tuple[int, str]]:
    """Other live compare_builds.py processes, as (pid, command line).

    Two of these running at once is worse than a clobbered output directory:
    they contend for the same GPU, and on a box whose launch-to-launch noise
    floor is already 5-18% that contention silently swamps the effect being
    measured. Best effort — /proc only, and a no-op elsewhere.
    """
    me = os.getpid()
    hits: list[tuple[int, str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return hits
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue  # process exited, or not ours to read
        parts = [c for c in raw.decode("utf-8", "replace").split("\0") if c]
        # Match the script as an argv element, so shell wrappers that merely
        # quote the whole pipeline in one arg do not count as a second run.
        if any(c.endswith("compare_builds.py") for c in parts):
            hits.append((pid, " ".join(parts)))
    return hits


def _warn_concurrent(out_dir: Path) -> None:
    """Loudly flag other live runs. Warn-only: never blocks the caller."""
    for pid, cmd in _concurrent_runs():
        # A run with no explicit --out-dir lands on the same default as this
        # one, so the two will also overwrite each other's report.
        shares_out = "--out-dir" not in cmd or str(out_dir) in cmd
        lines = [
            f"WARNING: another compare_builds.py is already running " f"(pid {pid}):",
            f"           {cmd[:160]}",
            "         Both runs share this GPU. Contended timings are not",
            "         comparable, and on this box the noise floor is already",
            "         5-18%, so the contention can swamp the effect.",
        ]
        if shares_out:
            lines.append(f"         They also both write {out_dir} — last writer wins.")
        lines.append("         Proceeding anyway (warn-only).")
        print("\n".join(lines), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Thermal guard
#
# Measured on this Thor box: two minutes of sustained mimo lg/xl load moved the
# GPU from 36 C to a 38.9 C peak and it settled back to 37 C. So thermal
# throttling is NOT the source of the launch-to-launch variance here, and the
# default threshold below is set high enough never to fire on this machine.
#
# It is kept anyway because it is nearly free and the harness is meant to be
# portable: a passively cooled or enclosed unit can absolutely throttle, and
# there the guard turns a silent 20% clock drop into a visible wait. Every
# launch records its temperature either way, so the question is answerable from
# the report instead of assumed.
#
# nvidia-smi reports temperature.gpu on Tegra, but clocks.current.sm and
# clocks_throttle_reasons.active both come back [N/A] — so throttle state cannot
# be read directly and temperature is the only available proxy. The sysfs
# thermal zones are cheaper (no subprocess) and are tried first.
# --------------------------------------------------------------------------

_THERMAL_ZONES = Path("/sys/devices/virtual/thermal")


def _gpu_temp_c() -> float | None:
    """Current GPU temperature in Celsius, or None if unreadable."""
    if _THERMAL_ZONES.is_dir():
        for zone in sorted(_THERMAL_ZONES.glob("thermal_zone*")):
            try:
                if (zone / "type").read_text().strip() != "gpu-thermal":
                    continue
                return int((zone / "temp").read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    if shutil.which("nvidia-smi"):
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return float(p.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None
    return None


def _await_cool(max_temp: float, timeout_s: float, note) -> float:
    """Block until the GPU is at or below max_temp. Returns seconds waited.

    Bounded by timeout_s so an unreadable sensor or a genuinely hot room can
    never hang a run indefinitely — it proceeds and says so, because a warm
    measurement is more useful than no measurement.
    """
    if max_temp <= 0:
        return 0.0
    t0 = time.monotonic()
    warned = False
    while True:
        t = _gpu_temp_c()
        if t is None or t <= max_temp:
            break
        waited = time.monotonic() - t0
        if waited >= timeout_s:
            note(
                f"[thermal] still {t:.0f}C after {_fmt_dur(waited)} "
                f"(limit {max_temp:.0f}C) — proceeding anyway"
            )
            break
        if not warned:
            note(f"[thermal] {t:.0f}C > {max_temp:.0f}C, waiting to cool ...")
            warned = True
        time.sleep(2.0)
    return time.monotonic() - t0


def _fmt_dur(seconds: float) -> str:
    """Compact duration: 45s, 12m04s, 1h07m."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class _Progress:
    """Launch-granular progress + self-correcting ETA.

    The unit of work is one perf-binary launch, and the total is known up
    front: ``operators x (rounds + 1 warm-up) x 2 sides``. That makes a real
    bar and a real ETA possible, rather than just naming the current operator.

    The ETA is *not* a flat mean over all launches — operators differ by well
    over an order of magnitude (a split-vs-fused scenario matrix dwarfs a
    deinterleave). Remaining launches for the operator in flight are costed
    from that operator's own observed launches; operators not yet started are
    costed from the global mean. Early estimates are rough and tighten as
    operators complete.
    """

    def __init__(self, plan: list[tuple[str, int, float | None]], style: str):
        # plan is [(operator, launches, est_seconds_per_launch|None)].
        # Launches vary because a cached baseline halves an operator's work;
        # est comes from the probe pass or the cache, and is None when neither
        # had anything to say.
        self.plan = plan
        self.total_ops = len(plan)
        self.per_op = plan[0][1] if plan else 0
        self.total = sum(n for _, n, _e in plan)
        self.done = 0
        # (predicted, actual) per completed operator, for calibrating the
        # estimates: the probe measures per-frame cost cold, so it tends to
        # run high, and a single global factor corrects most of that.
        self._calib: list[tuple[float, float]] = []
        self.temps: list[float] = []  # GPU temp sampled per launch
        self.cool_s = 0.0  # total seconds spent waiting to cool
        self.t0 = time.monotonic()
        self.op_times: list[float] = []  # current operator's launch durations
        self.all_times: list[float] = []
        self.op_name = ""
        self.op_index = 0
        self.style = style
        self._dirty = False

    # -- estimation --------------------------------------------------------
    def _eta(self) -> float | None:
        if not self.all_times:
            return None
        global_mean = statistics.fmean(self.all_times)
        op_mean = statistics.fmean(self.op_times) if self.op_times else None
        k = self._calibration()
        # Launches left on the operator currently in flight: prefer its own
        # observed rate, fall back to its calibrated estimate.
        left_here = self.per_op - len(self.op_times)
        here_rate = op_mean or self._rate(self.op_index, k, global_mean)
        # Operators not yet started are costed from their probe/cache estimate
        # where one exists; without it there is no per-operator prior and a
        # skewed workload is under-estimated until it is reached.
        left_after = sum(
            n * self._rate(i, k, global_mean)
            for i, (_o, n, _e) in enumerate(self.plan[self.op_index :], start=self.op_index + 1)
        )
        return left_here * here_rate + left_after

    def _rate(self, index: int, k: float, fallback: float) -> float:
        """Calibrated per-launch estimate for the 1-based index'th operator."""
        if 1 <= index <= len(self.plan):
            est = self.plan[index - 1][2]
            if est:
                return est * k
        return fallback

    def _calibration(self) -> float:
        """median(actual / predicted) over completed operators; 1.0 if none."""
        ratios = [a / p for p, a in self._calib if p > 0 and a > 0]
        return statistics.median(ratios) if ratios else 1.0

    # -- reporting ---------------------------------------------------------
    def _line(self) -> str:
        frac = self.done / self.total if self.total else 1.0
        bar_w = 22
        filled = int(round(frac * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        eta = self._eta()
        eta_s = f"eta ~{_fmt_dur(eta)}" if eta is not None else "eta --"
        return (
            f"[{self.done:>3}/{self.total}] {bar} {frac * 100:3.0f}% | "
            f"{self.op_name} {len(self.op_times)}/{self.per_op} "
            f"(op {self.op_index}/{self.total_ops}) | "
            f"elapsed {_fmt_dur(time.monotonic() - self.t0)} | {eta_s}"
        )

    def _emit(self, final: bool = False) -> None:
        if self.style == "none":
            return
        if self.style == "bar":
            sys.stdout.write("\r\033[2K" + self._line())
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
            self._dirty = not final
        else:  # plain — one line per event, safe for log files
            print(self._line(), flush=True)

    def start_operator(self, index: int) -> None:
        """Begin the 1-based index'th operator of the plan."""
        self._close_operator()
        self.op_name, self.per_op, self._op_est = self.plan[index - 1]
        self.op_index = index
        self.op_times = []
        self._emit()

    def _close_operator(self) -> None:
        """Feed the operator just finished into the calibration sample."""
        est = getattr(self, "_op_est", None)
        if est and self.op_times:
            self._calib.append((est, statistics.fmean(self.op_times)))

    def launch_done(self, seconds: float) -> None:
        self.done += 1
        self.op_times.append(seconds)
        self.all_times.append(seconds)
        self._emit()

    def drop_operator(self, launches_skipped: int) -> None:
        """An operator failed; shrink the denominator so the bar stays honest."""
        self.total = max(self.done, self.total - launches_skipped)
        self._emit()

    def revise_plan(self, plan: list[tuple[str, int, float | None]]) -> None:
        """Replace the remaining plan (e.g. the canary invalidated the cache)."""
        self.plan = plan
        self.total_ops = len(plan)
        self.total = sum(n for _, n, _e in plan)

    def record_temp(self, t: float | None) -> None:
        if t is not None:
            self.temps.append(t)

    def cooled(self, seconds: float) -> None:
        self.cool_s += seconds

    def note(self, msg: str) -> None:
        """Print a message without clobbering an in-place bar."""
        if self.style == "bar" and self._dirty:
            sys.stdout.write("\r\033[2K")
            self._dirty = False
        print(msg, flush=True)

    def finish(self) -> None:
        self._close_operator()
        self._emit(final=True)


def _perf_binaries(build_dir: Path) -> dict[str, Path]:
    """Map operator name -> perf binary under ``build_dir``."""
    cpp = build_dir / "performance" / "cpp"
    if not cpp.is_dir():
        return {}
    out = {}
    for p in sorted(cpp.glob("perf_*")):
        if p.is_file() and p.stat().st_mode & 0o111:
            out[p.name[len("perf_") :]] = p
    return out


# Set by main() from --pin-cpus. Prefixed onto every perf-binary launch, so
# both sides are pinned identically. Measured effect on this box (strict
# alternation, warm-up dropped): IQR 5.24% -> 2.63%, peak-to-peak 64.8% ->
# 11.2%. It also makes the run ~4.6% faster in absolute terms, so it is part of
# the cache key: a cached unpinned baseline must not be compared against a
# pinned after-side.
_PIN_PREFIX: list[str] = []


def _validate_pin(cpus: str) -> list[str]:
    """Return a taskset prefix for `cpus`, or [] with a warning if unusable."""
    if not cpus:
        return []
    if not shutil.which("taskset"):
        print("warning: --pin-cpus given but taskset not found; not pinning", file=sys.stderr)
        return []
    # Refuse to pin outside the cgroup's cpuset, where taskset would just fail
    # per launch and every measurement would die.
    try:
        allowed = Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()
    except OSError:
        allowed = ""
    if allowed:

        def expand(spec):
            out = set()
            for part in spec.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    out |= set(range(int(a), int(b) + 1))
                elif part:
                    out.add(int(part))
            return out

        try:
            if not expand(cpus) <= expand(allowed):
                print(
                    f"warning: --pin-cpus {cpus} is outside the cgroup "
                    f"cpuset {allowed}; not pinning",
                    file=sys.stderr,
                )
                return []
        except ValueError:
            pass
    return ["taskset", "-c", cpus]


def _run_one(binary: Path, args: list[str], checksum: bool) -> tuple[dict, list]:
    """Run one perf binary; return (parsed JSON report, checksum list)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.json"
        cmd = [*_PIN_PREFIX, str(binary), "--quiet", "--output", str(out), *args]
        if checksum:
            cmd.append("--checksum")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"{binary.name} failed (exit {proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        report = json.loads(out.read_text())
    sums = [(int(m.group(1)), m.group(2).lower()) for m in _CHECKSUM_RE.finditer(proc.stdout)]
    return report, sums


def _compute_rows(report: dict) -> dict[str, dict]:
    """Index the compute-bucket result rows by scenario_id."""
    rows = {}
    for r in report.get("results", []):
        if r.get("bucket") != _COMPUTE_BUCKET:
            continue
        if r.get("status") != "ok" or r.get("median_ms") is None:
            continue
        rows[r["scenario_id"]] = r
    return rows


def _measure(
    binary: Path,
    args: list[str],
    rounds: int,
    checksum: bool,
    progress: _Progress | None = None,
    thermal: tuple[float, float] | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict], list]:
    """Return (scenario -> [median_ms per round], scenario -> last row, sums)."""
    medians: dict[str, list[float]] = {}
    last_rows: dict[str, dict] = {}
    sums: list = []
    for _ in range(rounds):
        if thermal is not None and progress is not None:
            progress.cooled(_await_cool(thermal[0], thermal[1], progress.note))
        if progress is not None:
            progress.record_temp(_gpu_temp_c())
        t0 = time.monotonic()
        try:
            report, s = _run_one(binary, args, checksum)
        finally:
            # Count the launch even when it failed: the wall time was spent,
            # and the caller drops the operator's remaining launches instead.
            if progress is not None:
                progress.launch_done(time.monotonic() - t0)
        sums = s or sums
        for sid, row in _compute_rows(report).items():
            medians.setdefault(sid, []).append(row["median_ms"])
            last_rows[sid] = row
    return medians, last_rows, sums


def _predict_launch_seconds(report: dict, warmup: int, frames: int, repeats: int) -> float:
    """Predict the *measured* part of a launch from a cheap probe's report.

    Extrapolating the probe's wall time would be wrong: a launch costs
    ``sum_scenarios(setup + (warmup + frames*repeats) * per_frame)``, and setup
    (FFT plan builds, allocations) is a fixed per-scenario cost that dominates a
    2-frame probe. So predict analytically instead — the probe's report already
    carries ``setup_ms`` and ``median_ms`` per scenario, which is exactly the
    decomposition needed. Only ``median_ms`` is rough (measured cold), and that
    is calibrated out against observed launches as the run proceeds.

    This covers only what the report accounts for. Process overhead — CUDA
    context init, Holoscan graph construction, teardown — never appears in the
    report and dominates cheap operators (ignoring it under-predicted by ~4x on
    a 4-operator xs,sm,md run), so the caller measures and adds it separately.
    """
    setup: dict[str, float] = {}
    per_frame: dict[str, float] = {}
    for r in report.get("results", []):
        sid = r.get("scenario_id")
        if sid is None:
            continue
        if r.get("setup_ms") is not None:
            setup[sid] = setup.get(sid, 0.0) + r["setup_ms"]
        if r.get("bucket") == _COMPUTE_BUCKET and r.get("median_ms") is not None:
            per_frame[sid] = r["median_ms"]
    n = warmup + frames * max(1, repeats)
    total_ms = sum(
        setup.get(sid, 0.0) + n * per_frame.get(sid, 0.0) for sid in set(setup) | set(per_frame)
    )
    return total_ms / 1000.0


def _probe_operators(
    ops: list[str],
    probe_bins: dict[str, Path],
    sweep_flags: list[str],
    sizes: str,
    warmup: int,
    frames: int,
    repeats: int,
) -> tuple[dict[str, float], float]:
    """Cheap pre-pass estimating each operator's per-launch cost.

    Probes with the *same* ``--sizes`` as the real run, so the scenario x size
    structure that makes some operators expensive is preserved — a cheap
    ``--sizes xs`` probe would miss exactly that.

    Only one side is probed, which halves the cost. That is safe despite the
    asymmetry it appears to introduce: round 0 is measured and discarded, so
    every scored launch of *either* binary is already preceded by a launch of
    that same binary in this session, and the probe's extra warming is absorbed
    before anything counts. GPU-side state (clocks, persistence) is global and
    affects both sides regardless.

    ``--warmup 1 --frames 2`` is close to the accuracy floor: at ``--warmup 0``
    a single cold frame measured 6.96 ms where steady state was 0.25 ms, which
    would over-predict by more than an order of magnitude.
    """
    probe_args = ["--warmup", "1", "--frames", "2"]
    if sizes:
        probe_args += ["--sizes", sizes]
    probe_args += sweep_flags

    est: dict[str, float] = {}
    t0 = time.monotonic()
    for op in ops:
        t_probe = time.monotonic()
        try:
            report, _ = _run_one(probe_bins[op], probe_args, checksum=False)
        except RuntimeError:
            continue  # unprobeable operator falls back to the global mean
        wall = time.monotonic() - t_probe
        # Whatever the probe's wall time exceeds its own accounted-for work by
        # is process overhead (context init, graph build, teardown). That is a
        # per-launch constant, so carry it into the prediction rather than
        # scaling it with frame count.
        accounted = _predict_launch_seconds(report, 1, 2, 1)
        overhead = max(0.0, wall - accounted)
        est[op] = overhead + _predict_launch_seconds(report, warmup, frames, repeats)
    return est, time.monotonic() - t0


def _geomean(values: list[float]) -> float:
    return statistics.geometric_mean(values) if values else float("nan")


def compare_operator(
    op: str,
    before_bin: Path,
    after_bin: Path,
    before_args: list[str],
    after_args: list[str],
    rounds: int,
    checksum: bool,
    progress: _Progress | None = None,
    cached_before: dict | None = None,
    thermal: tuple[float, float] | None = None,
    warmup_rounds: int = 2,
) -> dict:
    """Run one operator on both builds and diff the compute-bucket medians.

    ``cached_before`` supplies a previously measured baseline; when present the
    before binary is not launched at all and only the after side runs. The
    result is marked ``before_cached`` so the report can say so.
    """
    if cached_before is not None:
        return _compare_with_cached_before(
            op,
            after_bin,
            after_args,
            rounds,
            checksum,
            progress,
            cached_before,
            thermal,
            warmup_rounds,
        )

    before_med: dict[str, list[float]] = {}
    after_med: dict[str, list[float]] = {}
    before_rows: dict[str, dict] = {}
    after_rows: dict[str, dict] = {}
    before_sums: list = []
    after_sums: list = []

    # Measurement methodology, all three points established empirically by
    # running this script with --before-build == --after-build (identity runs,
    # where the true delta is 0 by construction; see README).
    #
    # 1. The dominant variance is *between processes*, not between frames: two
    #    launches of the same binary measured 0.367 vs 0.280 ms on the same
    #    scenario, while each launch already medians 90 frames internally. More
    #    --frames therefore does not tighten the comparison; more launches does.
    # 2. The first launch pair is systematically slow (cold clocks, cold page
    #    cache). Round 0 is measured and discarded so the ramp lands outside the
    #    statistic instead of inside whichever side happened to run first — a
    #    fixed before-then-after order without this showed a ~10% bias, and
    #    ABBA-style alternation was worse (~24%) because it lets the middle side
    #    run back-to-back and inherit its own warm state.
    # 3. min-across-rounds rewards whichever side got the luckiest launch. The
    #    median across launches is the robust summary; the per-side spread is
    #    reported next to it so a delta inside the noise band reads as such.
    for r in range(rounds + warmup_rounds):
        bm, br, bs = _measure(before_bin, before_args, 1, checksum, progress, thermal)
        am, ar, as_ = _measure(after_bin, after_args, 1, checksum, progress, thermal)
        if r < warmup_rounds:
            continue  # discarded warm-up pair(s)
        for sid, v in bm.items():
            before_med.setdefault(sid, []).extend(v)
        for sid, v in am.items():
            after_med.setdefault(sid, []).extend(v)
        before_rows.update(br)
        after_rows.update(ar)
        before_sums = bs or before_sums
        after_sums = as_ or after_sums

    return _summarize(
        op,
        before_med,
        after_med,
        before_rows,
        after_rows,
        before_sums,
        after_sums,
        checksum,
        before_cached=False,
    )


def _spread_pct(vals: list[float]) -> float:
    """Robust spread of a side's per-launch medians, as % of median.

    Interquartile range, not peak-to-peak. Measured on this box, 20 consecutive
    launches of one binary had an IQR of 2.2% and a peak-to-peak of 113.6%: the
    distribution is tight with rare outliers (the CPU governor leaves cores at
    972 MHz after an idle gap and takes a launch or two to reach 2601 MHz).
    Peak-to-peak lets a single such launch define the whole band, which is what
    produced the +/-636% noise figures — and since the significance test
    compares against this band, it was suppressing real effects.

    Falls back to peak-to-peak below four samples, where quartiles are
    meaningless.
    """
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    if med <= 0:
        return 0.0
    v = sorted(vals)
    if len(v) < 4:
        return (v[-1] - v[0]) / med * 100.0
    q1 = v[int(0.25 * (len(v) - 1))]
    q3 = v[int(0.75 * (len(v) - 1))]
    return (q3 - q1) / med * 100.0


def _summarize(
    op: str,
    before_med: dict[str, list[float]],
    after_med: dict[str, list[float]],
    before_rows: dict[str, dict],
    after_rows: dict[str, dict],
    before_sums: list,
    after_sums: list,
    checksum: bool,
    before_cached: bool,
) -> dict:
    """Fold both sides' per-launch medians into the per-scenario verdicts."""
    scenarios = []
    ratios = []
    for sid in sorted(set(before_med) | set(after_med)):
        b = statistics.median(before_med[sid]) if sid in before_med else None
        a = statistics.median(after_med[sid]) if sid in after_med else None
        row = after_rows.get(sid) or before_rows.get(sid) or {}
        entry = {
            "scenario_id": sid,
            "size_id": row.get("size_id"),
            "variant": row.get("variant"),
            "before_ms": b,
            "after_ms": a,
        }
        if b is None or a is None:
            entry["status"] = "missing_before" if b is None else "missing_after"
        else:
            entry["status"] = "ok"
            entry["delta_ms"] = a - b
            entry["delta_pct"] = (a - b) / b * 100.0
            entry["speedup"] = b / a
            # Noise band: the larger per-side launch-to-launch spread. A delta
            # inside it is not evidence of a change in either direction.
            noise = max(_spread_pct(before_med[sid]), _spread_pct(after_med[sid]))
            entry["noise_pct"] = noise
            entry["significant"] = abs(entry["delta_pct"]) > noise
            # Raw per-launch medians, so a question about the noise (outlier vs
            # broad spread, positional vs random) can be answered from a saved
            # report instead of requiring a fresh multi-hour run.
            entry["before_launches"] = [round(v, 6) for v in before_med[sid]]
            entry["after_launches"] = [round(v, 6) for v in after_med[sid]]
            ratios.append(b / a)
            # Cold (first-frame) latency often carries the plan/alloc cost the
            # optimization targeted; surface it but don't fold it into the
            # steady-state summary.
            cb = before_rows.get(sid, {}).get("cold_ms")
            ca = after_rows.get(sid, {}).get("cold_ms")
            if cb and ca:
                entry["cold_before_ms"] = cb
                entry["cold_after_ms"] = ca
                entry["cold_delta_pct"] = (ca - cb) / cb * 100.0
        scenarios.append(entry)

    checksum_match = None
    if checksum and (before_sums or after_sums):
        checksum_match = before_sums == after_sums

    # Geomean over *significant* scenarios only. The plain geomean folds in
    # scenarios whose delta is inside their own noise band, which on a 3-size
    # run is enough to make an untouched operator read as 0.80x or 1.43x.
    sig_ratios = [
        s["before_ms"] / s["after_ms"]
        for s in scenarios
        if s.get("status") == "ok" and s.get("significant")
    ]
    return {
        "operator": op,
        "scenarios": scenarios,
        "geomean_speedup": _geomean(ratios),
        "geomean_significant": _geomean(sig_ratios) if sig_ratios else None,
        "n_significant": len(sig_ratios),
        "checksum_match": checksum_match,
        "checksums_before": before_sums if checksum else None,
        "checksums_after": after_sums if checksum else None,
        # True when the baseline came from a previous session rather than
        # being interleaved with the after side in this one.
        "before_cached": before_cached,
        "_before_med": before_med,
        "_before_rows": before_rows,
        "_before_sums": before_sums,
    }


def _canary_verdict(fresh: dict, cached: dict, floor_pct: float = 10.0) -> dict:
    """Does a freshly measured baseline still match the cached one?

    Compares per-scenario medians. The tolerance is the scenario's own cached
    launch-to-launch spread, floored at ``floor_pct`` so a scenario that
    happened to be quiet in the cached run does not produce a hair-trigger.
    Any scenario outside tolerance condemns the whole cache: a shifted machine
    shifts every operator, so a single clear disagreement is enough.
    """
    worst_sid, worst = None, 0.0
    checked = 0
    cached_med = cached.get("medians", {})
    for s in fresh["scenarios"]:
        sid = s["scenario_id"]
        vals = cached_med.get(sid)
        if s.get("status") != "ok" or not vals or s.get("before_ms") is None:
            continue
        base = statistics.median(vals)
        if base <= 0:
            continue
        checked += 1
        delta = abs(s["before_ms"] - base) / base * 100.0
        tol = max(_spread_pct(vals), floor_pct)
        if delta - tol > worst:
            worst, worst_sid = delta - tol, sid
    if checked == 0:
        return {
            "ok": False,
            "message": "no comparable scenarios in cache — treating as stale",
        }
    if worst_sid is None:
        return {
            "ok": True,
            "message": f"fresh baseline matches cache across {checked} "
            f"scenario(s); reusing cached baselines",
        }
    return {
        "ok": False,
        "message": f"fresh baseline disagrees with cache on {worst_sid} "
        f"by {worst:.1f}pp beyond tolerance — machine state has moved",
    }


def _compare_with_cached_before(
    op: str,
    after_bin: Path,
    args: list[str],
    rounds: int,
    checksum: bool,
    progress: _Progress | None,
    cached: dict,
    thermal: tuple[float, float] | None = None,
    warmup_rounds: int = 2,
) -> dict:
    """Measure only the after side; take the baseline from ``cached``.

    Same statistic as the interleaved path, so the two are comparable — the
    difference is solely that the baseline was not measured in this session.
    """
    after_med: dict[str, list[float]] = {}
    after_rows: dict[str, dict] = {}
    after_sums: list = []
    for r in range(rounds + warmup_rounds):
        am, ar, as_ = _measure(after_bin, args, 1, checksum, progress, thermal)
        if r < warmup_rounds:
            continue  # discarded, same as the interleaved path
        for sid, v in am.items():
            after_med.setdefault(sid, []).extend(v)
        after_rows.update(ar)
        after_sums = as_ or after_sums

    return _summarize(
        op,
        {k: list(v) for k, v in cached["medians"].items()},
        after_med,
        cached.get("rows", {}),
        after_rows,
        [tuple(c) for c in cached.get("checksums", [])],
        after_sums,
        checksum,
        before_cached=True,
    )


def _fmt(v, spec="{:.3f}"):
    return "—" if v is None else spec.format(v)


def render_markdown(result: dict) -> str:
    lines = [
        "# Before/after runtime comparison",
        "",
        f"* **before** — `{result['before_build']}` @ " f"`{result['before_commit']}`",
        f"* **after** — `{result['after_build']}` @ " f"`{result['after_commit']}`",
        *(
            [
                "* **side-specific flags** — before: "
                f"`{' '.join(result.get('before_args') or []) or '(none)'}`, "
                f"after: `{' '.join(result.get('after_args') or []) or '(none)'}`"
            ]
            if (result.get("before_args") or result.get("after_args"))
            else []
        ),
        f"* harness args — `{' '.join(result['args']) or '(defaults)'}`, "
        f"{result['rounds']} scored round(s) + "
        f"{result.get('warmup_rounds', 1)} discarded warm-up",
        f"* GPU — {result.get('gpu', 'unknown')}",
        *(
            [
                "* thermal — observed "
                f"{result['thermal']['observed_min_c']:.0f}-"
                f"{result['thermal']['observed_max_c']:.0f}C across "
                f"{result['thermal']['samples']} launches, limit "
                f"{result['thermal']['max_temp_c']:.0f}C, "
                f"{result['thermal']['cooldown_seconds']:.0f}s spent cooling"
            ]
            if result.get("thermal", {}).get("observed_max_c") is not None
            else []
        ),
        "",
        "Metric is the compute-bucket `median_ms` (steady-state per-frame "
        "latency), median across rounds; one extra warm-up round is measured "
        "and discarded. Negative Δ% is faster. `±noise %` is the "
        "launch-to-launch spread of the noisier side — a Δ inside it is "
        "marked _(noise)_ and is not evidence of a change.",
        "",
    ]
    n_cached = sum(1 for o in result["operators"] if o.get("before_cached"))
    if n_cached:
        lines += [
            f"> **{n_cached} of {len(result['operators'])} operators used a "
            "cached baseline** (marked `[cached baseline]`). Those before/after "
            "pairs were *not* measured in the same session, so their delta can "
            "absorb machine drift that the interleaved pairs cancel. A canary "
            "operator was re-measured on both sides and agreed with the cache, "
            "but the `±noise %` column still only reflects within-session "
            "spread. Re-run with `--no-cache` to confirm anything surprising.",
            "",
        ]
    lines += _tiered_table(result["operators"])
    return "\n".join(lines)


# Ladder order, so sizes read small-to-large rather than alphabetically.
_SIZE_ORDER = ["xs", "sm", "md", "lg", "xl"]


def _size_key(size_id: str) -> tuple[int, str]:
    try:
        return (_SIZE_ORDER.index(size_id), "")
    except ValueError:
        return (len(_SIZE_ORDER), size_id)


def _rollup(scenarios: list[dict]) -> dict:
    """Aggregate a group of leaf scenarios into one summary row.

    Two different questions, so two different numbers. Summed before/after ms
    answers "how much wall time did this group cost", which is dominated by the
    largest size — that is the point, it is where the time actually goes. The
    geomean speedup answers "how much did a typical scenario move", weighting
    every size equally. Reporting only one of them misleads.
    """
    ok = [s for s in scenarios if s.get("status") == "ok"]
    if not ok:
        return {}
    b = sum(s["before_ms"] for s in ok)
    a = sum(s["after_ms"] for s in ok)
    sig = [s for s in ok if s.get("significant")]
    return {
        "before_ms": b,
        "after_ms": a,
        "delta_pct": (a - b) / b * 100.0 if b else None,
        "geomean": _geomean([s["before_ms"] / s["after_ms"] for s in ok]),
        "geomean_sig": (_geomean([s["before_ms"] / s["after_ms"] for s in sig]) if sig else None),
        "n_sig": len(sig),
        "n": len(ok),
    }


def _tiered_table(operators: list[dict]) -> list[str]:
    """Operator -> scenario -> size, one table, coarsest tier first."""
    out = [
        "## Results",
        "",
        "Three tiers per operator: the operator as a whole, each scenario "
        "variant, then each size. Rollup rows sum the timings of their "
        "children and carry a geomean; `sig` counts scenarios whose delta "
        "cleared their own noise band, and only those are evidence.",
        "",
        "| operator | scenario | size | before ms | after ms | Δ % | " "±noise % | speedup | sig |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :--: |",
    ]

    # Most-improved first so real wins surface; ties fall back to name.
    def order(op):
        r = _rollup(op["scenarios"])
        return (-(r.get("geomean_sig") or 0.0), op["operator"])

    for op in sorted(operators, key=order):
        ok = [s for s in op["scenarios"] if s.get("status") == "ok"]
        top = _rollup(op["scenarios"])
        if not top:
            out.append(f"| **{op['operator']}** | — | — | — | — | — | — | — | — |")
            continue
        flag = ""
        if op.get("checksum_match") is False:
            flag = " ⚠"
        if op.get("before_cached"):
            flag += " ᶜ"
        gsig = f"**{top['geomean_sig']:.2f}x**" if top["geomean_sig"] else f"{top['geomean']:.2f}x"
        out.append(
            f"| **{op['operator']}**{flag} | **(all)** | **(all)** | "
            f"{_fmt(top['before_ms'])} | {_fmt(top['after_ms'])} | "
            f"{_fmt(top['delta_pct'], '{:+.1f}')} | | {gsig} | "
            f"**{top['n_sig']}/{top['n']}** |"
        )

        variants = sorted({s.get("variant") or "—" for s in ok})
        for v in variants:
            grp = [s for s in ok if (s.get("variant") or "—") == v]
            r = _rollup(grp)
            # Collapse the middle tier when it would just restate the operator
            # row: one variant covering every scenario adds a row and no
            # information.
            if len(variants) > 1:
                out.append(
                    f"| | **{v}** | (all) | {_fmt(r['before_ms'])} | "
                    f"{_fmt(r['after_ms'])} | "
                    f"{_fmt(r['delta_pct'], '{:+.1f}')} | | "
                    f"{r['geomean']:.2f}x | {r['n_sig']}/{r['n']} |"
                )
            for s in sorted(grp, key=lambda s: _size_key(s.get("size_id") or "")):
                mark = "" if s["significant"] else " _(noise)_"
                out.append(
                    f"| | | {s.get('size_id') or '—'} | "
                    f"{_fmt(s['before_ms'])} | {_fmt(s['after_ms'])} | "
                    f"{_fmt(s['delta_pct'], '{:+.1f}')}{mark} | "
                    f"{_fmt(s['noise_pct'], '{:.1f}')} | "
                    f"{_fmt(s['speedup'], '{:.2f}x')} | "
                    f"{'YES' if s['significant'] else '·'} |"
                )
        # Scenarios the run could not pair up are reported, never dropped.
        for s in op["scenarios"]:
            if s.get("status") != "ok":
                out.append(
                    f"| | `{s['scenario_id']}` | — | — | — | — | — | — | " f"{s['status']} |"
                )
    out.append("")
    out.append("⚠ outputs differ between builds · ᶜ baseline came from cache")
    out.append("")
    return out


def _git_describe(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%h %s"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _source_dir(build_dir: Path) -> Path:
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                return Path(line.split("=", 1)[1])
    return build_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-build", default="build-before/build")
    ap.add_argument("--after-build", default="build")
    ap.add_argument(
        "--operators",
        default="",
        help="comma-separated operator names (default: every perf binary "
        "present in both builds)",
    )
    ap.add_argument(
        "--skip-operators",
        default="",
        help="comma-separated operators to exclude. Useful for dropping the "
        "expensive ones from a quick pass: mimo_coherent_compensation runs "
        "three scenarios per size against a 537M-cell xl cube and alone "
        "accounted for ~27 of 53 minutes in a full sweep",
    )
    ap.add_argument("--sizes", default="sm,md,xl")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="pooled repeats *inside* one launch. Defaults to 1 on purpose: "
        "the dominant variance here is between processes, so extra in-process "
        "repeats cost wall time without tightening the comparison. Spend the "
        "budget on --rounds instead (default 1)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="scored before/after passes; one extra warm-up pass is always "
        "measured and discarded. Per-scenario stat is the median across "
        "scored passes. This is the knob that tightens the noise band — "
        "raise it in preference to --frames or --repeats (default 5)",
    )
    ap.add_argument(
        "--before-args",
        default="",
        help="extra perf-binary flags for the BEFORE side only (e.g. "
        '"--allocator stream_ordered"). Lets one build be A/B\'d against '
        "itself under two configurations by pointing both --before-build and "
        "--after-build at the same tree",
    )
    ap.add_argument(
        "--after-args",
        default="",
        help="extra perf-binary flags for the AFTER side only",
    )
    ap.add_argument("--sweep-variants", action="store_true")
    ap.add_argument("--sweep-overrides", action="store_true")
    ap.add_argument(
        "--checksum",
        action="store_true",
        help="also assert the two builds emit identical output checksums",
    )
    ap.add_argument(
        "--progress",
        choices=("auto", "bar", "plain", "none"),
        default="auto",
        help="'bar' redraws one line in place (needs a TTY), 'plain' prints "
        "one line per launch (safe to redirect to a log), 'auto' picks by "
        "isatty (default)",
    )
    ap.add_argument(
        "--pin-cpus",
        default="2-5",
        help="taskset CPU list for every perf-binary launch. The CPU governor "
        "is the dominant noise source here (idle cores sit at 972MHz and take "
        "a launch or two to reach 2601MHz); concentrating load on a fixed set "
        "measurably tightens the spread. Empty string disables (default 2-5)",
    )
    ap.add_argument(
        "--warmup-rounds",
        type=int,
        default=2,
        help="rounds measured and discarded before scoring begins. The "
        "governor ramp contaminates roughly the first two launches after an "
        "idle gap, so 1 was leaving a ~2x outlier in the scored set "
        "(default 2)",
    )
    ap.add_argument(
        "--max-temp",
        type=float,
        default=60.0,
        help="wait before each launch until the GPU is at or below this "
        "temperature (C), so thermal state is comparable across launches. "
        "Measured on Thor: sustained load peaks ~39C, so the default never "
        "fires there; it exists for hardware that does throttle. 0 disables "
        "(default 60)",
    )
    ap.add_argument(
        "--max-temp-wait",
        type=float,
        default=120.0,
        help="give up waiting to cool after this many seconds and proceed "
        "anyway, so an unreadable sensor cannot hang a run (default 120)",
    )
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the cheap pre-pass that estimates each operator's cost. "
        "The pass costs ~1%% of the run and makes the ETA meaningful from the "
        "first launch; without it, a skewed workload is under-estimated until "
        "the expensive operator is reached",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="always re-measure the before side in this session. Slower, but "
        "every number is same-session and interleaved",
    )
    ap.add_argument(
        "--cache-dir",
        default="performance/agent/runs/_cache",
        help="where cached before-side results live",
    )
    ap.add_argument(
        "--canary-ops",
        type=int,
        default=2,
        help="how many of the cheapest operators to re-measure fresh on both "
        "sides to validate a cached baseline (default 2, 0 disables the "
        "check — not recommended)",
    )
    ap.add_argument("--out-dir", default="performance/agent/runs/_compare")
    a = ap.parse_args(argv)

    global _PIN_PREFIX
    _PIN_PREFIX = _validate_pin(a.pin_cpus)

    _warn_concurrent(Path(a.out_dir))

    before_build = Path(a.before_build).resolve()
    after_build = Path(a.after_build).resolve()
    before_bins = _perf_binaries(before_build)
    after_bins = _perf_binaries(after_build)
    if not before_bins:
        print(f"error: no perf binaries under {before_build}", file=sys.stderr)
        return 1
    if not after_bins:
        print(f"error: no perf binaries under {after_build}", file=sys.stderr)
        return 1

    if a.operators:
        ops = [o.strip() for o in a.operators.split(",") if o.strip()]
    else:
        ops = sorted(set(before_bins) & set(after_bins))
    missing = [o for o in ops if o not in before_bins or o not in after_bins]
    if missing:
        print(
            f"error: not built in both trees: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    if a.skip_operators:
        skip = {o.strip() for o in a.skip_operators.split(",") if o.strip()}
        # A typo in a skip list would otherwise pass silently and run the very
        # operator the caller meant to drop.
        unknown = sorted(skip - set(ops))
        if unknown:
            print(
                f"warning: --skip-operators names nothing in this run: " f"{', '.join(unknown)}",
                file=sys.stderr,
            )
        dropped = [o for o in ops if o in skip]
        ops = [o for o in ops if o not in skip]
        if not ops:
            print("error: --skip-operators excluded every operator", file=sys.stderr)
            return 1
        if dropped:
            print(f"skipping {len(dropped)} operator(s): " f"{', '.join(dropped)}")

    args = ["--warmup", str(a.warmup), "--frames", str(a.frames)]
    if a.repeats > 1:
        args += ["--repeats", str(a.repeats)]
    if a.sizes:
        args += ["--sizes", a.sizes]
    before_extra = shlex.split(a.before_args)
    after_extra = shlex.split(a.after_args)
    sweep_flags = []
    if a.sweep_variants:
        sweep_flags.append("--sweep-variants")
    if a.sweep_overrides:
        sweep_flags.append("--sweep-overrides")
    args += sweep_flags

    style = a.progress
    if style == "auto":
        style = "bar" if sys.stdout.isatty() else "plain"

    both_sides = (a.rounds + a.warmup_rounds) * 2
    after_only = a.rounds + a.warmup_rounds

    # --- cache lookup -----------------------------------------------------
    cache_dir = Path(a.cache_dir)
    machine = _machine_fingerprint()
    keys = {
        op: _cache_key(
            before_bins[op],
            args + before_extra + _PIN_PREFIX,
            a.rounds,
            machine,
        )
        for op in ops
    }
    cached: dict[str, dict] = {}
    if not a.no_cache:
        for op in ops:
            entry = _cache_load(cache_dir, op, keys[op])
            if entry is not None:
                cached[op] = entry

    # Canary operators are re-measured fresh on both sides even though they
    # have a cached baseline, so the cache can be checked against reality.
    # Pick the cheapest by recorded launch cost — validation should be nearly
    # free relative to the run it is protecting.
    canaries: list[str] = []
    if cached and a.canary_ops > 0:
        canaries = sorted(cached, key=lambda o: cached[o].get("launch_seconds", 0.0))[
            : a.canary_ops
        ]

    # Canaries run first so the verdict is known before the cached operators.
    ordered = canaries + [o for o in ops if o not in canaries]

    def _launches(op: str) -> int:
        return after_only if (op in cached and op not in canaries) else both_sides

    # --- per-operator cost estimates --------------------------------------
    # A cached entry already knows what its operator costs; probe only what is
    # left. With a fully warm cache the probe pass is skipped entirely.
    est: dict[str, float] = {
        op: e["launch_seconds"] for op, e in cached.items() if e.get("launch_seconds")
    }
    probe_secs = 0.0
    unknown = [o for o in ordered if o not in est]
    if unknown and not a.no_probe:
        print(
            f"probing {len(unknown)} operator(s) to estimate runtime ...",
            flush=True,
        )
        probed, probe_secs = _probe_operators(
            unknown,
            before_bins,
            sweep_flags + before_extra,
            a.sizes,
            a.warmup,
            a.frames,
            a.repeats,
        )
        est.update(probed)

    plan = [(op, _launches(op), est.get(op)) for op in ordered]
    total_launches = sum(n for _, n, _e in plan)
    predicted = sum(n * est[o] for o, n, _e in plan if o in est)
    saved = len(ops) * both_sides - total_launches
    cache_note = f"      cache: {len(cached)}/{len(ops)} baselines reused"
    if saved:
        cache_note += f" ({saved} launches saved)"
    if canaries:
        cache_note += f", canary={', '.join(canaries)}"
    elif not cached:
        cache_note += " (cold cache — measuring both sides)"
    if predicted > 0:
        covered = sum(1 for o, _n, _e in plan if o in est)
        est_note = (
            f"      estimated runtime ~{_fmt_dur(predicted)} "
            f"({covered}/{len(plan)} operators estimated"
            + (f", probe cost {_fmt_dur(probe_secs)}" if probe_secs else "")
            + ")"
        )
    else:
        est_note = "      ETA appears once the first launch completes."
    print(
        f"plan: {len(ops)} operators, {total_launches} launches "
        f"({a.rounds} scored rounds + {a.warmup_rounds} warm-up)\n"
        f"      sizes={a.sizes} warmup={a.warmup} frames={a.frames} "
        f"repeats={a.repeats}\n"
        f"{cache_note}\n"
        f"{est_note}",
        flush=True,
    )

    prog = _Progress(plan, style)
    results: list[dict] = []
    cache_invalidated = False
    for i, op in enumerate(ordered, start=1):
        prog.start_operator(i)
        use_cached = op in cached and op not in canaries and not cache_invalidated
        t_op = time.monotonic()
        try:
            res = compare_operator(
                op,
                before_bins[op],
                after_bins[op],
                args + before_extra,
                args + after_extra,
                a.rounds,
                a.checksum,
                prog,
                cached[op] if use_cached else None,
                (a.max_temp, a.max_temp_wait),
                a.warmup_rounds,
            )
        except RuntimeError as exc:
            prog.drop_operator(prog.per_op - len(prog.op_times))
            prog.note(f"[compare] {op}: SKIPPED — {exc}")
            continue
        results.append(res)

        if not res["before_cached"]:
            # Freshly measured baseline: persist it, and if this was a canary,
            # check the cached copy still agrees with reality.
            if op in canaries and not cache_invalidated:
                verdict = _canary_verdict(res, cached[op])
                prog.note(f"[canary] {op}: {verdict['message']}")
                if not verdict["ok"]:
                    cache_invalidated = True
                    cached = {}
                    plan = [
                        (o, both_sides if j >= i else n, e)
                        for j, (o, n, e) in enumerate(plan, start=1)
                    ]
                    prog.revise_plan(plan)
                    prog.note(
                        "[canary] cache dropped for this run — remaining "
                        "operators will re-measure both sides."
                    )
            _cache_store(
                cache_dir,
                op,
                keys[op],
                {
                    "cache_version": _CACHE_VERSION,
                    "operator": op,
                    "key": keys[op],
                    "machine": machine,
                    "args": args,
                    "rounds": a.rounds,
                    "before_build": str(before_build),
                    "before_commit": _git_describe(_source_dir(before_build)),
                    "launch_seconds": (time.monotonic() - t_op) / max(1, prog.per_op),
                    "medians": res["_before_med"],
                    "rows": res["_before_rows"],
                    "checksums": res["_before_sums"],
                },
            )
    prog.finish()

    # Private measurement detail — useful for caching, noise for the report.
    for res in results:
        for k in ("_before_med", "_before_rows", "_before_sums"):
            res.pop(k, None)

    gpu = "unknown"
    if shutil.which("nvidia-smi"):
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu = p.stdout.strip().splitlines()[0] if p.stdout.strip() else gpu

    payload = {
        "before_build": str(before_build),
        "after_build": str(after_build),
        "before_commit": _git_describe(_source_dir(before_build)),
        "after_commit": _git_describe(_source_dir(after_build)),
        "args": args,
        "warmup_rounds": a.warmup_rounds,
        "pin_cpus": a.pin_cpus or None,
        "before_args": before_extra,
        "after_args": after_extra,
        "rounds": a.rounds,
        "gpu": gpu,
        "thermal": {
            "max_temp_c": a.max_temp,
            "observed_min_c": min(prog.temps) if prog.temps else None,
            "observed_max_c": max(prog.temps) if prog.temps else None,
            "cooldown_seconds": round(prog.cool_s, 1),
            "samples": len(prog.temps),
        },
        "operators": results,
    }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(payload, indent=2))
    md = render_markdown(payload)
    (out_dir / "compare.md").write_text(md)
    print()
    print(md)
    print(f"wrote {out_dir / 'compare.json'} and {out_dir / 'compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
