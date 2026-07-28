# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose an analysis.json against the perf rubric, emitting ranked findings.

Encodes the diagnosis flow and CFAR-lessons table from
``docs/planning/OPERATOR_DEVELOPMENT.md`` (Performance characterization) as
rules over the folded ``analysis.json``:

* cold-slow vs steady-slow (calibration / plan build / first-touch);
* per-tick dense ``cudaMalloc``/``cudaFree`` (the ping-pong pitfall);
* kernel-vs-graph split — is the tick GPU-bound or host/alloc/launch-bound
  (distance to architectural speed-of-light);
* host D2H readback on the steady path (an O(map) host-scan smell);
* costly ``[live]`` override rebuilds that would thrash if not cached;
* effective GB/s vs an optional device roofline.

Each finding carries the nsys evidence and the recommended fix pattern. Writes
``diagnosis.json`` next to the analysis; ``report.py`` renders it to markdown.

Thresholds are heuristics for *spotting obvious pitfalls*, not pass/fail gates
(the harness is a measurement tool). They live as named constants below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- thresholds (tunable; heuristic, not pass/fail) ------------------------
COLD_RATIO_WARN = 3.0  # cold_ms > N x median => cold-dominated
ALLOC_CALLS_PER_TICK_WARN = 1.0  # >= N alloc calls/tick => per-tick allocation
KERNEL_FRACTION_WARN = 0.5  # kernel < N of steady tick => host/graph-bound
# kernel >= N of steady tick => the kernel IS the tick. Without this the rubric
# could only ever say "do not optimize the kernel" (host_bound_tick) and never
# the converse, so a kernel-dominated operator produced no kernel finding at all.
KERNEL_FRACTION_DOMINANT = 0.7
OVERRIDE_COSTLY_MS = 5.0  # override delta > N ms => costly reconfig
OVERRIDE_HIGH_MS = 20.0  # override delta > N ms => high-severity reconfig

_SEV_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _f(
    finding_id, severity, title, detail, evidence, recommendation, rubric_ref, size=None
):
    return {
        "id": finding_id,
        "severity": severity,
        "size": size,
        "title": title,
        "detail": detail,
        "evidence": evidence,
        "recommendation": recommendation,
        "rubric_ref": rubric_ref,
    }


def _ns_ms(ns):
    return None if ns is None else ns / 1e6


def diagnose_size(s: dict) -> list:
    """Per-size findings."""
    out = []
    size = s["size"]
    lat = s["latency_ms"]
    der = s["derived"]
    cold = lat.get("cold")
    median = lat.get("median")
    ticks = der.get("ticks") or 0

    # --- cold vs steady ----------------------------------------------------
    if cold and median and median > 0 and cold > COLD_RATIO_WARN * median:
        out.append(
            _f(
                "cold_dominated",
                "info",
                size=size,
                title="Cold frame dominates (calibration / plan build / "
                "first-touch)",
                detail=f"cold {cold:.3f} ms is {cold / median:.1f}x the steady "
                f"median {median:.3f} ms. First-tick cost is the FFT-plan "
                f"build / window build / first-touch, not steady work.",
                evidence={
                    "cold_ms": cold,
                    "median_ms": median,
                    "setup_ms": lat.get("setup"),
                },
                recommendation="Expected when plans/coefficients build lazily on "
                "the first tick. Confirm they are cached (steady "
                "median should stay flat) — only a concern if it "
                "recurs on later ticks.",
                rubric_ref="Diagnose: 'cold -> calibration / first alloc / plan "
                "build / first-touch'",
            )
        )

    # --- kernel fraction of the steady tick (shared signal) ----------------
    # Kernel time is GPU-side (profiled run); median is the clean steady wall.
    # Use parse.py's steady-only per-tick kernel time, not the whole-run total:
    # the latter includes one-time builds, teardown, and any background work
    # overlapping the measured window, and dividing that by tick count reports
    # a per-tick cost that never happened per tick.
    kern_pt_ns = der.get("kernel_ns_per_tick")
    steady_ns = median * 1e6 if median else None
    kern_frac = (kern_pt_ns / steady_ns) if (kern_pt_ns and steady_ns) else None
    # kern_frac mixes measurement domains -- nsys-profiled kernel time over a
    # clean wall-clock median -- and nsys inflates kernel time. Above 1.0 the
    # ratio is not physically meaningful (a tick cannot spend longer on the GPU
    # than it takes), so treat it as kernel-dominated but do not report the
    # number as a literal share.
    domain_mismatch = kern_frac is not None and kern_frac > 1.0
    host_bound = kern_frac is not None and kern_frac < KERNEL_FRACTION_WARN
    kernel_bound = kern_frac is not None and kern_frac >= KERNEL_FRACTION_DOMINANT

    # --- per-tick allocation (ping-pong pitfall) ---------------------------
    # Trigger on the per-tick alloc *frequency* (robust), not the whole-run
    # free-time share (which is inflated by one-time teardown frees). Severity
    # tracks impact: only a real win when the tick is host-bound — if the kernel
    # dominates, pooling barely moves it, so it drops to low.
    alloc_pt = der.get("alloc_calls_per_tick") or 0
    free_pt = der.get("free_calls_per_tick") or 0
    free_pct = der.get("host_api_pct", {}).get("free", 0.0)
    if alloc_pt >= ALLOC_CALLS_PER_TICK_WARN:
        sev = "low" if (kern_frac is not None and not host_bound) else "medium"
        impact = (
            "a minor cost — the tick is kernel-bound, so pooling is a small win"
            if sev == "low"
            else "likely the dominant steady cost"
        )
        out.append(
            _f(
                "per_tick_alloc",
                sev,
                size=size,
                title="Steady path allocates every tick (pool / ping-pong)",
                detail=f"~{alloc_pt:.1f} cudaMalloc + ~{free_pt:.1f} cudaFree "
                f"calls per tick — the output tensor(s) are allocated every "
                f"frame, so alloc/free sit on the hot path ({impact}).",
                evidence={
                    "alloc_calls_per_tick": alloc_pt,
                    "free_calls_per_tick": free_pt,
                    # whole-run share, inflated by teardown — context only.
                    "free_pct_of_host_api_whole_run": round(free_pct, 1),
                    "kernel_fraction": (
                        round(kern_frac, 3) if kern_frac is not None else None
                    ),
                    "top_api": _top_api(s),
                },
                recommendation="Pool the dense output buffers and ping-pong a "
                "workspace keyed on the input geometry, so steady "
                "ticks reuse device memory instead of "
                "malloc/free-ing it (CFAR lesson: 'Pool dense "
                "buffers; ping-pong workspace on geometry key').",
                rubric_ref="CFAR lessons: per-tick dense cudaMalloc -> pool / "
                "ping-pong on geometry key",
            )
        )

    # --- kernel vs graph/host (distance to speed-of-light) -----------------
    if host_bound:
        out.append(
            _f(
                "host_bound_tick",
                "medium",
                size=size,
                title="Tick is host/graph-bound, not kernel-bound",
                detail=f"GPU kernel time is ~{_ns_ms(kern_pt_ns):.4f} ms/tick "
                f"({100 * kern_frac:.0f}% of the {median:.3f} ms steady "
                f"tick). The rest is graph/shell/alloc/launch/sync "
                f"overhead — optimizing the kernel will not move the "
                f"tick.",
                evidence={
                    "kernel_ms_per_tick": _ns_ms(kern_pt_ns),
                    "steady_median_ms": median,
                    "kernel_fraction": round(kern_frac, 3),
                    "kernels": s["gpu"].get("kernels", [])[:3],
                },
                recommendation="Attack the shell first: pool allocations (see "
                "the per-tick allocation finding), cut per-tick "
                "syncs, and reduce launch/graph overhead. Do not "
                "rewrite the stencil/FFT when it is a minority of "
                "the tick.",
                rubric_ref="Diagnose: 'kernel fast, graph slow -> shell / "
                "alloc / compact / scheduler'",
            )
        )

    # --- kernel dominates the tick (the converse of host_bound_tick) -------
    if kernel_bound:
        share = (
            "at or above the whole tick (nsys inflates kernel time, so read "
            "this as kernel-dominated rather than a literal share)"
            if domain_mismatch
            else f"{100 * kern_frac:.0f}% of the {median:.3f} ms steady tick"
        )
        out.append(
            _f(
                "kernel_bound_tick",
                "high" if domain_mismatch else "medium",
                size=size,
                title="Tick is kernel-bound — the kernel is the target",
                detail=f"Steady GPU kernel time is ~{_ns_ms(kern_pt_ns):.4f} "
                f"ms/tick, {share}. Shell-level fixes (pooling, gating "
                f"allocations) cannot move this tick; the win has to come "
                f"from the kernel itself.",
                evidence={
                    "kernel_ms_per_tick": _ns_ms(kern_pt_ns),
                    "steady_median_ms": median,
                    "kernel_fraction": (
                        round(kern_frac, 3) if kern_frac is not None else None
                    ),
                    "domain_mismatch": domain_mismatch,
                    "top_kernels": [
                        {"name": k["name"], "ns": k["ns"]}
                        for k in (s["gpu"].get("kernels") or [])[:3]
                    ],
                },
                recommendation="Attack the kernel: hoist per-cell work fixed "
                "by geometry rather than data (counts, bounds, "
                "branches) into closed form or loop structure; "
                "avoid materialising a buffer a streaming "
                "formulation never needs; check "
                "`cuobjdump -res-usage` for a per-thread stack "
                "frame before assuming registers.",
                rubric_ref="CFAR lessons: kernel-bound tick -> hoist "
                "geometry-determined work; stream instead of materialise",
            )
        )

    # --- host D2H readback on steady path ----------------------------------
    d2h = s["gpu"].get("memcpy", {}).get("d2h_ns") or 0
    if d2h > 0:
        out.append(
            _f(
                "host_readback",
                "medium",
                size=size,
                title="Device-to-host copy on the steady path",
                detail=f"{_ns_ms(d2h):.4f} ms of D2H memcpy — a host readback / "
                f"O(map) scan on the hot path is a classic CFAR-style "
                f"pitfall.",
                evidence={"d2h_ns": d2h, "memcpy": s["gpu"].get("memcpy")},
                recommendation="Move the work onto the GPU (device compaction / "
                "closed-form bounds) and keep results device-side; "
                "avoid full-map D2H every tick.",
                rubric_ref="CFAR lessons: host reachability scan / full-mask D2H "
                "-> gate unused work; GPU compact",
            )
        )

    return out


def _top_api(s: dict, n: int = 5) -> list:
    by_name = s["host_api"].get("by_name", {})
    rows = sorted(by_name.items(), key=lambda kv: kv[1].get("ns", 0), reverse=True)
    return [
        {
            "name": k,
            "ms": round((v.get("ns", 0)) / 1e6, 3),
            "calls": v.get("calls", 0),
            "category": v.get("category"),
        }
        for k, v in rows[:n]
    ]


def diagnose_overrides(analysis: dict) -> list:
    """Findings over the [live] override sweep."""
    out = []
    sweep = analysis.get("override_sweep", [])
    if not sweep:
        return out

    # Representative steady median (smallest size) for the thrash comparison.
    costly = [
        o for o in sweep if (o.get("override_delta_ms") or 0) >= OVERRIDE_COSTLY_MS
    ]
    cheap = [o for o in sweep if (o.get("override_delta_ms") or 0) < OVERRIDE_COSTLY_MS]
    for o in sorted(costly, key=lambda x: -(x["override_delta_ms"] or 0)):
        sev = "high" if (o["override_delta_ms"] or 0) >= OVERRIDE_HIGH_MS else "medium"
        out.append(
            _f(
                f"override_costly_{o['param']}_{o['size']}",
                sev,
                size=o["size"],
                title=f"Live '{o['param']}' override triggers a costly rebuild",
                detail=f"Changing {o['param']} at runtime costs "
                f"{o['override_delta_ms']:.2f} ms (peak "
                f"{o['override_ms']:.3f} vs steady {o['median_ms']:.3f} "
                f"ms) — a plan/coefficient rebuild.",
                evidence={
                    "param": o["param"],
                    "size": o["size"],
                    "override_delta_ms": o["override_delta_ms"],
                    "override_ms": o["override_ms"],
                    "median_ms": o["median_ms"],
                },
                recommendation="A one-time reconfiguration is fine if the rebuild "
                "is cached (warn-once-and-cache); ensure prior "
                "configs are retained so re-selecting one is free. "
                "Callers must NOT sweep this param every tick — "
                f"that thrashes at ~{o['override_delta_ms']:.0f} "
                "ms/tick.",
                rubric_ref="[live] convention: costly overrides warn once and "
                "cache; structural params stay static",
            )
        )

    if cheap:
        params = sorted({o["param"] for o in cheap})
        # Worst cheap reconfig, clamped: the override cost is a per-window peak,
        # so sub-ms deltas (including small negatives) are measurement noise,
        # not real cost — report the bound, not a specific noisy value.
        worst = max((o.get("override_delta_ms") or 0) for o in cheap)
        worst = max(worst, 0.0)
        out.append(
            _f(
                "override_cheap",
                "info",
                size=None,
                title="Cheap sticky-flip live params (negligible reconfig cost)",
                detail=f"{', '.join(params)} all reconfigure in under "
                f"{OVERRIDE_COSTLY_MS:.0f} ms (worst ~{worst:.3f} ms) — safe "
                f"to change at runtime frequently. Sub-millisecond deltas sit "
                f"near the measurement floor (the override cost is a "
                f"per-window peak), so treat them as ~0.",
                evidence={
                    "cheap_params": params,
                    "worst_delta_ms": round(worst, 3),
                },
                recommendation="No action — these confirm the cheap-flip half of "
                "the [live] surface is genuinely cheap (contrast "
                "with the rebuild params above).",
                rubric_ref="Required coverage: cover cheap sticky flips AND costly "
                "rebuild paths",
            )
        )
    return out


def summarize(findings: list) -> dict:
    counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    return counts


def diagnose(analysis_path: Path, peak_gbps: float | None = None) -> Path:
    analysis = json.loads(analysis_path.read_text())
    findings = []
    for s in analysis.get("sizes", []):
        findings.extend(diagnose_size(s))
    findings.extend(diagnose_overrides(analysis))

    # Optional roofline note (distance to device speed-of-light).
    if peak_gbps:
        for s in analysis.get("sizes", []):
            gbps = s["throughput"].get("estimated_effective_gbps")
            if gbps:
                findings.append(
                    _f(
                        "roofline",
                        "info",
                        size=s["size"],
                        title="Effective bandwidth vs device roofline",
                        detail=f"~{gbps:.2f} GB/s effective = "
                        f"{100 * gbps / peak_gbps:.1f}% of the "
                        f"{peak_gbps:.0f} GB/s device peak.",
                        evidence={
                            "effective_gbps": gbps,
                            "peak_gbps": peak_gbps,
                            "pct_of_peak": round(100 * gbps / peak_gbps, 1),
                        },
                        recommendation="Low % with a host-bound tick means the "
                        "shell, not memory bandwidth, is the limit "
                        "— see the host-bound finding.",
                        rubric_ref="Goals: judge distance to architectural "
                        "speed-of-light",
                    )
                )

    findings.sort(key=lambda f: (_SEV_RANK[f["severity"]], f.get("size") or ""))

    diagnosis = {
        "schema": "perf-diagnosis/v1",
        "operator": analysis.get("operator"),
        "generated_from": str(analysis_path),
        "platform": analysis.get("platform", {}),
        "summary": summarize(findings),
        "findings": findings,
    }
    out = analysis_path.parent / "diagnosis.json"
    out.write_text(json.dumps(diagnosis, indent=2))
    print(out)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose analysis.json against the perf rubric"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--analysis", type=Path, help="analysis.json path")
    g.add_argument("--run-dir", type=Path, help="directory holding analysis.json")
    p.add_argument(
        "--peak-gbps",
        type=float,
        default=None,
        help="device peak memory bandwidth (GB/s) for a roofline " "note; omit to skip",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    analysis = args.analysis or (args.run_dir / "analysis.json")
    if not analysis.exists():
        raise SystemExit(f"[perf-diagnose] analysis not found: {analysis}")
    diagnose(analysis, peak_gbps=args.peak_gbps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
