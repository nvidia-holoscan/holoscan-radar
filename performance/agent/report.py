# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render analysis.json + diagnosis.json into a per-operator markdown report.

The output is meant to be pasted into an operator's ``OPEN_ITEMS.md`` (or read
directly): ranked findings with nsys evidence and a recommended fix, followed
by the supporting per-size latency / attribution tables and the ``[live]``
override sweep. This is the human-facing end of the harness; it stops at
recommendations (no autonomous edit loop this phase).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SEV_BADGE = {
    "high": "🔴 HIGH",
    "medium": "🟠 MEDIUM",
    "low": "🟡 LOW",
    "info": "🔵 INFO",
}


def _ms(v, prec=3):
    return "-" if v is None else f"{v:.{prec}f}"


def _fmt_int(v):
    return "-" if v is None else f"{int(v)}"


def _plat_line(platform: dict) -> str:
    return (
        f"{platform.get('gpu_name', '?')} | driver "
        f"{platform.get('gpu_driver', '?')} | CUDA "
        f"{platform.get('cuda_runtime', '?')} | Holoscan "
        f"{platform.get('holoscan_version', '?')}"
    )


def render(analysis: dict, diagnosis: dict) -> str:
    op = analysis.get("operator", "?")
    prov = analysis.get("provenance", {})
    plat = analysis.get("platform", {})
    nsys = analysis.get("nsys", {})
    lines = []
    a = lines.append

    a(f"# Performance report — {op}")
    a("")
    a(f"- **Platform:** {_plat_line(plat)}")
    a(
        f"- **Commit:** `{prov.get('git_commit', '?')}`"
        f"{' (dirty)' if prov.get('git_dirty') else ''}"
        f" — {prov.get('timestamp_utc', '?')}"
    )
    a(f"- **nsys:** {nsys.get('version', '?')} " f"(trace `{nsys.get('trace', '?')}`)")
    a(f"- **CPU:** {plat.get('cpu_model', '?')} | {plat.get('os', '?')}")
    a("")
    a(
        "> Latencies are valid only on the machine/GPU captured on. Kernel time "
        "is GPU-side; CUDA-API time is host-side wall — they overlap, they do "
        "not sum. Attribution counts include the binary's discarded "
        "process-warmup session."
    )
    a("")

    # --- findings ----------------------------------------------------------
    summary = diagnosis.get("summary", {})
    a("## Findings")
    a("")
    a(
        f"{summary.get('high', 0)} high · {summary.get('medium', 0)} medium · "
        f"{summary.get('low', 0)} low · {summary.get('info', 0)} info"
    )
    a("")
    findings = diagnosis.get("findings", [])
    if not findings:
        a("_No findings._")
        a("")
    for i, f in enumerate(findings, 1):
        badge = _SEV_BADGE.get(f["severity"], f["severity"])
        sz = f" · size `{f['size']}`" if f.get("size") else ""
        a(f"### {i}. {badge}{sz} — {f['title']}")
        a("")
        a(f["detail"])
        a("")
        a(f"- **Recommendation:** {f['recommendation']}")
        a(f"- **Rubric:** {f['rubric_ref']}")
        ev = _prune_evidence(f.get("evidence"))
        if ev:
            blob = json.dumps(ev, separators=(",", ":"))
            if len(blob) > 600:
                blob = blob[:597] + "..."
            a(f"- **Evidence:** `{blob}`")
        a("")

    # --- per-size latency + attribution ------------------------------------
    a("## Per-size latency & attribution")
    a("")
    a(
        "| size | shape | setup | cold | median | p90 | kernel/tick | "
        "alloc/tick | free/tick | GB/s |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in analysis.get("sizes", []):
        lat = s["latency_ms"]
        der = s["derived"]
        kern_total = s["gpu"].get("kernel_ns_total") or 0
        ticks = der.get("ticks") or 0
        kern_pt = (kern_total / ticks / 1e6) if ticks else None
        shape = "x".join(str(x) for x in (s.get("input_shape") or []))
        a(
            f"| {s['size']} | {shape} | {_ms(lat.get('setup'), 2)} | "
            f"{_ms(lat.get('cold'))} | {_ms(lat.get('median'))} | "
            f"{_ms(lat.get('p90'))} | {_ms(kern_pt, 4)} | "
            f"{_ms(der.get('alloc_calls_per_tick'), 1)} | "
            f"{_ms(der.get('free_calls_per_tick'), 1)} | "
            f"{_ms(s['throughput'].get('estimated_effective_gbps'), 2)} |"
        )
    a("")

    # --- steady breakdown per size -----------------------------------------
    for s in analysis.get("sizes", []):
        a(f"### Steady breakdown — size `{s['size']}`")
        a("")
        a("compute() NVTX phases (per-tick wall vs GPU-projected):")
        a("")
        a("| phase | wall avg (ms) | GPU proj total (ms) | instances |")
        a("|---|---|---|---|")
        for p in s.get("phases", []):
            a(
                f"| {p['phase']} | {_ms(p['wall_avg_ns'] / 1e6, 4)} | "
                f"{_ms(p['gpu_proj_ns_total'] / 1e6, 4)} | {p['instances']} |"
            )
        a("")
        top = _top_api(s)
        if top:
            a("Top CUDA API by time (host-side, whole run):")
            a("")
            a("| api | ms | calls | category |")
            a("|---|---|---|---|")
            for t in top:
                a(
                    f"| {t['name']} | {_ms(t['ms'])} | {t['calls']} | "
                    f"{t['category']} |"
                )
            a("")
        mem = s["gpu"].get("memcpy", {})
        a(
            f"memcpy: H2D {_ms((mem.get('h2d_ns') or 0) / 1e6, 4)} ms · "
            f"D2H {_ms((mem.get('d2h_ns') or 0) / 1e6, 4)} ms"
        )
        a("")

    # --- override sweep ----------------------------------------------------
    sweep = analysis.get("override_sweep", [])
    if sweep:
        a("## [live] override sweep (per-param reconfiguration cost)")
        a("")
        a(
            "| size | param | steady median (ms) | override peak (ms) | "
            "**delta (ms)** |"
        )
        a("|---|---|---|---|---|")
        for o in sweep:
            a(
                f"| {o['size']} | {o['param']} | {_ms(o.get('median_ms'))} | "
                f"{_ms(o.get('override_ms'))} | "
                f"**{_ms(o.get('override_delta_ms'))}** |"
            )
        a("")

    return "\n".join(lines) + "\n"


def _prune_evidence(ev):
    """Drop verbose keys (mangled kernel raw_name) so inline evidence stays
    readable; the full detail lives in diagnosis.json."""
    if isinstance(ev, dict):
        return {k: _prune_evidence(v) for k, v in ev.items() if k != "raw_name"}
    if isinstance(ev, list):
        return [_prune_evidence(v) for v in ev]
    return ev


def _top_api(s: dict, n: int = 5) -> list:
    by_name = s["host_api"].get("by_name", {})
    rows = sorted(by_name.items(), key=lambda kv: kv[1].get("ns", 0), reverse=True)
    return [
        {
            "name": k,
            "ms": (v.get("ns", 0)) / 1e6,
            "calls": v.get("calls", 0),
            "category": v.get("category"),
        }
        for k, v in rows[:n]
    ]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a markdown perf report from analysis + diagnosis"
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="directory holding analysis.json + diagnosis.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output markdown path (default <run-dir>/report.md)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    analysis_path = args.run_dir / "analysis.json"
    diagnosis_path = args.run_dir / "diagnosis.json"
    if not analysis_path.exists():
        raise SystemExit(f"[perf-report] analysis not found: {analysis_path}")
    if not diagnosis_path.exists():
        raise SystemExit(f"[perf-report] diagnosis not found: {diagnosis_path}")
    analysis = json.loads(analysis_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    md = render(analysis, diagnosis)
    out = args.out or (args.run_dir / "report.md")
    out.write_text(md)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
