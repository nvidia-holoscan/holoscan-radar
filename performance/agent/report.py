# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render analysis.json + diagnosis.json into a per-operator perf report.

Emits two artifacts, both next to the analysis:

* ``report.md`` — ranked findings with nsys evidence + recommended fix, then
  the per-size latency/attribution tables and the ``[live]`` override sweep.
  Plain markdown, meant to be pasted into an operator's ``OPEN_ITEMS.md``.
* ``report.html`` — a self-contained page: the same findings + tables plus the
  ``visualize.py`` chart dashboard embedded as a base64 PNG (no external files).

The charts need ``matplotlib``; when it is not installed ``report.html`` is
still written, minus the dashboard image (the harness core stays
dependency-free). This is the human-facing end of the harness; it stops at
recommendations (no autonomous edit loop this phase).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

# Charts are optional: import matplotlib (Agg, headless) + visualize.py lazily
# so a missing matplotlib degrades report.html to text-only instead of failing.
try:
    import matplotlib

    matplotlib.use("Agg")
    import visualize as viz  # same directory; on sys.path for CLI use

    _HAVE_VIZ = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_VIZ = False

_SEV_BADGE = {
    "high": "🔴 HIGH",
    "medium": "🟠 MEDIUM",
    "low": "🟡 LOW",
    "info": "🔵 INFO",
}

# (hex, label) per severity for the HTML badges.
_SEV_HTML = {
    "high": ("#d03b3b", "HIGH"),
    "medium": ("#ec835a", "MEDIUM"),
    "low": ("#fab219", "LOW"),
    "info": ("#2a78d6", "INFO"),
}


def _ms(v, prec=3):
    return "-" if v is None else f"{v:.{prec}f}"


def _span(lat: dict, prec=3) -> str:
    """ "min–max" observed-spread string for a latency block."""
    lo, hi = lat.get("min"), lat.get("max")
    if lo is None or hi is None:
        return "-"
    return f"{lo:.{prec}f}–{hi:.{prec}f}"


def _has_variants(analysis: dict) -> bool:
    """True when any size carries more than one primary-axis variant."""
    return any(len(s.get("variants", [])) > 1 for s in analysis.get("sizes", []))


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
        "| size | shape | setup | cold | median | min–max | N | p90 | "
        "kernel/tick | alloc/tick | free/tick | GB/s |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
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
            f"{_span(lat)} | {lat.get('n') or '-'} | {_ms(lat.get('p90'))} | "
            f"{_ms(kern_pt, 4)} | "
            f"{_ms(der.get('alloc_calls_per_tick'), 1)} | "
            f"{_ms(der.get('free_calls_per_tick'), 1)} | "
            f"{_ms(s['throughput'].get('estimated_effective_gbps'), 2)} |"
        )
    a("")
    a(
        "> Latency = median over N pooled samples (measured_frames × repeats); "
        "min–max is the observed spread across those samples. Attribution "
        "columns are size-level (profiled at the primary variant)."
    )
    a("")

    # --- per-variant steady latency (only when a primary axis was swept) -----
    if _has_variants(analysis):
        a("## Per-variant steady latency (primary axis × size)")
        a("")
        a("| size | variant | median (ms) | min–max (ms) | p90 (ms) | N |")
        a("|---|---|---|---|---|---|")
        for s in analysis.get("sizes", []):
            for v in s.get("variants", []):
                vl = v["latency_ms"]
                a(
                    f"| {s['size']} | {v['variant']} | {_ms(vl.get('median'))} | "
                    f"{_span(vl)} | {_ms(vl.get('p90'))} | {vl.get('n') or '-'} |"
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
                a(f"| {t['name']} | {_ms(t['ms'])} | {t['calls']} | " f"{t['category']} |")
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
        a("| size | param | steady median (ms) | override peak (ms) | " "**delta (ms)** |")
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


_HTML_STYLE = """
:root { color-scheme: light; }
body { background:#f9f9f7; color:#0b0b0b; margin:0; padding:2rem;
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1040px; margin:0 auto; }
.surface { background:#fcfcfb; border:1px solid rgba(11,11,11,0.10);
  border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1.25rem; }
h1 { font-size:1.4rem; margin:0 0 .5rem; }
h2 { font-size:1.1rem; margin:1.5rem 0 .75rem; }
.meta, .muted { color:#52514e; font-size:.9rem; }
.note { color:#52514e; border-left:3px solid #e1e0d9; padding:.5rem .75rem;
  margin:.75rem 0; font-size:.9rem; }
.badge { display:inline-block; color:#fff; font-weight:600; font-size:.72rem;
  padding:.1rem .5rem; border-radius:6px; letter-spacing:.02em; }
.finding { border-top:1px solid #e1e0d9; padding:.9rem 0; }
.finding h3 { font-size:1rem; margin:.25rem 0 .4rem; }
.finding .rubric { color:#898781; font-size:.82rem; }
.evidence { font-family: ui-monospace,Menlo,monospace; font-size:.76rem;
  color:#52514e; word-break:break-all; }
table { border-collapse:collapse; width:100%; font-size:.86rem;
  font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:.35rem .6rem; border-bottom:1px solid #e1e0d9; }
th { color:#52514e; font-weight:600; }
img.dash { width:100%; height:auto; border:1px solid #e1e0d9; border-radius:8px; }
"""


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=viz.SURFACE, bbox_inches="tight")
    viz.plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _dashboard_png(run_dir) -> str | None:
    """Render the visualize.py dashboard to a base64 PNG, or None if charts are
    unavailable (matplotlib missing or a rendering error)."""
    if not _HAVE_VIZ:
        return None
    try:
        return _fig_to_base64(viz.dashboard(str(run_dir)))
    except Exception:  # pragma: no cover - defensive
        return None


def _html_latency_table(analysis: dict) -> str:
    head = (
        "<tr><th>size</th><th>shape</th><th>setup</th><th>cold</th>"
        "<th>median</th><th>min–max</th><th>N</th><th>p90</th>"
        "<th>kernel/tick</th><th>alloc/tick</th>"
        "<th>free/tick</th><th>GB/s</th></tr>"
    )
    rows = [head]
    for s in analysis.get("sizes", []):
        lat = s["latency_ms"]
        der = s["derived"]
        ticks = der.get("ticks") or 0
        kern_total = s["gpu"].get("kernel_ns_total") or 0
        kern_pt = (kern_total / ticks / 1e6) if ticks else None
        shape = "x".join(str(x) for x in (s.get("input_shape") or []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(s['size'])}</td><td>{shape}</td>"
            f"<td>{_ms(lat.get('setup'), 2)}</td><td>{_ms(lat.get('cold'))}</td>"
            f"<td>{_ms(lat.get('median'))}</td><td>{_span(lat)}</td>"
            f"<td>{lat.get('n') or '-'}</td><td>{_ms(lat.get('p90'))}</td>"
            f"<td>{_ms(kern_pt, 4)}</td>"
            f"<td>{_ms(der.get('alloc_calls_per_tick'), 1)}</td>"
            f"<td>{_ms(der.get('free_calls_per_tick'), 1)}</td>"
            f"<td>{_ms(s['throughput'].get('estimated_effective_gbps'), 2)}</td>"
            "</tr>"
        )
    table = "<table>" + "".join(rows) + "</table>"
    if not _has_variants(analysis):
        return table
    vrows = [
        "<tr><th>size</th><th>variant</th><th>median</th><th>min–max</th>"
        "<th>p90</th><th>N</th></tr>"
    ]
    for s in analysis.get("sizes", []):
        for v in s.get("variants", []):
            vl = v["latency_ms"]
            vrows.append(
                "<tr>"
                f"<td>{html.escape(s['size'])}</td>"
                f"<td>{html.escape(str(v['variant']))}</td>"
                f"<td>{_ms(vl.get('median'))}</td><td>{_span(vl)}</td>"
                f"<td>{_ms(vl.get('p90'))}</td><td>{vl.get('n') or '-'}</td>"
                "</tr>"
            )
    return (
        table
        + "<h3>Per-variant steady latency (primary axis × size)</h3>"
        + "<table>"
        + "".join(vrows)
        + "</table>"
    )


def _html_override_table(analysis: dict) -> str:
    sweep = analysis.get("override_sweep", [])
    if not sweep:
        return ""
    rows = [
        "<tr><th>size</th><th>param</th><th>steady median (ms)</th>"
        "<th>override peak (ms)</th><th>delta (ms)</th></tr>"
    ]
    for o in sweep:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(o.get('size')))}</td>"
            f"<td>{html.escape(str(o.get('param')))}</td>"
            f"<td>{_ms(o.get('median_ms'))}</td>"
            f"<td>{_ms(o.get('override_ms'))}</td>"
            f"<td><b>{_ms(o.get('override_delta_ms'))}</b></td>"
            "</tr>"
        )
    return (
        "<h2>[live] override sweep (per-param reconfiguration cost)</h2>"
        "<table>" + "".join(rows) + "</table>"
    )


def render_html(analysis: dict, diagnosis: dict, run_dir) -> str:
    op = analysis.get("operator", "?")
    prov = analysis.get("provenance", {})
    plat = analysis.get("platform", {})
    nsys = analysis.get("nsys", {})
    esc = html.escape
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Performance report — {esc(op)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body><div class='wrap'>",
        "<div class='surface'>",
        f"<h1>Performance report — {esc(op)}</h1>",
        f"<div class='meta'>{esc(_plat_line(plat))}</div>",
        f"<div class='meta'>commit <code>{esc(prov.get('git_commit', '?'))}</code>"
        f"{' (dirty)' if prov.get('git_dirty') else ''} · "
        f"{esc(prov.get('timestamp_utc', '?'))} · nsys "
        f"{esc(nsys.get('version', '?'))}</div>",
        "<div class='note'>Latencies are valid only on the machine/GPU captured "
        "on. Kernel time is GPU-side; CUDA-API time is host-side wall — they "
        "overlap, they do not sum. Counts include the discarded process-warmup "
        "session.</div>",
        "</div>",
    ]

    # Findings
    summary = diagnosis.get("summary", {})
    parts.append("<div class='surface'><h2>Findings</h2>")
    parts.append(
        f"<div class='meta'>{summary.get('high', 0)} high · "
        f"{summary.get('medium', 0)} medium · {summary.get('low', 0)} low · "
        f"{summary.get('info', 0)} info</div>"
    )
    for i, f in enumerate(diagnosis.get("findings", []), 1):
        color, label = _SEV_HTML.get(f["severity"], ("#898781", f["severity"]))
        sz = f" · size <code>{esc(str(f['size']))}</code>" if f.get("size") else ""
        ev = _prune_evidence(f.get("evidence"))
        ev_blob = json.dumps(ev, separators=(",", ":")) if ev else ""
        if len(ev_blob) > 600:
            ev_blob = ev_blob[:597] + "..."
        parts.append(
            "<div class='finding'>"
            f"<h3><span class='badge' style='background:{color}'>{label}</span> "
            f"{i}. {esc(f['title'])}{sz}</h3>"
            f"<div>{esc(f['detail'])}</div>"
            f"<div class='rec'><b>Recommendation:</b> {esc(f['recommendation'])}"
            "</div>"
            f"<div class='rubric'>{esc(f['rubric_ref'])}</div>"
            + (f"<div class='evidence'>{esc(ev_blob)}</div>" if ev_blob else "")
            + "</div>"
        )
    parts.append("</div>")

    # Charts (embedded)
    parts.append("<div class='surface'><h2>Dashboard</h2>")
    png = _dashboard_png(run_dir)
    if png:
        parts.append(f"<img class='dash' alt='dashboard' " f"src='data:image/png;base64,{png}'>")
    else:
        parts.append(
            "<div class='muted'>charts omitted — install matplotlib "
            "(<code>pip install matplotlib</code>) to embed the dashboard.</div>"
        )
    parts.append("</div>")

    # Tables
    parts.append("<div class='surface'><h2>Per-size latency &amp; attribution" "</h2>")
    parts.append(_html_latency_table(analysis))
    ovr = _html_override_table(analysis)
    if ovr:
        parts.append(ovr)
    parts.append("</div>")

    parts.append("</div></body></html>")
    return "\n".join(parts)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render markdown + HTML perf reports from analysis + " "diagnosis"
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

    md_out = args.out or (args.run_dir / "report.md")
    md_out.write_text(render(analysis, diagnosis))
    print(md_out)

    html_out = md_out.with_suffix(".html")
    html_out.write_text(render_html(analysis, diagnosis, args.run_dir))
    print(html_out)
    if not _HAVE_VIZ:
        print(
            "[perf-report] matplotlib not found — report.html written without "
            "the chart dashboard (pip install matplotlib to include it)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
