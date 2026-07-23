# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an optimization report (before -> after) for perf-optimize / autotune.

Folds a baseline and a final ``analysis.json`` (from perf-analyze runs) plus a
``session.json`` the optimizing skill accumulates into one report that shows:

* which bottleneck(s) each change targeted,
* how much the steady runtime improved (per-size before -> after + delta),
* what changed (file / diffstat summary) and the gate results
  (correctness, byte-equivalence, speed),

with an embedded before/after chart. Emits ``optimization_report.md`` and a
self-contained ``optimization_report.html`` (charts need matplotlib; without it
the HTML is written text-only).

session.json schema (the skill writes this)::

    {
      "operator": "FMCWRangeProcessorOp",
      "mode": "optimize" | "autotune",
      "changes": [
        {"finding": "per-tick alloc / host-bound tick",
         "summary": "Cache the range_bins axis ...",
         "files": ["operators/.../fmcw_range_processor_op.cu"],
         "diffstat": "+31 -5",
         "gates": {"correctness": "11/11 unit tests pass",
                   "byte_equivalent": true,
                   "speed_delta_pct": {"xs": -12.6, "md": -24.9, "xl": -29.5}},
         "kept": true}
      ],
      "notes": "optional free text"
    }
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import visualize as viz  # same directory; on sys.path for CLI use

    _HAVE_VIZ = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_VIZ = False

_HTML_STYLE = """
:root { color-scheme: light; }
body { background:#f9f9f7; color:#0b0b0b; margin:0; padding:2rem;
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1040px; margin:0 auto; }
.surface { background:#fcfcfb; border:1px solid rgba(11,11,11,0.10);
  border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1.25rem; }
h1 { font-size:1.4rem; margin:0 0 .5rem; }
h2 { font-size:1.1rem; margin:1.5rem 0 .75rem; }
h3 { font-size:1rem; margin:.4rem 0; }
.meta, .muted { color:#52514e; font-size:.9rem; }
.change { border-top:1px solid #e1e0d9; padding:.9rem 0; }
.gate { display:inline-block; margin-right:1rem; font-size:.9rem; }
.ok { color:#0ca30c; font-weight:600; }
.win { color:#006300; font-weight:600; }
table { border-collapse:collapse; width:100%; font-size:.86rem;
  font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:.35rem .6rem; border-bottom:1px solid #e1e0d9; }
th { color:#52514e; font-weight:600; }
img.chart { width:100%; max-width:720px; height:auto; border:1px solid #e1e0d9;
  border-radius:8px; }
code { font-family: ui-monospace,Menlo,monospace; font-size:.82rem; }
"""


def _median(analysis: dict, size: str):
    for s in analysis.get("sizes", []):
        if s["size"] == size:
            return s["latency_ms"].get("median")
    return None


def _size_order(baseline: dict, final: dict) -> list:
    fin = {s["size"] for s in final.get("sizes", [])}
    return [s["size"] for s in baseline.get("sizes", []) if s["size"] in fin]


def _overall_rows(baseline: dict, final: dict) -> list:
    """[(size, before_ms, after_ms, delta_pct), ...]."""
    rows = []
    for size in _size_order(baseline, final):
        b = _median(baseline, size)
        a = _median(final, size)
        d = (100 * (a - b) / b) if (b and a is not None) else None
        rows.append((size, b, a, d))
    return rows


def _variant_median(analysis: dict, size: str, variant: str):
    for s in analysis.get("sizes", []):
        if s["size"] == size:
            for v in s.get("variants", []):
                if v["variant"] == variant:
                    return v["latency_ms"].get("median")
    return None


def _has_variants(*analyses) -> bool:
    return any(
        len(s.get("variants", [])) > 1 for a in analyses for s in a.get("sizes", [])
    )


def _variant_rows(baseline: dict, final: dict) -> list:
    """[(size, variant, before_ms, after_ms, delta_pct), ...] across the swept
    primary axis, so a per-variant optimization is verified on every mode."""
    rows = []
    for size in _size_order(baseline, final):
        fin = next((s for s in final.get("sizes", []) if s["size"] == size), {})
        for v in fin.get("variants", []):
            var = v["variant"]
            b = _variant_median(baseline, size, var)
            a = _variant_median(final, size, var)
            d = (100 * (a - b) / b) if (b and a is not None) else None
            rows.append((size, var, b, a, d))
    return rows


def _fmt(v, prec=3):
    return "-" if v is None else f"{v:.{prec}f}"


def _chart_png(baseline: dict, final: dict) -> str | None:
    if not _HAVE_VIZ:
        return None
    try:
        fig = viz.plot_before_after(baseline, final)
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=110,
            facecolor=viz.SURFACE,
            bbox_inches="tight",
        )
        viz.plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # pragma: no cover - defensive
        return None


def render_md(session: dict, baseline: dict, final: dict) -> str:
    op = session.get("operator", final.get("operator", "?"))
    mode = session.get("mode", "optimize")
    plat = final.get("platform", {})
    prov = final.get("provenance", {})
    kept = [c for c in session.get("changes", []) if c.get("kept", True)]
    lines = [f"# Optimization report — {op} ({mode})", ""]
    lines.append(
        f"- **Platform:** {plat.get('gpu_name', '?')} | CUDA "
        f"{plat.get('cuda_runtime', '?')} | Holoscan "
        f"{plat.get('holoscan_version', '?')}"
    )
    lines.append(
        f"- **Commit:** `{prov.get('git_commit', '?')}` — "
        f"{prov.get('timestamp_utc', '?')}"
    )
    lines.append(f"- **Changes kept:** {len(kept)}")
    lines.append("")

    lines.append("## Runtime improvement (steady median)")
    lines.append("")
    lines.append("| size | before (ms) | after (ms) | delta |")
    lines.append("|---|---|---|---|")
    for size, b, a, d in _overall_rows(baseline, final):
        dtxt = "-" if d is None else f"{d:+.1f}%"
        lines.append(f"| {size} | {_fmt(b)} | {_fmt(a)} | **{dtxt}** |")
    lines.append("")

    if _has_variants(baseline, final):
        lines.append("## Runtime improvement by variant (primary axis × size)")
        lines.append("")
        lines.append("| size | variant | before (ms) | after (ms) | delta |")
        lines.append("|---|---|---|---|---|")
        for size, var, b, a, d in _variant_rows(baseline, final):
            dtxt = "-" if d is None else f"{d:+.1f}%"
            lines.append(f"| {size} | {var} | {_fmt(b)} | {_fmt(a)} | **{dtxt}** |")
        lines.append("")

    lines.append("## Changes")
    lines.append("")
    for i, c in enumerate(kept, 1):
        lines.append(f"### {i}. {c.get('summary', '(change)')}")
        lines.append("")
        lines.append(f"- **Bottleneck:** {c.get('finding', '?')}")
        if c.get("files"):
            lines.append(
                f"- **Files:** {', '.join(c['files'])}" f" ({c.get('diffstat', '')})"
            )
        g = c.get("gates", {})
        se = g.get("speed_delta_pct", {})
        se_txt = ", ".join(f"{k} {v:+.1f}%" for k, v in se.items())
        lines.append(
            f"- **Gates:** correctness — {g.get('correctness', '?')}; "
            f"byte-equivalent — {'yes' if g.get('byte_equivalent') else 'NO'}; "
            f"speed — {se_txt or 'n/a'}"
        )
        lines.append("")

    if session.get("notes"):
        lines.append("## Notes")
        lines.append("")
        lines.append(session["notes"])
        lines.append("")
    return "\n".join(lines) + "\n"


def render_html(session: dict, baseline: dict, final: dict) -> str:
    op = session.get("operator", final.get("operator", "?"))
    mode = session.get("mode", "optimize")
    plat = final.get("platform", {})
    prov = final.get("provenance", {})
    esc = html.escape
    kept = [c for c in session.get("changes", []) if c.get("kept", True)]
    p = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Optimization report — {esc(op)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body><div class='wrap'>",
        "<div class='surface'>",
        f"<h1>Optimization report — {esc(op)} <span class='muted'>({esc(mode)})"
        "</span></h1>",
        f"<div class='meta'>{esc(plat.get('gpu_name', '?'))} · CUDA "
        f"{esc(plat.get('cuda_runtime', '?'))} · Holoscan "
        f"{esc(plat.get('holoscan_version', '?'))} · commit "
        f"<code>{esc(prov.get('git_commit', '?'))}</code></div>",
        f"<div class='meta'>{len(kept)} change(s) kept</div>",
        "</div>",
    ]

    # Runtime improvement table + chart
    p.append("<div class='surface'><h2>Runtime improvement (steady median)</h2>")
    p.append(
        "<table><tr><th>size</th><th>before (ms)</th><th>after (ms)</th>"
        "<th>delta</th></tr>"
    )
    for size, b, a, d in _overall_rows(baseline, final):
        cls = "win" if (d is not None and d < 0) else ""
        dtxt = "-" if d is None else f"{d:+.1f}%"
        p.append(
            f"<tr><td>{esc(size)}</td><td>{_fmt(b)}</td><td>{_fmt(a)}</td>"
            f"<td class='{cls}'>{dtxt}</td></tr>"
        )
    p.append("</table>")
    png = _chart_png(baseline, final)
    if png:
        p.append(
            f"<p><img class='chart' alt='before/after' "
            f"src='data:image/png;base64,{png}'></p>"
        )
    if _has_variants(baseline, final):
        p.append("<h3>By variant (primary axis × size)</h3>")
        p.append(
            "<table><tr><th>size</th><th>variant</th><th>before (ms)</th>"
            "<th>after (ms)</th><th>delta</th></tr>"
        )
        for size, var, b, a, d in _variant_rows(baseline, final):
            cls = "win" if (d is not None and d < 0) else ""
            dtxt = "-" if d is None else f"{d:+.1f}%"
            p.append(
                f"<tr><td>{esc(size)}</td><td>{esc(str(var))}</td>"
                f"<td>{_fmt(b)}</td><td>{_fmt(a)}</td>"
                f"<td class='{cls}'>{dtxt}</td></tr>"
            )
        p.append("</table>")
    p.append("</div>")

    # Changes
    p.append("<div class='surface'><h2>Changes</h2>")
    for i, c in enumerate(kept, 1):
        g = c.get("gates", {})
        se = g.get("speed_delta_pct", {})
        se_txt = ", ".join(f"{k} {v:+.1f}%" for k, v in se.items())
        be = "yes" if g.get("byte_equivalent") else "NO"
        files = ""
        if c.get("files"):
            files = (
                f"<div class='meta'>Files: "
                f"{esc(', '.join(c['files']))} "
                f"({esc(c.get('diffstat', ''))})</div>"
            )
        p.append(
            "<div class='change'>"
            f"<h3>{i}. {esc(c.get('summary', '(change)'))}</h3>"
            f"<div class='meta'>Bottleneck: {esc(c.get('finding', '?'))}</div>"
            f"{files}"
            f"<div><span class='gate'>correctness "
            f"<span class='ok'>{esc(g.get('correctness', '?'))}</span></span>"
            f"<span class='gate'>byte-equivalent <span class='ok'>{be}</span>"
            "</span>"
            f"<span class='gate'>speed <span class='win'>{esc(se_txt)}</span>"
            "</span></div></div>"
        )
    p.append("</div>")

    if session.get("notes"):
        p.append(
            f"<div class='surface'><h2>Notes</h2><div class='meta'>"
            f"{esc(session['notes'])}</div></div>"
        )
    p.append("</div></body></html>")
    return "\n".join(p)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a before/after optimization report")
    p.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="baseline analysis.json (before the changes)",
    )
    p.add_argument(
        "--final",
        type=Path,
        required=True,
        help="final analysis.json (after the changes)",
    )
    p.add_argument(
        "--session",
        type=Path,
        required=True,
        help="session.json describing the changes + gates",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output directory for optimization_report.{md,html}",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    baseline = json.loads(args.baseline.read_text())
    final = json.loads(args.final.read_text())
    session = json.loads(args.session.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md = args.out_dir / "optimization_report.md"
    md.write_text(render_md(session, baseline, final))
    print(md)
    html_out = args.out_dir / "optimization_report.html"
    html_out.write_text(render_html(session, baseline, final))
    print(html_out)
    if not _HAVE_VIZ:
        print(
            "[optimize-report] matplotlib not found — HTML written without "
            "the before/after chart"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
