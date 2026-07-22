# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matplotlib visualizations of the performance harness outputs.

Reads an ``analysis.json`` + ``diagnosis.json`` run directory (produced by
``parse.py`` / ``diagnose.py``) and renders the charts ``report.py`` embeds into
``report.html`` (``dashboard()`` composes them all). Each form is chosen by the
data's job:

* **latency by size** — grouped bars (magnitude across a size ladder);
* **kernel vs shell** — a stacked bar per size showing what fraction of the
  steady tick is GPU kernel vs host/graph overhead (the headline: is the tick
  kernel-bound or host-bound);
* **host-API composition** — a 100%-stacked bar (where non-kernel time goes:
  alloc / free / memcpy / sync / launch / module-load);
* **NVTX phases** — grouped bars of per-phase wall vs GPU-projected time (only
  for NVTX-instrumented operators);
* **override sweep** — a log-scale bar of per-param ``override_delta_ms`` (only
  when the operator implements ``--sweep-overrides``);
* **findings** — severity counts.

Colors follow a CVD-validated palette (categorical hues in fixed order; status
palette reserved for severity). Pure matplotlib + stdlib — no pandas.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- palette (CVD-validated, light surface) --------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical hues in fixed order (never cycled/reordered).
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Status palette (reserved for severity — never a series color).
SEVERITY_COLOR = {
    "high": "#d03b3b",  # critical
    "medium": "#ec835a",  # serious
    "low": "#fab219",  # warning
    "info": "#2a78d6",  # (neutral/blue — informational)
}

# host-API categories in a fixed color assignment.
API_CATEGORIES = [
    "free",
    "alloc",
    "memcpy",
    "sync",
    "launch",
    "module_load",
    "other",
]


def load_run(run_dir) -> tuple[dict, dict]:
    """Load (analysis, diagnosis) from a run directory."""
    run_dir = Path(run_dir)
    analysis = json.loads((run_dir / "analysis.json").read_text())
    dpath = run_dir / "diagnosis.json"
    diagnosis = json.loads(dpath.read_text()) if dpath.exists() else {}
    return analysis, diagnosis


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid + muted axes, per the mark spec."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def _ensure_ax(ax, figsize=(7, 4)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _kernel_ms_per_tick(size: dict):
    ticks = (size.get("derived") or {}).get("ticks") or 0
    kt = (size.get("gpu") or {}).get("kernel_ns_total") or 0
    return (kt / ticks / 1e6) if ticks else None


def plot_latency(analysis: dict, ax=None):
    """Grouped bars: cold / median / p90 latency across the size ladder."""
    fig, ax = _ensure_ax(ax)
    sizes = analysis.get("sizes", [])
    labels = [s["size"] for s in sizes]
    series = [
        ("cold", CATEGORICAL[1]),
        ("median", CATEGORICAL[0]),
        ("p90", CATEGORICAL[2]),
    ]
    n = len(series)
    width = 0.8 / n
    x = list(range(len(labels)))
    for i, (key, color) in enumerate(series):
        vals = [(s["latency_ms"].get(key) or 0) for s in sizes]
        offs = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(offs, vals, width * 0.92, label=key, color=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    _style_axes(ax, "Latency by size", ylabel="ms per tick (log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=3)
    return fig


def plot_kernel_fraction(analysis: dict, ax=None):
    """Stacked bar per size: GPU kernel vs host/graph overhead of the tick."""
    fig, ax = _ensure_ax(ax)
    sizes = analysis.get("sizes", [])
    labels = [s["size"] for s in sizes]
    x = list(range(len(labels)))
    kernel, shell = [], []
    for s in sizes:
        med = s["latency_ms"].get("median") or 0
        k = _kernel_ms_per_tick(s) or 0
        k = min(k, med)  # kernel is GPU-side; clamp to the wall tick
        kernel.append(k)
        shell.append(max(med - k, 0))
    ax.bar(x, kernel, 0.6, label="GPU kernel", color=CATEGORICAL[0], zorder=3)
    ax.bar(
        x,
        shell,
        0.6,
        bottom=kernel,
        label="host / graph overhead",
        color=CATEGORICAL[1],
        zorder=3,
    )
    for xi, s in zip(x, sizes):
        med = s["latency_ms"].get("median") or 0
        k = min(_kernel_ms_per_tick(s) or 0, med)
        pct = (100 * k / med) if med else 0
        ax.text(
            xi,
            med,
            f"{pct:.0f}% kernel",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _style_axes(ax, "Where the steady tick goes", ylabel="ms per tick (median)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_host_api(analysis: dict, ax=None):
    """100%-stacked bar per size: host CUDA-API time by category."""
    fig, ax = _ensure_ax(ax)
    sizes = analysis.get("sizes", [])
    labels = [s["size"] for s in sizes]
    x = list(range(len(labels)))
    bottoms = [0.0] * len(sizes)
    for ci, cat in enumerate(API_CATEGORIES):
        vals = []
        for s in sizes:
            cats = s["host_api"]["by_category"]
            total = sum(c.get("ns", 0) for c in cats.values()) or 1
            vals.append(100 * cats.get(cat, {}).get("ns", 0) / total)
        ax.bar(
            x,
            vals,
            0.6,
            bottom=bottoms,
            label=cat,
            color=CATEGORICAL[ci % len(CATEGORICAL)],
            zorder=3,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    _style_axes(ax, "Host CUDA-API time composition", ylabel="% of host-API time")
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    return fig


def plot_phases(analysis: dict, size=None, ax=None):
    """Grouped bars: per-phase wall vs GPU-projected time (instrumented ops)."""
    fig, ax = _ensure_ax(ax)
    sizes = analysis.get("sizes", [])
    chosen = None
    for s in sizes:
        if size is None or s["size"] == size:
            if s.get("phases"):
                chosen = s
                break
    if chosen is None:
        ax.text(
            0.5,
            0.5,
            "no NVTX phases\n(operator not instrumented)",
            ha="center",
            va="center",
            color=INK_MUTED,
            fontsize=11,
            transform=ax.transAxes,
        )
        _style_axes(ax, "compute() NVTX phases")
        ax.grid(False)
        return fig
    phases = chosen["phases"]
    names = [p["phase"] for p in phases]
    wall = [p["wall_avg_ns"] / 1e6 for p in phases]
    ticks = chosen["derived"]["ticks"] or 1
    proj = [p["gpu_proj_ns_total"] / 1e6 / ticks for p in phases]
    y = list(range(len(names)))
    h = 0.38
    ax.barh(
        [yi + h / 2 for yi in y],
        wall,
        h,
        label="wall (per tick)",
        color=CATEGORICAL[1],
        zorder=3,
    )
    ax.barh(
        [yi - h / 2 for yi in y],
        proj,
        h,
        label="GPU-projected (per tick)",
        color=CATEGORICAL[0],
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    _style_axes(ax, f"compute() phases — size {chosen['size']}", xlabel="ms")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_override_sweep(analysis: dict, size=None, ax=None):
    """Log-scale horizontal bar of per-param override_delta_ms."""
    fig, ax = _ensure_ax(ax)
    sweep = analysis.get("override_sweep", [])
    if not sweep:
        ax.text(
            0.5,
            0.5,
            "no override sweep\n(operator lacks --sweep-overrides)",
            ha="center",
            va="center",
            color=INK_MUTED,
            fontsize=11,
            transform=ax.transAxes,
        )
        _style_axes(ax, "[live] override reconfiguration cost")
        ax.grid(False)
        return fig
    if size is None:
        size = sweep[0]["size"]
    rows = [o for o in sweep if o["size"] == size]
    rows.sort(key=lambda o: o.get("override_delta_ms") or 0)
    names = [o["param"] for o in rows]
    vals = [max(o.get("override_delta_ms") or 0, 1e-3) for o in rows]
    # Highlight the costly rebuilds (>= 5 ms) with the serious status hue.
    colors = [SEVERITY_COLOR["medium"] if v >= 5 else CATEGORICAL[0] for v in vals]
    y = list(range(len(names)))
    ax.barh(y, vals, 0.6, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(
            v,
            yi,
            f" {v:.3f}",
            va="center",
            ha="left",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    _style_axes(ax, f"[live] override cost — size {size}", xlabel="delta ms (log)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return fig


def plot_before_after(baseline: dict, after: dict, ax=None):
    """Grouped bars: steady median per size, before vs after an optimization,
    with the per-size % delta labelled. For optimize / autotune reports."""
    fig, ax = _ensure_ax(ax)
    bb = {s["size"]: s for s in baseline.get("sizes", [])}
    aa = {s["size"]: s for s in after.get("sizes", [])}
    sizes = [s["size"] for s in baseline.get("sizes", []) if s["size"] in aa]
    x = list(range(len(sizes)))
    w = 0.4
    before = [bb[s]["latency_ms"].get("median") or 0 for s in sizes]
    after_v = [aa[s]["latency_ms"].get("median") or 0 for s in sizes]
    ax.bar(
        [xi - w / 2 for xi in x],
        before,
        w * 0.92,
        label="before",
        color=CATEGORICAL[1],
        zorder=3,
    )
    ax.bar(
        [xi + w / 2 for xi in x],
        after_v,
        w * 0.92,
        label="after",
        color=CATEGORICAL[0],
        zorder=3,
    )
    for xi, b, a in zip(x, before, after_v):
        if b:
            ax.text(
                xi,
                max(b, a),
                f"{100 * (a - b) / b:+.0f}%",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    _style_axes(ax, "Steady median: before vs after", ylabel="ms per tick (median)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_findings(diagnosis: dict, ax=None):
    """Severity counts of the diagnosis findings."""
    fig, ax = _ensure_ax(ax, figsize=(5, 3))
    order = ["high", "medium", "low", "info"]
    counts = diagnosis.get("summary", {})
    vals = [counts.get(k, 0) for k in order]
    colors = [SEVERITY_COLOR[k] for k in order]
    x = list(range(len(order)))
    ax.bar(x, vals, 0.6, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        if v:
            ax.text(
                xi,
                v,
                str(v),
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    _style_axes(ax, "Findings by severity", ylabel="count")
    return fig


def findings_table(diagnosis: dict) -> str:
    """A compact text table of findings (severity · size · title)."""
    rows = ["severity  size  title"]
    for f in diagnosis.get("findings", []):
        rows.append(f"{f['severity']:<8}  {str(f.get('size') or '-'):<4}  {f['title']}")
    return "\n".join(rows)


def dashboard(run_dir, figsize=(14, 11)):
    """Compose the full per-operator dashboard from one run directory."""
    analysis, diagnosis = load_run(run_dir)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    op = analysis.get("operator", "?")
    plat = analysis.get("platform", {})
    fig.suptitle(
        f"{op}   ·   {plat.get('gpu_name', '?')} · "
        f"CUDA {plat.get('cuda_runtime', '?')} · "
        f"Holoscan {plat.get('holoscan_version', '?')}",
        color=INK_PRIMARY,
        fontsize=14,
        x=0.02,
        ha="left",
    )
    # Use one representative (middle-of-ladder) size for the phase + override
    # panels so the dashboard reads consistently.
    sizes = analysis.get("sizes", [])
    rep = sizes[len(sizes) // 2]["size"] if sizes else None
    gs = fig.add_gridspec(
        3, 2, hspace=0.5, wspace=0.22, top=0.9, bottom=0.08, left=0.08, right=0.97
    )
    plot_kernel_fraction(analysis, ax=fig.add_subplot(gs[0, 0]))
    plot_latency(analysis, ax=fig.add_subplot(gs[0, 1]))
    plot_host_api(analysis, ax=fig.add_subplot(gs[1, 0]))
    plot_phases(analysis, size=rep, ax=fig.add_subplot(gs[1, 1]))
    plot_override_sweep(analysis, size=rep, ax=fig.add_subplot(gs[2, 0]))
    if diagnosis:
        plot_findings(diagnosis, ax=fig.add_subplot(gs[2, 1]))
    return fig


def compare_kernel_fraction(run_dirs, ax=None):
    """Cross-operator: kernel % of the steady tick. Color = operator (legend),
    x-tick = size, so long operator names never collide on the axis."""
    from matplotlib.patches import Patch

    fig, ax = _ensure_ax(ax, figsize=(8, 4.5))
    labels, fracs, colors = [], [], []
    legend = []
    for i, rd in enumerate(run_dirs):
        analysis, _ = load_run(rd)
        op = analysis.get("operator", Path(rd).name)
        color = CATEGORICAL[i % len(CATEGORICAL)]
        legend.append(Patch(facecolor=color, label=op))
        for s in analysis.get("sizes", []):
            med = s["latency_ms"].get("median") or 0
            k = min(_kernel_ms_per_tick(s) or 0, med)
            labels.append(s["size"])
            fracs.append(100 * k / med if med else 0)
            colors.append(color)
    x = list(range(len(labels)))
    ax.bar(x, fracs, 0.6, color=colors, zorder=3)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    for xi, v in zip(x, fracs):
        ax.text(
            xi,
            v,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    _style_axes(
        ax,
        "Kernel share of the steady tick (host-bound < 50% < kernel-bound)",
        ylabel="% of tick that is GPU kernel",
    )
    ax.legend(
        handles=legend,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        loc="upper left",
    )
    return fig
