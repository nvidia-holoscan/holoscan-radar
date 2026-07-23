# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fold nsys stat tables + the schema-v2 perf JSON into one analysis.json.

Reads a ``runner_manifest.json`` (from ``runner.py``) and, per (operator,
size), joins:

* the **clean** perf JSON — latency (setup/cold/median/p90), override delta,
  throughput, provenance/platform;
* **cuda_gpu_kern_sum** — GPU kernel time breakdown;
* **cuda_api_sum** — CUDA API time, categorized into alloc / free / memcpy /
  sync / launch / module-load (the ``cudaMalloc``/``cudaFree`` split the
  diagnosis rubric keys on);
* **cuda_gpu_mem_time_sum** — memcpy H2D/D2H by time;
* **nvtx_pushpop_sum** — per-compute()-phase wall time (our RADAR_NVTX ranges);
* **nvtx_gpu_proj_sum** — per-phase GPU-*projected* time, so a stage-kernel
  phase reports true GPU-kernel duration, not just the enqueue wall-time.

The join is deliberately NVTX-phase-with-GPU-timeline, per the plan's caveat: a
compute() phase range bounds kernel *enqueue*; its real GPU cost comes from the
projection table, never from the phase wall-time.

Emits ``analysis.json`` (schema ``perf-analysis/v1``) next to the manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- nsys report field keys (Nsight Systems 2026.2.1) ----------------------
_TOTAL_NS = "Total Time (ns)"
_PROJ_NS = "Total Proj Time (ns)"
_RANGE_NS = "Total Range Time (ns)"


def _load(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return d if isinstance(d, list) else []


# CUDA API name -> category. First matching predicate wins; the categories the
# rubric cares about (alloc/free/sync) are checked before the generic ones.
def _api_category(name: str) -> str:
    low = name.lower()
    if "free" in low:  # cudaFree, cudaFreeAsync, cuMemFree
        return "free"
    if "malloc" in low or "memalloc" in low or "memcreate" in low or "hostalloc" in low:
        return "alloc"
    if "memcpy" in low:
        return "memcpy"
    if "memset" in low:
        return "memset"
    if "synchronize" in low:
        return "sync"
    if "launchkernel" in low or low in ("culaunchkernel",):
        return "launch"
    if (
        "moduleload" in low
        or "libraryload" in low
        or "moduleunload" in low
        or "modulegetfunction" in low
    ):
        return "module_load"
    return "other"


def _strip_domain(range_name: str) -> str:
    """nsys prefixes default-domain NVTX ranges with ':'. Drop it."""
    return range_name[1:] if range_name.startswith(":") else range_name


def _short_kernel(name: str) -> str:
    """A readable short label for a mangled kernel name."""
    if "vector_fft" in name:
        return "cuFFT vector_fft"
    m = re.search(r"matx::detail::(\w+)", name)
    if m:
        return f"matx {m.group(1)}"
    m = re.search(r"(\w+)\s*<", name)
    if m:
        return m.group(1)
    return name[:60]


def parse_api_sum(rows: list) -> dict:
    """Categorize cuda_api_sum into {category: {ns, calls}} + per-name detail."""
    cats: dict = {}
    by_name = {}
    for r in rows:
        name = r.get("Name", "")
        ns = r.get(_TOTAL_NS, 0) or 0
        calls = r.get("Num Calls", 0) or 0
        cat = _api_category(name)
        c = cats.setdefault(cat, {"ns": 0, "calls": 0})
        c["ns"] += ns
        c["calls"] += calls
        by_name[name] = {"ns": ns, "calls": calls, "category": cat}
    return {"by_category": cats, "by_name": by_name}


def parse_mem_time(rows: list) -> dict:
    """cuda_gpu_mem_time_sum -> h2d/d2h/memset ns by Operation string."""
    out = {"h2d_ns": 0, "d2h_ns": 0, "dtod_ns": 0, "memset_ns": 0}
    for r in rows:
        op = (r.get("Operation") or "").lower()
        ns = r.get(_TOTAL_NS, 0) or 0
        if "host-to-device" in op:
            out["h2d_ns"] += ns
        elif "device-to-host" in op:
            out["d2h_ns"] += ns
        elif "device-to-device" in op:
            out["dtod_ns"] += ns
        elif "memset" in op:
            out["memset_ns"] += ns
    return out


def parse_kernels(rows: list) -> dict:
    total = 0
    kernels = []
    for r in rows:
        ns = r.get(_TOTAL_NS, 0) or 0
        inst = r.get("Instances", 0) or 0
        total += ns
        kernels.append(
            {
                "name": _short_kernel(r.get("Name", "")),
                "raw_name": r.get("Name", ""),
                "ns": ns,
                "instances": inst,
            }
        )
    kernels.sort(key=lambda k: k["ns"], reverse=True)
    return {"kernel_ns_total": total, "kernels": kernels}


def parse_phases(pushpop: list, proj: list, op_name: str) -> dict:
    """Join per-phase wall time (pushpop) with GPU-projected time (proj).

    Keeps only this operator's compute()/setup phases (range name starts with
    the operator class name); also surfaces the operator's GXF tick range as
    whole-op-per-tick GPU time.
    """
    proj_by_range = {r.get("Range", ""): r for r in proj}
    phases = []
    for r in pushpop:
        raw = r.get("Range", "")
        name = _strip_domain(raw)
        if not name.startswith(op_name):
            continue
        pr = proj_by_range.get(raw, {})
        short = name.split(":")[-1]  # e.g. "stage-kernel"
        inst = r.get("Instances", 0) or 0
        phases.append(
            {
                "phase": short,
                "range": name,
                "wall_ns_total": r.get(_TOTAL_NS, 0) or 0,
                "instances": inst,
                "wall_avg_ns": r.get("Avg (ns)", 0) or 0,
                "gpu_proj_ns_total": pr.get(_PROJ_NS, 0) or 0,
                "gpu_ops": pr.get("Total GPU Ops", 0) or 0,
            }
        )
    phases.sort(key=lambda p: p["wall_ns_total"], reverse=True)

    # Whole-op GPU per tick: the GXF "tick codelet: <instance>" range with the
    # most GPU-projected time is the operator under test.
    whole = None
    for r in proj:
        rng = r.get("Range", "")
        if rng.startswith("GXF:tick codelet:"):
            cand = {
                "range": rng,
                "gpu_proj_ns_total": r.get(_PROJ_NS, 0) or 0,
                "wall_ns_total": r.get(_RANGE_NS, 0) or 0,
                "instances": r.get("Range Instances", 0) or 0,
            }
            if whole is None or cand["gpu_proj_ns_total"] > whole["gpu_proj_ns_total"]:
                whole = cand
    return {"phases": phases, "whole_op_tick": whole}


def _compute_row(perf_json: dict):
    for r in perf_json.get("results", []):
        if r.get("bucket") == "compute":
            return r
    return None


def _compute_rows(perf_json: dict) -> list:
    """All compute rows — one per primary-axis variant when the clean run swept
    ``--sweep-variants`` (else a single default-variant row)."""
    return [r for r in perf_json.get("results", []) if r.get("bucket") == "compute"]


def _setup_row(perf_json: dict):
    for r in perf_json.get("results", []):
        if r.get("bucket") == "setup":
            return r
    return None


def _n_samples(clean_json: dict) -> int:
    """Pooled sample count behind each latency stat = frames x repeats."""
    rs = clean_json.get("run_settings", {})
    return int(rs.get("measured_frames") or 0) * int(rs.get("repeats") or 1)


def _variant_tag(comp: dict) -> str:
    """Primary-axis label for a compute row (e.g. msub/sdel/ddel); 'default' when
    the operator has no primary categorical axis."""
    return comp.get("variant") or "default"


def _latency_of(comp: dict, setup: dict, n: int) -> dict:
    """Latency block for one compute row, carrying the full spread (min/p99/max)
    and the trial count so the report can draw whiskers and print N."""
    return {
        "setup": setup.get("setup_ms"),
        "cold": comp.get("cold_ms"),
        "min": comp.get("min_ms"),
        "median": comp.get("median_ms"),
        "p90": comp.get("p90_ms"),
        "p99": comp.get("p99_ms"),
        "max": comp.get("max_ms"),
        "n": n,
        "override": comp.get("override_ms"),
        "override_delta": comp.get("override_delta_ms"),
    }


def _tick_count(phases: list) -> int:
    """Compute-tick count = max instances among the per-tick compute phases
    (setup/initialize run ~once, so they are excluded by the max)."""
    per_tick = [
        p["instances"] for p in phases if p["phase"] not in ("setup", "initialize")
    ]
    return max(per_tick) if per_tick else 0


def build_size_analysis(size: str, clean_json: dict, stats: dict, op_name: str) -> dict:
    comps = _compute_rows(clean_json)
    comp = comps[0] if comps else {}  # primary/default variant (back-compat)
    setup = _setup_row(clean_json) or {}
    n = _n_samples(clean_json)

    kern = parse_kernels(_load(stats.get("cuda_gpu_kern_sum")))
    api = parse_api_sum(_load(stats.get("cuda_api_sum")))
    mem = parse_mem_time(_load(stats.get("cuda_gpu_mem_time_sum")))
    ph = parse_phases(
        _load(stats.get("nvtx_pushpop_sum")),
        _load(stats.get("nvtx_gpu_proj_sum")),
        op_name,
    )

    # Compute-tick count: prefer our per-tick NVTX phases; for operators not yet
    # NVTX-instrumented fall back to Holoscan's GXF tick range, then to the
    # most-frequent kernel's instance count.
    ticks = _tick_count(ph["phases"])
    if not ticks and ph["whole_op_tick"]:
        ticks = ph["whole_op_tick"].get("instances") or 0
    if not ticks and kern["kernels"]:
        ticks = max((k["instances"] for k in kern["kernels"]), default=0)
    cats = api["by_category"]

    def cat_ns(name):
        return cats.get(name, {}).get("ns", 0)

    def cat_calls(name):
        return cats.get(name, {}).get("calls", 0)

    alloc_free_ns = cat_ns("alloc") + cat_ns("free")
    host_api_ns = sum(c.get("ns", 0) for c in cats.values())
    derived = {
        "ticks": ticks,
        "kernel_ns_per_tick": kern["kernel_ns_total"] / ticks if ticks else None,
        "alloc_free_ns_per_tick": alloc_free_ns / ticks if ticks else None,
        "alloc_calls_per_tick": cat_calls("alloc") / ticks if ticks else None,
        "free_calls_per_tick": cat_calls("free") / ticks if ticks else None,
        "memcpy_calls_per_tick": cat_calls("memcpy") / ticks if ticks else None,
        "sync_calls_per_tick": cat_calls("sync") / ticks if ticks else None,
        # Host-API time share (whole run) — where non-kernel time goes.
        "host_api_pct": {
            k: (100.0 * v.get("ns", 0) / host_api_ns) if host_api_ns else 0.0
            for k, v in cats.items()
        },
        "host_api_ns_total": host_api_ns,
    }

    # Per-variant latency (one row per primary-axis value the clean run swept).
    # Attribution (gpu / host_api / phases / derived) is size-level: the profiled
    # nsys run covers the primary variant only, so it is shared across variants.
    variants = [
        {
            "variant": _variant_tag(c),
            "map_shape": c.get("map_shape"),
            "latency_ms": _latency_of(c, setup, n),
            "throughput": {
                "cells_per_second": c.get("cells_per_second"),
                "estimated_effective_gbps": c.get("estimated_effective_gbps"),
            },
        }
        for c in comps
    ]

    return {
        "size": size,
        "variant": _variant_tag(comp),  # primary/default variant label
        "n": n,
        "input_shape": comp.get("input_shape"),
        "map_shape": comp.get("map_shape"),
        "latency_ms": _latency_of(comp, setup, n),  # primary variant (back-compat)
        "throughput": {
            "cells_per_second": comp.get("cells_per_second"),
            "estimated_effective_gbps": comp.get("estimated_effective_gbps"),
        },
        "gpu": {
            "kernel_ns_total": kern["kernel_ns_total"],
            "kernels": kern["kernels"],
            "memcpy": mem,
        },
        "host_api": {
            "by_category": cats,
            "by_name": api["by_name"],
        },
        "phases": ph["phases"],
        "whole_op_tick": ph["whole_op_tick"],
        "derived": derived,
        "variants": variants,
    }


def _override_param(row: dict) -> str:
    """Best-effort [live] param name from an override-sweep compute row."""
    sid = row.get("scenario_id", "")
    size = row.get("size_id", "")
    if "_ovr_" in sid:
        rest = sid.split("_ovr_", 1)[1]
        for suf in (f"_{size}_compute", "_compute", f"_{size}"):
            if rest.endswith(suf):
                return rest[: -len(suf)]
    m = re.search(r"live (\w+) param", row.get("notes", ""))
    return m.group(1) if m else sid


def build_override_sweep(sweep_json: dict) -> list:
    out = []
    for r in sweep_json.get("results", []):
        if r.get("bucket") != "compute":
            continue
        out.append(
            {
                "param": _override_param(r),
                "size": r.get("size_id"),
                "scenario_id": r.get("scenario_id"),
                "median_ms": r.get("median_ms"),
                "override_ms": r.get("override_ms"),
                "override_delta_ms": r.get("override_delta_ms"),
            }
        )
    out.sort(key=lambda x: (x["size"] or "", -(x["override_delta_ms"] or 0)))
    return out


def analyze(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    op_out = manifest_path.parent

    # Operator class name (e.g. FMCWRangeProcessorOp) from any clean perf JSON.
    op_name = manifest["operator"]
    first_clean = {}
    for ps in manifest.get("per_size", []):
        first_clean = json.loads(Path(ps["clean_json"]).read_text())
        break
    comp = _compute_row(first_clean)
    if comp and comp.get("operator"):
        op_name = comp["operator"]

    sizes = []
    for ps in manifest.get("per_size", []):
        clean = json.loads(Path(ps["clean_json"]).read_text())
        sizes.append(
            build_size_analysis(ps["size"], clean, ps.get("stats_json", {}), op_name)
        )

    sweep = {}
    swp = manifest.get("clean_override_sweep_json")
    if swp and Path(swp).exists():
        sweep = json.loads(Path(swp).read_text())

    analysis = {
        "schema": "perf-analysis/v1",
        "operator": op_name,
        "provenance": first_clean.get("provenance", {}),
        "platform": first_clean.get("platform", {}),
        "run_settings": first_clean.get("run_settings", {}),
        "nsys": {
            "version": manifest.get("nsys_version", ""),
            "trace": manifest.get("trace", ""),
            "reports": manifest.get("reports", []),
            "note": "attribution counts include the binary's discarded "
            "process-warmup session; kernel time is GPU-side, API "
            "time is host-side wall — they overlap, not sum",
        },
        "sizes": sizes,
        "override_sweep": build_override_sweep(sweep),
    }
    out = op_out / "analysis.json"
    out.write_text(json.dumps(analysis, indent=2))
    print(out)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fold nsys tables + perf JSON into analysis.json"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--manifest", type=Path, help="runner_manifest.json produced by runner.py"
    )
    g.add_argument(
        "--run-dir", type=Path, help="directory holding runner_manifest.json"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = args.manifest or (args.run_dir / "runner_manifest.json")
    if not manifest.exists():
        raise SystemExit(f"[perf-parse] manifest not found: {manifest}")
    analyze(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
