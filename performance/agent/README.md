<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# performance/agent — operator perf-optimization harness

A deterministic measurement/analysis substrate that turns the manual
"eyeball an nsys profile and write prose in `OPEN_ITEMS.md`" loop into a
repeatable pipeline. It profiles one operator across small/medium/large
workloads, sweeps every runtime-mutable (`[live]`) override param, attributes
time to kernel vs `cudaMalloc`/memcpy/sync vs `compute()` NVTX phases, and emits
a ranked, evidence-backed perf report.

The `perf-optimize` agent skill (`.claude/skills/perf-optimize/`) drives it;
this README is the substrate reference.

## Layout

| File | Role |
| --- | --- |
| `runner.py` | Runs `perf_<op>` clean (latency) + under `nsys` (attribution); exports the nsys stat tables. Writes `runner_manifest.json`. |
| `parse.py` | Folds the nsys tables + the schema-v2 perf JSON into one `analysis.json` per (operator, size, scenario). |
| `diagnose.py` | Encodes the `docs/planning/OPERATOR_DEVELOPMENT.md` perf rubric + CFAR-lessons table as rules over `analysis.json`; emits ranked `diagnosis.json`. |
| `report.py` | Renders `analysis.json` + `diagnosis.json` into a markdown perf report. |
| `visualize.py` | Matplotlib charts of a run (kernel-vs-shell, latency, host-API composition, NVTX phases, override sweep, findings) + a cross-operator comparison. |
| `visualize.ipynb` | Thin notebook that composes `visualize.py` into a per-operator dashboard. |

## Prerequisites

Everything runs **inside `./holohub run-container`** — the host has no `nsys`,
no CUDA toolchain, and no built binary. `runner.py` asserts this in preflight.

The operator must be built with NVTX instrumentation on for the per-phase split:

```bash
cmake -S . -B build-perf -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON \
  -DHOLOSCAN_RADAR_PERF_NVTX=ON -DBUILD_ALL=ON
cmake --build build-perf --target perf_fmcw_range_processor -j"$(nproc)"
```

`HOLOSCAN_RADAR_PERF_NVTX=ON` adds the `RADAR_NVTX` define to the operator
targets (via `radar_enable_perf_nvtx()` in `cmake/RadarPerfNvtx.cmake`), turning
the `radar_common/sdk/nvtx.hpp` ranges from no-ops into real NVTX ranges. A
normal build keeps a zero-overhead hot path.

## Run

One shot (runner → parse → diagnose → report), from the repo root:

```bash
performance/agent/run.sh fmcw_range_processor xs,md,xl
# -> performance/agent/runs/fmcw_range_processor/report.md
```

Or run the stages individually (to inspect an intermediate, or re-render a
report without re-profiling):

```bash
OP=fmcw_range_processor
python3 performance/agent/runner.py   --operator $OP --build-dir build-perf \
    --out-dir performance/agent/runs --sizes xs,md,xl
python3 performance/agent/parse.py    --run-dir performance/agent/runs/$OP
python3 performance/agent/diagnose.py --run-dir performance/agent/runs/$OP
python3 performance/agent/report.py   --run-dir performance/agent/runs/$OP
# -> performance/agent/runs/$OP/report.md
```

## Visualize (optional)

`report.md` is the primary deliverable; for charts, open `visualize.ipynb`
(needs `pip install matplotlib` — not a harness dependency, and the charts read
plain JSON so this runs on the host, no GPU). Set `RUN` to a run directory and
run all cells for a dashboard: kernel-vs-shell (is the tick kernel- or
host-bound), latency by size, host-API composition, `compute()` NVTX phases,
the `[live]` override sweep, and findings — plus a cross-operator comparison.
The plotting functions in `visualize.py` are usable directly, e.g.

```python
import sys; sys.path.insert(0, "performance/agent")
import visualize as viz
viz.dashboard("performance/agent/runs/fmcw_range_processor")
```

Artifacts (`.nsys-rep`, `.sqlite`, stat JSON, `analysis.json`, `diagnosis.json`,
`report.md`) land under `performance/agent/runs/<op>/` (gitignored) in the
mounted repo tree, so they persist on the host.

## How attribution works (and its limits)

- **Latency** always comes from the **clean** run; **attribution** always from
  the **profiled** run. nsys perturbs timing, so the two are never mixed.
- A `compute()` NVTX phase range bounds the kernel **enqueue**, not the async GPU
  execution. `parse.py` therefore joins the phases with the CUDA GPU timeline via
  `nvtx_gpu_proj_sum` (per-phase GPU-*projected* time), never treating NVTX
  wall-time as GPU time.
- Attribution is **per-size steady** (the steady path is identical across
  override params); per-param reconfiguration cost rides the clean
  `--sweep-overrides` `override_delta_ms`.
- nsys stat counts include the binary's **discarded process-warmup** session.
- Kernel time is GPU-side; CUDA-API time (`cudaMalloc`/`cudaFree`/sync) is
  host-side wall. They **overlap**, they do not sum.

## nsys report tables consumed

`nvtx_pushpop_sum`, `nvtx_gpu_proj_sum`, `cuda_gpu_kern_sum`, `cuda_api_sum`,
`cuda_gpu_mem_time_sum` (verified against Nsight Systems 2026.2.1). The SQLite
export (`nsys export --type sqlite`) is kept beside each `.nsys-rep` as a robust
fallback if a report name differs on another nsys version.

## Extending to another operator

The runner/parser/diagnoser are operator-agnostic. For full per-phase +
per-param detail a new operator needs, in its own PR:

1. NVTX ranges in its `compute()` (`RADAR_NVTX_RANGE` / `nvtx::Phase` from
   `radar_common/sdk/nvtx.hpp`) and `radar_enable_perf_nvtx(<target>)` in its
   operator `CMakeLists.txt`.
2. A `--sweep-overrides` matrix in its `perf_<op>.cpp` (an `OverrideKind` enum
   over its `[live]` params + a `wire_override()` switch + `build_override_sweep`
   in its Traits), mirroring `perf_fmcw_range_processor.cpp`.

Without those, the harness still reports kernel/alloc/memcpy attribution and
latency — only the per-phase and per-param columns are empty.
```
