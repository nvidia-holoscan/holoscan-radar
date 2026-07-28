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

The `perf-analyze` agent skill (`skills/perf-analyze/`) drives it; this README is
the substrate reference.

## Layout

| File | Role |
| --- | --- |
| `runner.py` | Runs `perf_<op>` clean (latency) + under `nsys` (attribution); exports the nsys stat tables. Writes `runner_manifest.json`. |
| `parse.py` | Folds the nsys tables + the schema-v2 perf JSON into one `analysis.json` per (operator, size, scenario). |
| `diagnose.py` | Encodes the `docs/planning/OPERATOR_DEVELOPMENT.md` perf rubric + CFAR-lessons table as rules over `analysis.json`; emits ranked `diagnosis.json`. |
| `report.py` | Renders `report.md` (text) + a self-contained `report.html` with the chart dashboard embedded as a base64 PNG. |
| `visualize.py` | Matplotlib charts of a run (kernel-vs-shell, latency, host-API composition, NVTX phases, override sweep, findings, before/after) + a cross-operator comparison; used by `report.py` / `optimize_report.py` and importable directly. |
| `optimize_report.py` | Before→after optimization report (md + html) for the `perf-optimize` / `perf-autotune` skills: bottlenecks targeted, runtime improvement, changes + gate results, embedded before/after chart. |
| `compare_builds.py` | Runs the perf binaries out of two build trees with identical args and diffs the steady-state medians; writes `compare.md` + `compare.json`. |

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

## Branch-level before/after (`build-before/`)

`runner.py` characterizes *one* build. To ask "what did this optimization
branch actually buy us", build the pre-optimization tree once and diff the two:

```bash
# Build a baseline tree first (see below), then:
python3 performance/agent/compare_builds.py \
    --before-build <baseline build dir> --after-build build \
    --sizes sm,md,xl --checksum
# -> performance/agent/runs/_compare/compare.md
```

Building the baseline side has one non-obvious requirement: **overlay the
current `performance/` onto the baseline checkout before configuring it.** Any
branch that changes the harness as well as the operators (size ladders,
live/sustained modes, the async-mempool retain that alone moved measurements
~5x on reduction-heavy ticks) would otherwise fold harness changes into the
operator delta. With the overlay the harness sources are byte-identical on both
sides, so only the operator implementations differ — worth asserting explicitly,
because a silent violation looks exactly like an optimization.

In practice that means: check the baseline commit out into a git worktree,
`git checkout <current-ref> -- performance/` inside it, configure and build its
`perf_*` targets, then point `--before-build` at the result. A scripted version
of this lives on the `deustice/compare` branch, which is not part of this
tree.

`--checksum` additionally asserts both builds emit identical output bytes, which
is the correctness gate for a "pure speedup" claim. Re-point either side with
`--before-build` / `--after-build` to compare any two trees (e.g. two candidate
fixes).

### Measurement noise (read before believing a number)

On Thor the dominant variance is **between processes**, not between frames: two
launches of the *same* binary on the same scenario measured 0.367 vs 0.280 ms,
even though each launch already medians 90 frames internally. Raising
`--frames` does not tighten this; raising `--rounds` (more launches) does.

Two biases were found and fixed by running the script in **identity mode**
(`--before-build build --after-build build`, where the true delta is 0 by
construction):

* a fixed before-then-after order gave the second side a systematic **~10%**
  advantage (ramped clocks, warm page cache);
* ABBA-style alternation was *worse* (**~24%**) — it lets the middle side run
  back-to-back and inherit its own warm state.

So the script measures and discards one warm-up round, then reports the
**median** across scored rounds (not the min, which just rewards the luckiest
launch). Each row carries a `±noise %` — the launch-to-launch spread of the
noisier side — and any Δ inside that band is marked `_(noise)_`.

Residual noise floor at the defaults is roughly **±5%** per scenario, with some
(e.g. `cfar_CA_xl`) as high as **±18%**. Treat anything under ~10% as
unproven and re-run it with more `--rounds`. Re-validate the floor any time the
machine or settings change:

#### Why `--repeats` defaults to 1

`--repeats` pools more frames *inside a single launch*; `--rounds` adds more
launches. Since the variance that matters is between processes, in-process
repeats buy very little and cost a lot: `--repeats 3` triples the frame count
per scenario to shrink a component that was never the bottleneck. The same wall
time spent on `--rounds` samples the distribution that actually moves.

| setting | frames/scenario | launches/op | rel. cost | samples vs noise |
| --- | ---: | ---: | ---: | ---: |
| `--repeats 3 --rounds 3` | 95 | 8 | 1.00x | 3 |
| `--repeats 1 --rounds 5` (default) | 35 | 12 | **0.55x** | **5** |

So the default is both roughly half the wall time *and* a tighter band. Raise
`--frames`/`--repeats` only when you specifically care about within-launch
frame-to-frame jitter.

#### Runtime pre-estimate (the probe)

Before measuring, the script probes each operator once at `--warmup 1
--frames 2` and predicts its per-launch cost, so the plan line carries an
`estimated runtime` and the ETA is meaningful from the first launch instead of
after the first surprise. Disable with `--no-probe`.

The prediction is analytic, not an extrapolation of the probe's wall time. A
launch costs `Σ_scenarios(setup + (warmup + frames×repeats) × per_frame)`, and
the probe's own report already separates `setup_ms` from `median_ms`. Two
corrections were needed to make it usable:

* **Process overhead.** CUDA context init, graph construction and teardown
  never appear in the report and dominate cheap operators. Ignoring them
  under-predicted by ~4x. The probe measures its own wall time, subtracts what
  the report accounts for, and carries the remainder as a per-launch constant.
* **Cold `median_ms`.** The probe measures per-frame cost cold, so it runs
  high. A running `median(actual / predicted)` over completed operators
  calibrates the remaining estimates.

Measured against a full 16-operator run: 13 of 16 predictions within ±25%, and
the expensive operator correctly ranked first (73 s/launch predicted vs 41 s
for the runner-up) *before* it ran. Whole-run prediction was 32 min against
43 min actual — good enough for pacing, not a promise.

**Cost is ~4.7% of a cold run** (122 s of a 43 min sweep) and zero once the
cache is warm, since cached entries carry their own `launch_seconds`. Two
things keep it honest: only one side is probed (round 0 is discarded, so the
warming asymmetry is absorbed before anything is scored), and `--warmup 1` is
the floor — at `--warmup 0` a single cold frame measured 6.96 ms against a
0.25 ms steady state.

#### Cost is not uniform across operators

`perf_mimo_coherent_compensation` is the outlier: `--sizes` gives it **three**
scenarios per size (`tdm`, `ddm`, `ddm_nl`) where most operators get one, and
its cube is Tx x Rx x Doppler x Range, so `xl` is 537M cells — 16,384x its `xs`
and 8x its `lg`. One launch of it measured **374 s** against 1-9 s for every
other operator. If you only need a quick signal, exclude it
(`--operators` accepts an explicit list) and run it separately.

```bash
python3 performance/agent/compare_builds.py \
    --before-build build --after-build build --sizes sm,md,xl
# every row should read _(noise)_ and geomean should sit near 1.00x
```

## Reports (`report.md` + `report.html`)

Every run writes two reports next to the analysis:

* **`report.md`** — text findings + tables, meant to paste into an operator's
  `OPEN_ITEMS.md`.
* **`report.html`** — a self-contained page: the same findings + tables with the
  chart dashboard (kernel-vs-shell, latency, host-API composition, `compute()`
  NVTX phases, `[live]` override sweep, findings) embedded as a base64 PNG. Open
  it in any browser — no external files.

Charts need `matplotlib` (`pip install matplotlib` — *not* a harness
dependency: `runner`/`parse`/`diagnose` stay dependency-free, and `report.py`
degrades to an image-less `report.html` when it is absent). The charts read
plain JSON, so rendering runs on the host with no GPU. `visualize.py` is also
importable directly for ad-hoc plots or a cross-operator comparison:

```python
import sys; sys.path.insert(0, "performance/agent")
import visualize as viz
viz.dashboard("performance/agent/runs/fmcw_range_processor")
viz.compare_kernel_fraction([
    "performance/agent/runs/fmcw_range_processor",
    "performance/agent/runs/cfar_detector",
])
```

Artifacts (`.nsys-rep`, `.sqlite`, stat JSON, `analysis.json`, `diagnosis.json`,
`report.md`) land under `performance/agent/runs/<op>/` (gitignored) in the
mounted repo tree, so they persist on the host.

## How attribution works (and its limits)

* **Latency** always comes from the **clean** run; **attribution** always from
  the **profiled** run. nsys perturbs timing, so the two are never mixed.
* A `compute()` NVTX phase range bounds the kernel **enqueue**, not the async GPU
  execution. `parse.py` therefore joins the phases with the CUDA GPU timeline via
  `nvtx_gpu_proj_sum` (per-phase GPU-*projected* time), never treating NVTX
  wall-time as GPU time.
* Attribution is **per-size steady** (the steady path is identical across
  override params); per-param reconfiguration cost rides the clean
  `--sweep-overrides` `override_delta_ms`.
* nsys stat counts include the binary's **discarded process-warmup** session.
* Kernel time is GPU-side; CUDA-API time (`cudaMalloc`/`cudaFree`/sync) is
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
