---
name: perf-analyze
description: Analyze a holoscan-radar operator's runtime performance with Nsight Systems and write a ranked bottleneck report. Use when asked to profile, characterize, or find bottlenecks in a perf_<operator> (e.g. FMCWRangeProcessorOp, CFARDetectorOp) — sweeps small/medium/large sizes plus every [live] override param, attributes time to kernel vs cudaMalloc/memcpy/sync vs compute() NVTX phases, and emits report.md + report.html. Read-only: analyze + recommend, never edits code (applying fixes is the separate perf-optimize / perf-autotune skills).
---

# perf-analyze — operator performance analysis harness

Drive the deterministic substrate under `performance/agent/` (runner → parse →
diagnose → report) to profile one operator and produce a ranked, evidence-backed
perf report suitable to paste into that operator's `OPEN_ITEMS.md`.

This is the **read-only analysis tier**: profile, diagnose, report — do not edit
operator code, rebuild in a keep-if-faster loop, or change any locked contract.
Applying the fixes it recommends is a separate skill (`perf-optimize` — supervised
single-fix with a byte-equivalence gate; `perf-autotune` — autonomous loop over
this analysis). Surface findings; let a human decide.

## Prerequisites (assert first)

Everything runs **inside the dev container** (`nsys` + the CUDA toolchain; the
host has neither). The sanctioned entry is **`./holohub run-container`** — open
that shell and run every command below from it. `runner.py` preflight asserts
`nsys` and the built binary are present and fails with guidance if not.

Driving it non-interactively (an agent gets a fresh shell per command and can't
hold that session) — wrap each command in the image `./holohub` built. **Resolve
the image; do not pin the SDK-versioned tag** (e.g. `...:ngc-v4.3.0-cuda13` —
it drifts with the SDK):

```bash
IMG=$(docker images --filter=reference='holoscan-radar:*' \
        --format '{{.Repository}}:{{.Tag}}' | head -1)   # what ./holohub built
docker run --rm --runtime nvidia --gpus all --cap-add CAP_SYS_PTRACE --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/workspace/holoscan-radar -w /workspace/holoscan-radar \
  "$IMG" bash -lc '<command>'
```

## Steps

Let `OP` be the operator stem (the binary is `perf_<OP>`, e.g.
`fmcw_range_processor`). Run all of this **in-container**:

1. **Build** the perf suite with NVTX instrumentation on (once per checkout):

   ```bash
   cmake -S . -B build-perf -DCMAKE_CUDA_ARCHITECTURES=<cc> \
     -DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
     -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON \
     -DHOLOSCAN_RADAR_PERF_NVTX=ON -DBUILD_ALL=ON
   cmake --build build-perf --target perf_<OP> -j"$(nproc)"
   ```

   `<cc>` is the device compute capability (e.g. `110` for Thor/`native` when a
   GPU is present). `HOLOSCAN_RADAR_PERF_NVTX=ON` is what makes the compute()
   phase ranges (resolve-overrides / classify-validate / stage-kernel / emit)
   visible to nsys; a normal build leaves them as zero-overhead no-ops.

2. **Run the analysis pipeline** — one shot (runner → parse → diagnose →
   report):

   ```bash
   performance/agent/run.sh <OP> xs,md,xl
   ```

   `xs,md,xl` are small/medium/large representatives of the operator's 5-tier
   ladder (`xs,sm,md,lg,xl`) — enough to show the scaling trend (e.g. the
   host-bound → kernel-bound crossover) while keeping the run fast (each size is
   a clean run + an nsys-profiled run). Pass the full ladder for a finer scaling
   curve at ~1.7× the wall time; pass a single size for a quick check. (Some
   operators use extra tiers, e.g. CFAR's `xlw`; see the `perf-analyze` note on
   ladders or the operator's `perf_<op>.cpp`.)

   Prints the path to `report.md`. It chains the four stages below; run them
   individually when you need to inspect an intermediate or re-render without
   re-profiling:

   ```bash
   python3 performance/agent/runner.py   --operator <OP> \
       --build-dir build-perf --out-dir performance/agent/runs --sizes xs,md,xl
   python3 performance/agent/parse.py    --run-dir performance/agent/runs/<OP>
   python3 performance/agent/diagnose.py --run-dir performance/agent/runs/<OP>
   #   diagnose.py takes an optional --peak-gbps <device BW> for a roofline note
   python3 performance/agent/report.py   --run-dir performance/agent/runs/<OP>
   ```

   `runner.py` writes per-size `.nsys-rep` + stat-table JSON, the clean latency
   JSON, the clean `--sweep-overrides` JSON, and `runner_manifest.json`;
   `parse.py` folds them into `analysis.json`; `diagnose.py` emits ranked
   `diagnosis.json`; `report.py` renders both `report.md` (text, for pasting
   into `OPEN_ITEMS.md`) and a self-contained `report.html` (the same findings +
   tables plus the `visualize.py` chart dashboard embedded as a base64 PNG).

3. **Charts** are embedded in `report.html` automatically when `matplotlib` is
   installed (`pip install matplotlib`; charts read plain JSON, no GPU). Without
   matplotlib the pass still writes `report.html`, minus the dashboard image.
   `visualize.py` is also importable directly (`viz.dashboard(run_dir)`) for
   ad-hoc plots or a cross-operator comparison.

## Present

Relay `performance/agent/runs/<OP>/report.md` to the user: the ranked findings
(severity, evidence, recommended fix pattern) first, then the per-size
latency/attribution and the `[live]` override sweep. Call out the top 1–3
actionable findings in prose, and point them at `report.html` for the embedded
charts. Note that latencies are machine-specific and that GPU-kernel time and
host CUDA-API time overlap (they do not sum).

## Interpreting findings (the rubric this encodes)

- **cold ≫ steady** → first-tick plan/coefficient build or first-touch. Fine if
  cached (steady stays flat); a concern only if it recurs.
- **per-tick `cudaMalloc`/`cudaFree`** → the steady path allocates the output
  every tick. Pool + ping-pong a workspace keyed on input geometry.
- **host/graph-bound tick** (kernel is a small % of the tick) → attack the shell
  (alloc/launch/sync), not the stencil/FFT.
- **D2H on the steady path** → a host readback / O(map) scan; move it onto the
  GPU.
- **costly `[live]` override** (large `override_delta_ms`) → a plan/coefficient
  rebuild. Ensure warn-once-and-cache; callers must not sweep it every tick.

## Notes / limits

- Attribution is per-size steady (the steady path is identical across override
  params); per-param cost rides the clean override sweep's `override_delta_ms`.
- nsys stat counts include the binary's discarded process-warmup session.
- New operators need the NVTX ranges (`RADAR_NVTX_RANGE` / `nvtx::Phase` from
  `radar_common/sdk/nvtx.hpp`) and a `--sweep-overrides` matrix in their
  `perf_<op>.cpp` before this skill gives full per-phase + per-param detail;
  without them it still reports kernel/alloc/memcpy attribution and latency.
```
