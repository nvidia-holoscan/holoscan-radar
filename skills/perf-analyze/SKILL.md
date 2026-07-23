---
name: perf-analyze
description: Analyze a holoscan-radar operator's runtime performance with Nsight Systems and write a ranked bottleneck report. Use when asked to profile, characterize, or find bottlenecks in a perf_<operator> (e.g. FMCWRangeProcessorOp, CFARDetectorOp) — sweeps small/medium/large sizes plus every [live] override param, attributes time to kernel vs cudaMalloc/memcpy/sync vs compute() NVTX phases, emits report.md + report.html, and by default runs a scoped Nsight Compute (ncu) kernel deep-dive on the dominant kernel (memory-vs-compute, occupancy, measured roofline) — skipped only on user opt-out ("no ncu" / "nsys-only"), when ncu is unavailable, or when the tick is purely host-bound. Read-only: analyze + recommend, never edits code (applying fixes is the separate perf-optimize / perf-autotune skills).
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

`run-container` is interactive (no one-shot command passthrough), and the image
it uses is **branch-derived** (`holoscan-radar:<branch>`), so never pin a tag.
An agent gets a fresh shell per command and can't hold that session — so resolve
the image `./holohub` would use via `--dryrun` and `docker run` it headless
(build it once with `./holohub run-container` if it does not exist yet):

```bash
IMG=$(./holohub run-container --no-docker-build --dryrun 2>/dev/null \
        | grep -oE 'holoscan-radar[^[:space:]]*' | tail -1)   # holohub resolves it
# Fallback: --dryrun can print nothing (older holohub / no cached image). Then
# take the branch image if present, else any local holoscan-radar image:
[ -z "$IMG" ] && IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '^holoscan-radar:' | head -1)
docker run --rm --runtime nvidia --gpus all --cap-add CAP_SYS_PTRACE --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/workspace/holoscan-radar -w /workspace/holoscan-radar \
  "$IMG" bash -lc '<command>'
```

**Build/run wall time > the agent shell's 2-minute foreground cap.** A cold
CUDA build (and each nsys-profiled size) runs for minutes; a foreground call is
SIGTERM'd at the timeout mid-build. Launch the build and the analysis pipeline
as **background** commands (write logs to the mounted tree; poll the log / binary
mtime), or the container is killed and you rebuild from scratch. Keep `-j4`
(not `-j$(nproc)` — heavy parallelism truncates captured stdout on this host).

**The dev image is the dep sandbox (Dockerfile-provisioned).** `./holohub
run-container` builds the dev image from the repo `Dockerfile` (the wrapper points
holoscan-cli at it via `HOLOSCAN_CLI_DEFAULT_DOCKERFILE`), and both the
interactive shell **and** the agent's headless `docker run` target that same
`holoscan-radar:<branch>` tag. So the Dockerfile **is** the skills' dependency
sandbox — add a dep there, rebuild, and every entry point has it, reproducibly and
without the staleness of a `docker commit`:
- **`matplotlib`** (for `report.html` charts) is baked in unconditionally.
- **Nsight Compute (`ncu`)** is an opt-in layer behind the `INSTALL_NCU` build-arg
  (OFF by default, ~700 MB). Bake it into the sandbox once:
  ```bash
  ./holohub run-container --build-args '--build-arg INSTALL_NCU=1'
  ```
  That rebuilds `holoscan-radar:<branch>` with `ncu` present; a plain rebuild
  leaves it out.

For an **interactive root shell** to poke at the image, prefer holohub's native
flags over hand-rolling docker: `--as-root` enters as root (the default mapped
user is your non-root host UID with no working sudo, which is why plain
`apt-get`/`sudo` fail in a normal session), and `--docker-opts='…'` forwards
arbitrary `docker run` args, e.g. `--docker-opts='--cap-add SYS_ADMIN'`.

Provisioning `ncu` is necessary but **not sufficient**: its counters need root
**and** `SYS_ADMIN` *at run time* — see [Deep kernel dive → Availability
gate](#deep-kernel-dive--nsight-compute-ncu). Short version: build once with
`INSTALL_NCU=1`, then in the agent's headless `docker run` add **both**
`--cap-add SYS_ADMIN` (next to `CAP_SYS_PTRACE`) **and** `--user 0:0` whenever a
command invokes `ncu`. On this host the non-root mapped user fails with
`ERR_NVGPUCTRPERM` even *with* `SYS_ADMIN`; root (`--user 0:0`) + `SYS_ADMIN` is
what makes counters collect. **Verified working on this Thor host.**

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
   performance/agent/run.sh <OP>            # full 5-tier ladder (default)
   performance/agent/run.sh <OP> xs,md,xl   # a fast 3-size subset
   ```

   The default now sweeps the **full 5-tier ladder** (`xs,sm,md,lg,xl`) for a
   complete scaling curve; pass a subset (e.g. `xs,md,xl`) or a single size to go
   faster. (Some operators use extra tiers, e.g. CFAR's `xlw`; see the operator's
   `perf_<op>.cpp`.)

   Two things the run does automatically:
   - **Primary-axis (variant) breakdown.** If the operator has a primary
     categorical mode (clutter `method`, range `window`, cfar `variant`, …) the
     clean latency runs sweep it (`--sweep-variants`), so the report breaks
     runtime out **per variant × size** — not just the default mode. Small tuning
     knobs (e.g. range's `fft_norm`) stay on the separate one-at-a-time
     `[live]` override sweep, avoiding a combinatoric blow-up. Size-only
     operators just report the single mode.
   - **Adaptive trials + variance.** Each size is timed once, then re-run pooled
     over an **adaptive repeat count** (aim ~10, scaled by op speed under a
     wall-time budget — fast ops get more, slow ops fewer) so latency stats carry
     inter-run variance. The report shows `N` (= frames × repeats) and a
     `min–max` spread, and the charts draw min–max whiskers. Tune with
     `--target-repeats` / `--repeat-budget-s` / `--repeats N` (fixed).

   Prints the path to `report.md`. It chains the four stages below; run them
   individually when you need to inspect an intermediate or re-render without
   re-profiling:

   ```bash
   python3 performance/agent/runner.py   --operator <OP> \
       --build-dir build-perf --out-dir performance/agent/runs --sizes xs,sm,md,lg,xl
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

3. **Charts** are embedded in `report.html` automatically — `matplotlib` is baked
   into the dev image (Dockerfile), so the dashboard renders by default (charts
   read plain JSON, no GPU). On an older image built before that, the pass still
   writes `report.html` minus the dashboard; `./holohub run-container` rebuilds to
   pick it up. `visualize.py` is also importable directly (`viz.dashboard(run_dir)`)
   for ad-hoc plots or a cross-operator comparison.

4. **Deep kernel dive — on by default.** Unless the user opted out or `ncu` is
   unavailable, run the scoped Nsight Compute dive (see
   [Deep kernel dive](#deep-kernel-dive--nsight-compute-ncu)) on the dominant
   steady kernel of size `md` and fold its Speed-of-Light / occupancy / measured
   roofline into what you relay. If you skip it, state which of the three skip
   reasons applied.

## Present

Relay `performance/agent/runs/<OP>/report.md` to the user: the ranked findings
(severity, evidence, recommended fix pattern) first, then the per-size
latency/attribution and the `[live]` override sweep. Call out the top 1–3
actionable findings in prose, and point them at `report.html` for the embedded
charts. Note that latencies are machine-specific and that GPU-kernel time and
host CUDA-API time overlap (they do not sum).

**Steady medians carry real run-to-run variance** (observed ~2–3× on Thor between
otherwise-identical runs — thermal/clock/scheduling). Don't over-read a single
run's absolute number or a small cross-size wobble; trust the *shape* (scaling
trend, kernel-fraction, alloc/tick) over any one median. When a finding hinges on
a latency delta, say it's approximate.

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
  **This is advisory, not a steady-path optimization target.** It ranks high by
  `override_delta_ms`, but when the rebuild is already cached (re-selecting a
  prior config is free) there is no code change to make — it does not touch the
  steady tick. Flag it for the operator's docs/callers; don't hand it to
  `perf-optimize`/`perf-autotune` as a fix. (Known rubric wart: these outrank the
  genuinely-actionable steady-path findings — alloc/tick, host-bound, D2H — by
  severity; read past them to the top steady-path item.)

## Deep kernel dive — Nsight Compute (`ncu`)

**On by default; one level below nsys.** nsys tells you *where the tick's wall-time
goes* (kernel vs `cudaMalloc`/memcpy/sync vs host/graph). `ncu` answers the next
question: *is the dominant kernel near the hardware ceiling, and if not, why?* —
Speed-of-Light (memory-bound vs compute-bound), achieved occupancy, L1/L2
locality, warp-stall reasons, and a **measured** roofline (which supersedes
`diagnose.py`'s estimated `--peak-gbps` note). Use it to decide whether a kernel
rewrite is even worth it, or whether the win is in the shell (alloc/launch/D2H)
instead. Still read-only — this tier analyzes and recommends.

**Run it by default** on the *one* dominant steady kernel a size's attribution
points at (default size `md`). There are exactly three reasons to skip it, and
you must say which one applies:
1. **User opt-out** — the user asked to skip it (e.g. "no ncu", "skip Nsight
   Compute", "nsys-only", "don't run ncu"). Their request is sticky for the
   session; honor it without re-asking.
2. **Unavailable *after trying to provision it*** — `ncu` absent is **not** a skip
   reason by itself: the image just wasn't built with `INSTALL_NCU=1` (see the
   availability gate below). Rebuild the sandbox with it (or do the one-off
   in-session install) first. Only fall back to nsys-only (and say so) if it
   genuinely can't be provisioned or the perf counters stay locked
   (`ERR_NVGPUCTRPERM`) after adding `SYS_ADMIN` + root.
3. **No kernel worth profiling** — the tick is so host/graph-bound that no kernel
   meaningfully contributes (kernel ≪ the tick at every size). Note that the
   attribution already answered the question (the win is in the shell), so the
   dive would only confirm a negligible kernel.

`runner.py` does not yet drive it (a future stage may), so it stays a **manual**
invocation — but a default step of the analysis, not an occasional escalation.

**Availability gate — the image should already have `ncu`.** `ncu` is **not in the
stock base image**, but the repo Dockerfile installs it behind the `INSTALL_NCU`
build-arg, so the sanctioned way to have it is to **build the dev sandbox with
`INSTALL_NCU=1`** (see the dep-sandbox note in Prerequisites) — reproducible,
survives `--rm`, no `docker commit`. **Verified on this Thor host (2026-07-23).**

- **Check it's there:** `NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null |
  head -1)`. The CLI installs under `/opt/nvidia/nsight-compute/<ver>/ncu` and is
  **not on `PATH`**, so always resolve it this way. Empty `$NCU` ⇒ the current
  image wasn't built with `INSTALL_NCU=1`.
- **If it's missing, provision it:** rebuild the sandbox —
  `./holohub run-container --build-args '--build-arg INSTALL_NCU=1'` — then re-run.
  *One-off alternative (no rebuild):* as root in the same container, `apt-get
  update && apt-get install -y --no-install-recommends nsight-compute`. **`apt-get
  update` FIRST** — the base lists the NVIDIA repos but hasn't fetched them, so a
  bare `apt-cache search`/`install` finds nothing and wrongly reads "unavailable."
  This install is thrown away on `--rm`, so prefer the build-arg.
- **Package:** `nsight-compute` (standalone devtools build; candidate `2026.2.1`,
  matches the image's nsys era). `cuda-nsight-compute-13-2` also works
  (CUDA-toolkit-pinned).
- **Counters need `--cap-add SYS_ADMIN` *and* root (`--user 0:0`).** The harness's
  headless `docker run` only adds `CAP_SYS_PTRACE` and maps a non-root UID; for
  `ncu` add `--cap-add SYS_ADMIN` **and** `--user 0:0` (interactive:
  `--docker-opts='--cap-add SYS_ADMIN'`). **Verified on this Thor host:** root +
  `SYS_ADMIN` collects real Speed-of-Light numbers; the *non-root* mapped user
  fails with `ERR_NVGPUCTRPERM` **even with `SYS_ADMIN`**, so both are required
  here. (Unlike profiling, plain nsys and the perf binary run fine as non-root —
  only `ncu` counters need root.)
- **Probe once:** `"$NCU" --version`, then a single-metric run. Only if it still
  prints `ERR_NVGPUCTRPERM` after adding **both** `SYS_ADMIN` and `--user 0:0`
  should you **stay nsys-only and say so** — never fabricate kernel metrics.

> **NVTX filter gotcha:** `--nvtx-include "stage-kernel/"` can silently match
> **zero** launches (ncu exits 0 but profiles nothing). Rely on
> `--launch-skip/--launch-count` for scoping; treat `--nvtx-include` as optional
> and confirm the run actually reported kernels before trusting an empty result.

**Scope it hard or it never finishes.** Kernel replay makes `ncu` 10–100× slower
*per launch*, so never run it across the size×variant×repeat matrix. Profile
**one size** (default `md`), the **primary variant only**, a **curated section
set**, and only a handful of **steady-tick** launches (skip warmup):

```bash
# Run as root with SYS_ADMIN against the ncu-enabled sandbox (built INSTALL_NCU=1):
#   docker run ... --cap-add SYS_ADMIN --user 0:0 holoscan-radar:<branch> ...
NCU=$(ls /opt/nvidia/nsight-compute/*/ncu | head -1)
"$NCU" --target-processes all \
    --section SpeedOfLight --section Occupancy \
    --section LaunchStats --section MemoryWorkloadAnalysis \
    --launch-skip 20 --launch-count 3 \
    --export performance/agent/runs/<OP>/<OP>_md.ncu-rep --force-overwrite \
    ./build-perf/performance/cpp/perf_<OP> --sizes md --frames 30 --quiet
```

`--launch-skip/--launch-count` restrict capture to a few steady kernels (not cold
first-touch); `--set full` and multi-size are exactly what turn a 5-second run
into an hour. (Add `--nvtx --nvtx-include "…/"` only if you confirm it matches
launches — see the gotcha above.) Read it back with `"$NCU" --import
<…>.ncu-rep --page details` (or `--csv --page raw` to parse), and map to the
rubric:

- **DRAM% ≈ peak, compute% low** → memory-bound *and* near ceiling: the kernel has
  little headroom; wins come from the shell or from moving less data, not the
  stencil/FFT.
- **compute% ≈ peak** → compute-bound: needs an algorithmic change, not a shell
  fix.
- **both low + low occupancy / high stalls** → launch-config or latency problem
  (block size, registers, shared mem); read LaunchStats + the stall-reason
  breakdown.
- The **measured** roofline point supersedes the estimated `--peak-gbps` note.

**Cleanup.** `ncu` writes a `.ncu-rep` (and can drop `report*.ncu-rep` in cwd);
keep it in the run dir, don't commit it, and leave the tree as you found it —
same discipline as the temporary instrumentation below.

## Temporary instrumentation (must be removed before finishing)

This skill is read-only for *permanent* code, but to sharpen an ambiguous
finding you MAY add **temporary** analysis code — extra NVTX ranges
(`RADAR_NVTX_PHASE` / `RADAR_NVTX_RANGE` from `radar_common/sdk/nvtx.hpp`),
scratch timers, debug prints. Mark each with a `// PERF-SCRATCH` comment and
**remove all of it before this analysis pass ends** — leave the tree exactly as
you found it. Verify with `grep -rn "PERF-SCRATCH" operators/ performance/`
(must return nothing) and rebuild.

## Notes / limits

- Attribution is per-size steady (the steady path is identical across override
  params); per-param cost rides the clean override sweep's `override_delta_ms`.
- A hotspot can live in code **shared across operators** (e.g. `range_stage.hpp`
  is used by both the range and range-doppler processors). That finding is still
  in scope for `perf-optimize` — but name the sharing operators (reverse-include
  across `operators/`) so the optimizer knows the full byte-equivalence gate set
  (every consumer must re-pass its checksum). A hotspot in `radar_common/**` or
  otherwise outside `operators/` is **surface-only** — optimizing it would need a
  gate wider than the operator set.
- nsys stat counts include the binary's discarded process-warmup session.
- New operators need the NVTX ranges (`RADAR_NVTX_RANGE` / `nvtx::Phase` from
  `radar_common/sdk/nvtx.hpp`) and a `--sweep-overrides` matrix in their
  `perf_<op>.cpp` before this skill gives full per-phase + per-param detail;
  without them it still reports kernel/alloc/memcpy attribution and latency.
```
