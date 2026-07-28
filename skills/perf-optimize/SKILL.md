---
name: perf-optimize
description: Apply ONE performance fix to a holoscan-radar operator and prove it is faster AND byte-equivalent before proposing it to land. Use when asked to optimize / speed up / pool buffers in / apply a fix to a perf_<operator> (e.g. FMCWRangeProcessorOp) — takes a specific behavior-preserving change (usually a perf-analyze finding), applies it in an isolated checkpoint, gates it on unit tests (correctness) + an output checksum (byte-equivalence) + the harness (speed), and keeps or reverts it. Supervised single fix; stops for human approval before landing. NOT an autonomous loop (that is perf-autotune).
---

# perf-optimize — supervised single-fix operator optimizer

Apply one **behavior-preserving** performance change to an operator and verify it
against three gates before asking a human to land it. This is the middle tier of
the trio: [`perf-analyze`](../perf-analyze/SKILL.md) finds bottlenecks (read-only);
this applies **one** fix under verification; [`perf-autotune`](../perf-autotune/SKILL.md)
is the autonomous loop over this.

**Scope of one invocation: exactly one candidate fix.** Pick it from a
`perf-analyze` finding (or take it from the user). Never batch changes; never
loop unattended here.

## The three gates (all must pass to keep a change)

1. **Correctness** — the operator's unit tests (`ctest -L unit`). Non-negotiable.
2. **Byte-equivalence** — the output on the harness's deterministic input is
   **bit-identical** before and after (`perf_<op> --checksum`), for the operator
   under test **and every other operator that shares a file the change touched**
   (see the shared-code gate in Hard guardrails). A pure perf refactor (pooling,
   gating, caching, kernel/launch tuning, removing redundant work) must not change
   a single output byte for any of them.
3. **Speed** — the steady median improves beyond run-to-run noise, measured
   with `compare_builds.py` (interleaved A/B; see Steps for the invocation and
   why `--frames` matters). A change that passes 1+2 but isn't measurably
   faster is not worth landing.

> **Near the noise floor, pair the timing with a structural measurement.** A
> deterministic count from the profile — a memcpy row that goes 15 calls → 0,
> an alloc count that drops by one per tick — proves the change did what it
> claims, independent of the timer, and distinguishes "too small to measure"
> from "didn't happen". Timing and mechanism agreeing is what makes a small
> delta trustworthy:
> `nsys stats --report cuda_gpu_mem_time_sum --format csv <rep>.nsys-rep`

> A change that *intends* to alter numerics (e.g. reordering a floating-point
> reduction) will fail gate 2 by design. That requires **explicit opt-in** from
> the user AND re-validation against the analytic-simulator tests — do not
> silently waive byte-equivalence.

## Hard guardrails

- **Never change the locked contract** (`OPERATOR_API.md`: ports, shapes, dtype
  pairs, parameter names/defaults, `[live]` tags). Performance refactors only.
- **Scope = anything under `operators/**`.** Fair game: the operator shell **and**
  its compute internals (kernels, staging, DSP), including headers **shared across
  operators** (e.g. `range_stage.hpp`, shared by the range and range-doppler
  processors). Off-limits: `radar_common/**`, generated code, and anything outside
  `operators/` — a hotspot there is a finding to **surface**, not a file to edit
  (optimizing it needs a gate wider than the operator set).
- **Touch shared code → gate every operator that uses it.** The checksum only
  protects operators you actually run it on, so if a change touches a file used by
  more than the operator-under-test, **every** consuming operator must pass
  correctness + byte-equivalence — not just the one you're tuning. Get that
  consumer set by reverse-include across `operators/` (see Steps); each consumer
  needs a `--checksum` sink + unit tests, else wire them first or the change is out
  of scope.
- **Numerics still need opt-in.** Byte-equivalence is the hard gate for *every*
  affected operator; changing the math (reordered FP reduction, normalization,
  precision) needs explicit user opt-in AND analytic-sim re-validation for each
  affected operator. A byte-preserving change to compute (launch config, fusion,
  redundant-work removal) does not — the checksum proves it.
- **Correctness and byte-equivalence are gates, not suggestions.** Fail either on
  **any** affected operator → revert, report why, stop.
- **Do not land it yourself.** End by presenting the diff + numbers; a human
  approves before it is committed to the feature branch / pushed.

## Prerequisites

Run inside `./holohub run-container` (see `perf-analyze` for the raw `docker run`
form, the native `--as-root` / `--docker-opts` entry, the image-resolution
fallback, and — important — the rule that CUDA builds and nsys runs exceed the
agent shell's 2-minute foreground cap, so **background every build/analysis
command** or the container is SIGTERM'd mid-build). The correctness gate needs
unit tests, so build with **testing ON** too:

```bash
cmake -S . -B build-opt -DCMAKE_CUDA_ARCHITECTURES=<cc> \
  -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
  -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON \
  -DHOLOSCAN_RADAR_PERF_NVTX=ON -DBUILD_ALL=ON
```

The operator's `perf_<op>.cpp` must wire `PerfChecksumSinkOp` (a `--checksum`
line) for gate 2 — other operators need it wired first (one fan-out `add_flow`,
mirror an operator that has it).

**Audit *which ports* it covers — "a sink exists" is the wrong check.** A sink
wired to one output while the rest go to `PerfNullSinkOp` passes a
does-it-exist check while being blind to most of the operator. Before trusting
gate 2, confirm the checksummed set spans the **blast radius of your change**:
if you are touching compute, the outputs carrying that arithmetic must be
covered. (CFAR shipped with a sink on `detection_idx` alone — which encodes
*which* cells fired but no per-cell value — so a kernel numerics change would
have passed unchecked until `detection_score` / `noise_estimate` were added.)

Coverage can be transitive, so you do not always need a sink per port: an
index output that is exactly the set cells of a mask, in a defined order,
covers that mask. Prefer that reasoning to blanket fan-out — some dtypes do not
round-trip a `TensorMap` receive (bool does not on HSDK 4.3). When you do wire
several sinks, make sure the emitted lines are distinguishable: same-shaped
outputs have identical byte counts, so label them (the sink appends its
operator name).

## Steps

Let `OP` be the operator stem (`fmcw_range_processor`) and `BUILD=build-opt`.

1. **Baseline.** Capture the numbers the gates compare against, and **save the
   baseline analysis** (the after-run overwrites it):
   ```bash
   cmake --build $BUILD --target perf_<OP> -j4                   # background this
   performance/agent/run.sh <OP>                 # perf-analyze speed baseline (full 5-tier
                                                 # ladder by default; pass e.g. xs,md,xl to
                                                 # speed up the rebuild loop if needed)
   cp performance/agent/runs/<OP>/analysis.json \
      performance/agent/runs/<OP>/baseline_analysis.json
   ./$BUILD/performance/cpp/perf_<OP> --sizes md --checksum --warmup 0 \
       --frames 2 --quiet 2>/dev/null | grep output_checksum   # byte baseline
   # Discover the actual unit-test name — ctest -R matches the *gtest suite*
   # (e.g. FMCWRangeProcessorOpTest), NOT the CMake target (fmcw_range_processor):
   ctest --test-dir $BUILD -N -L unit | grep -i <OP-keyword>
   ctest --test-dir $BUILD -L unit -R <SuiteName>                # must be green now
   ```

   **Snapshot the baseline *binary* now** — step 4 rebuilds over it, and gate 3
   needs both sides present at once to interleave them:
   ```bash
   mkdir -p .gate/before/performance/cpp
   cp $BUILD/performance/cpp/perf_<OP> .gate/before/performance/cpp/
   ```
   (Any directory mirroring `<dir>/performance/cpp/perf_<OP>` works —
   `compare_builds.py` takes build *dirs*, not binary paths.) Note the real
   `median_ms` is the first non-null `results[]` entry (a single-size clean/
   checksum run can lead with a null-median calibration entry).

2. **Checkpoint (isolation).** For a supervised single fix, work in place and
   revert with git — it keeps incremental rebuilds fast (touching one `.cu`
   recompiles that TU only):
   ```bash
   git stash list; git status         # ensure a clean tree first
   # ... you will `git checkout -- <files>` to revert if a gate fails ...
   ```
   (The autonomous `perf-autotune` tier uses a `git worktree` per candidate for
   parallel/unattended isolation; for one supervised change, in-place is faster.)

3. **Apply the fix.** Make the smallest change that implements the finding's
   recommended pattern (pool + ping-pong a workspace keyed on input geometry; or,
   in a compute internal, a byte-preserving launch/fusion/redundant-work change).
   Match surrounding style; stay within `operators/**` and don't touch the
   contract. **Then compute the affected-operator set** — every operator that uses
   a file you changed, because they all get gated:
   ```bash
   git diff --name-only                                  # what you touched
   # ops whose dir you edited directly, PLUS any op that #includes a changed
   # header (chase a hop or two — when unsure, gate MORE operators, not fewer):
   { git diff --name-only | sed -n 's#^\(operators/[^/]*/[^/]*_op\)/.*#\1#p';
     for h in $(git diff --name-only | xargs -r -n1 basename | grep -E '\.(hpp|cuh)$'); do
       grep -rl "$h" operators/ --include=*.cu --include=*.hpp --include=*.cuh; done \
       | sed -n 's#^\(operators/[^/]*/[^/]*_op\)/.*#\1#p'; } | sort -u
   ```
   If this lists more than the operator-under-test (touching `range_stage.hpp`
   lists **both** range and range-doppler), every extra operator is a **mandatory**
   gate in steps 5–6 — and you must have a byte baseline (step 1) for each. If any
   affected operator lacks a `--checksum` sink or unit tests, wire them first or
   abandon the change.

4. **Rebuild** the perf binary + unit test for **every affected operator** (not
   just the one you're tuning):
   ```bash
   cmake --build $BUILD --target perf_<OP> <OP-test-target> \
       perf_<affected> <affected-test-target> … -j4
   ```

5. **Gate 1 — correctness:** run the unit tests for **every affected operator**
   (`ctest --test-dir $BUILD -L unit -R '<OP|affected|…>' --output-on-failure`).
   Any failure on any of them → revert (step 7), report, stop.

6. **Gate 2 — byte-equivalence:** re-run the `--checksum` command from step 1 at a
   couple of sizes **for every affected operator**; each hash **must match that
   operator's baseline** exactly. Any mismatch → the change altered that operator's
   output → revert (unless the user opted into a numerics-changing change, which
   then requires the sim tests to re-pass for **each** affected operator).

7. **Gate 3 — speed:** use `compare_builds.py`, **not** hand-rolled repeated
   runs. It interleaves before/after launches, pins CPUs (`taskset`), discards
   warm-up rounds, gates on GPU temperature, and scores each scenario against
   its own noise band — all of which a sequential "measure baseline, then
   measure candidate" loop gets wrong:
   ```bash
   python3 performance/agent/compare_builds.py \
     --before-build .gate/before --after-build $BUILD \
     --operators <OP> --sizes <ladder> \
     --frames 200 --rounds 15 --warmup-rounds 3 --checksum --no-cache \
     --out-dir performance/agent/runs/<OP>/compare_<n>
   ```
   Keep only if `sig` counts real wins (a scenario whose Δ cleared its own
   `±noise` band). `--checksum` makes this cover gate 2 as well — read
   `checksum_match` from the emitted `compare.json`; do **not** parse the
   markdown, whose trailing `⚠ outputs differ between builds` is an
   unconditional *legend* line, not a warning.

   **`--frames 200` is load-bearing.** The perf harnesses count only the
   pre-override window as steady state, so usable samples per launch are
   ~`frames/2`; at the tool's default `--frames 30` the large tiers show noise
   bands of 30–55% and nothing under ~10% can be proven. Raising frames
   collapses those bands to a few percent. **Raising `--rounds` does not fix
   this** — `±noise` is the launch-to-launch IQR, not the standard error of the
   median, so it does not shrink with rounds (measured: 7 → 25 rounds made the
   bands slightly *worse*). This overrides `--rounds`' own help text.

   **Be skeptical both ways** — reject a close call as noise, but also
   sanity-check an implausibly *large* delta against the mechanism before
   believing it. No real gain → revert.

   **Revert** (any gate fails): `git checkout -- <changed files>` and rebuild, so
   the tree is back to baseline.

8. **Report + present.** If all three gates pass, remove any temporary
   instrumentation (see below), write a `session.json` recording the change, and
   render the optimization report:
   ```bash
   # session.json: {"operator","mode":"optimize","changes":[{finding,summary,
   #   files,diffstat,gates:{correctness,byte_equivalent,speed_delta_pct},kept}],
   #   "notes"}  — see optimize_report.py's docstring for the full schema.
   python3 performance/agent/optimize_report.py \
     --baseline performance/agent/runs/<OP>/baseline_analysis.json \
     --final    performance/agent/runs/<OP>/analysis.json \
     --session  session.json --out-dir performance/agent/runs/<OP>
   ```
   Present to the human: the diff (tie it to the finding), the gate results
   (tests green + identical checksum **for every affected operator**, before→after
   median), the affected-operator set if the change touched shared code, and
   `optimization_report.{md,html}` (before/after per size + chart).

   **Stop there.** Do not commit to the feature branch or push until the human
   says to land it.

## Temporary instrumentation (must be removed)

To get finer attribution you MAY add temporary analysis code — extra NVTX ranges
(`RADAR_NVTX_PHASE` / `RADAR_NVTX_RANGE` from `radar_common/sdk/nvtx.hpp`),
scratch timers, or debug prints. Mark every such edit with a `// PERF-SCRATCH`
comment so it is greppable, and **remove all of it before the loop ends** — it is
analysis scaffolding, never part of the fix. This is a single-fix loop, so "the
loop" is this one pass: before rendering the report / presenting, assert the tree
is clean of it and rebuild so the kept state is only the real change:

```bash
grep -rn "PERF-SCRATCH" operators/ performance/   # must return nothing
```

**Scratch logs.** Write backgrounded build/gate logs to
`performance/agent/runs/<OP>/_scratch/`, never loose in `runs/`. Remove `_scratch/`
on a clean finish (after all gates pass and the report is rendered); on a failure
or abort, keep it and report the path. Durable outputs (report, `analysis.json`,
`session.json`, `.nsys-rep`/`.ncu-rep`) stay in the operator subdir.

## Notes / limits

- One fix per invocation by design. To chain fixes, run again, or use
  `perf-autotune` for the autonomous loop.
- Gate 2 (`--checksum`) needs `PerfChecksumSinkOp` wired in the operator's
  `perf_<op>.cpp`; gate 1 needs the operator to have a `-L unit` test target.
- Byte-equivalence is proven on the harness's fixed input; the unit tests cover
  the broader correctness surface. Together they are the keep/revert decision.
- The speed number is machine-specific and noisy at the sub-tick scale — trust a
  clear multi-run improvement, not a single close call.
- **`ncu` is evidence, not a gate — and on by default.** To understand *why* a
  change did or didn't help at the kernel level — memory- vs compute-bound,
  occupancy, roofline — run the Nsight Compute deep-dive from `perf-analyze`
  **by default** once the gates have decided, on the kernel the change touched.
  `ncu` absent is **not** a skip reason: install it and profile as root with
  `SYS_ADMIN` per `perf-analyze`'s **Availability gate** (the single source for the
  install / container / command recipe). Skip it only when the user opted out
  (e.g. "no ncu" / "nsys-only" — sticky for the session), when `ncu` genuinely
  can't be installed / its counters stay locked (stay nsys-only and say so), or
  when the change is purely host-side (pooling/gating/caching) so no kernel
  changed — then say the dive was uninformative and why. It only ever *explains* a result; it never changes the
  keep/revert decision — the three gates above are the sole arbiters.
