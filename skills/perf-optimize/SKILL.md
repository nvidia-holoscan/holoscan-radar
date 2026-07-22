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
2. **Byte-equivalence** — the operator's output on the harness's deterministic
   input is **bit-identical** before and after (`perf_<op> --checksum`). This is
   the default gate: a pure perf refactor (pooling, gating, caching, removing
   redundant work) must not change a single output byte.
3. **Speed** — the steady median (`perf-analyze`) improves beyond run-to-run
   noise. A change that passes 1+2 but isn't measurably faster is not worth
   landing.

> A change that *intends* to alter numerics (e.g. reordering a floating-point
> reduction) will fail gate 2 by design. That requires **explicit opt-in** from
> the user AND re-validation against the analytic-simulator tests — do not
> silently waive byte-equivalence.

## Hard guardrails

- **Never change the locked contract** (`OPERATOR_API.md`: ports, shapes, dtype
  pairs, parameter names/defaults, `[live]` tags). Performance refactors only.
- **Never touch the signal math** on a byte-equivalence change — pooling/gating/
  caching only.
- **Correctness and byte-equivalence are gates, not suggestions.** Fail either →
  revert, report why, stop.
- **Do not land it yourself.** End by presenting the diff + numbers; a human
  approves before it is committed to the feature branch / pushed.

## Prerequisites

Run inside `./holohub run-container` (see `perf-analyze` for the raw `docker run`
form, the image-resolution fallback, and — important — the rule that CUDA builds
and nsys runs exceed the agent shell's 2-minute foreground cap, so **background
every build/analysis command** or the container is SIGTERM'd mid-build). The
correctness gate needs unit tests, so build with **testing ON** too:

```bash
cmake -S . -B build-opt -DCMAKE_CUDA_ARCHITECTURES=<cc> \
  -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
  -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON \
  -DHOLOSCAN_RADAR_PERF_NVTX=ON -DBUILD_ALL=ON
```

The operator's `perf_<op>.cpp` must wire `PerfChecksumSinkOp` (a `--checksum`
line) for gate 2 — currently done for `fmcw_range_processor`; other operators
need it wired first (one fan-out `add_flow`, mirror the range op).

## Steps

Let `OP` be the operator stem (`fmcw_range_processor`) and `BUILD=build-opt`.

1. **Baseline.** Capture the numbers the gates compare against, and **save the
   baseline analysis** (the after-run overwrites it):
   ```bash
   cmake --build $BUILD --target perf_<OP> -j4                   # background this
   performance/agent/run.sh <OP> xs,md,xl        # perf-analyze: the speed baseline
   cp performance/agent/runs/<OP>/analysis.json \
      performance/agent/runs/<OP>/baseline_analysis.json
   ./$BUILD/performance/cpp/perf_<OP> --sizes md --checksum --warmup 0 \
       --frames 2 --quiet 2>/dev/null | grep output_checksum   # byte baseline
   # Discover the actual unit-test name — ctest -R matches the *gtest suite*
   # (e.g. FMCWRangeProcessorOpTest), NOT the CMake target (fmcw_range_processor):
   ctest --test-dir $BUILD -N -L unit | grep -i <OP-keyword>
   ctest --test-dir $BUILD -L unit -R <SuiteName>                # must be green now
   ```

   **Record a *stable* baseline median now** — you cannot cheaply re-measure it
   later (step 4 rebuilds over the baseline binary). The harness median has
   ~2–3× run-to-run variance on this host, so take the median of a few clean runs
   per size (`--warmup 8 --frames 40`, repeated 3×), not one shot. Note the real
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
   recommended pattern (e.g. pool the dense output + ping-pong a workspace keyed
   on the input geometry). Match surrounding style; do not touch the contract or
   the math.

4. **Rebuild** the operator + perf binary + its test:
   ```bash
   cmake --build $BUILD --target perf_<OP> <OP-test-target> -j4
   ```

5. **Gate 1 — correctness:** `ctest --test-dir $BUILD -L unit -R <OP-pattern>
   --output-on-failure`. Any failure → revert (step 7), report, stop.

6. **Gate 2 — byte-equivalence:** re-run the `--checksum` command from step 1 at
   a couple of sizes; the hash **must match the baseline** exactly. Mismatch →
   the change altered output → revert (unless the user opted into a
   numerics-changing change, which then requires the sim tests to re-pass).

7. **Gate 3 — speed:** re-run `perf-analyze`; compare the steady median to the
   baseline. Require an improvement clearly beyond noise — with ~2–3× run-to-run
   variance, one run is not enough: confirm the win with a few repeated clean runs
   per size (as in step 1). **Be skeptical both ways** — reject a close call as
   noise, but also sanity-check an implausibly *large* delta against the
   mechanism before believing it (in the first real run a −84% xl delta was real
   — a 32 MB per-tick `cudaFree` removed from a host-bound tick — but only trust
   that after repeats confirm it and it matches the change). No real gain →
   revert.

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
   (tests green, identical checksum, before→after median), and
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

## Notes / limits

- One fix per invocation by design. To chain fixes, run again, or use
  `perf-autotune` for the autonomous loop.
- Gate 2 (`--checksum`) needs `PerfChecksumSinkOp` wired in the operator's
  `perf_<op>.cpp`; gate 1 needs the operator to have a `-L unit` test target.
- Byte-equivalence is proven on the harness's fixed input; the unit tests cover
  the broader correctness surface. Together they are the keep/revert decision.
- The speed number is machine-specific and noisy at the sub-tick scale — trust a
  clear multi-run improvement, not a single close call.
