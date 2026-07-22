---
name: perf-autotune
description: Autonomously optimize a holoscan-radar operator — loop analyze → apply a fix → re-measure → keep-if-faster-and-byte-equivalent, over multiple bottlenecks, until a stop condition, then present a cumulative optimization report for approval. Use ONLY when the user explicitly asks to autonomously optimize / auto-tune / iteratively speed up a perf_<operator> and accepts multiple unattended code changes. Isolated in a git worktree; every change is gated (unit tests + output checksum + harness); nothing lands without human approval. This edits code in a loop — do not invoke unless explicitly requested.
---

# perf-autotune — autonomous operator optimizer (the loop)

The top tier of the trio: [`perf-analyze`](../perf-analyze/SKILL.md) finds
bottlenecks (read-only), [`perf-optimize`](../perf-optimize/SKILL.md) applies
**one** gated fix, and this loops that — analyze → pick the top finding → apply →
gate → re-analyze → repeat — accumulating kept changes until a stop condition,
then presents a cumulative report.

**Explicit opt-in only.** This makes multiple unattended code changes. Invoke it
only when the user asked for autonomous / iterative optimization. Every change
still passes the same three gates as `perf-optimize`, and **nothing is landed
without human approval** — the loop produces a batch of verified changes in an
isolated worktree and stops.

## Gates & guardrails (inherited from perf-optimize, non-negotiable)

- **Correctness** (unit tests), **byte-equivalence** (output checksum), **speed**
  (harness steady median) — all three must pass to keep a change; else revert it.
- Never touch the locked contract (`OPERATOR_API.md`) or the signal math;
  behavior-preserving perf refactors only.
- Byte-equivalence is the default gate; a numerics-changing change needs explicit
  user opt-in + sim-test re-validation.
- **Do not land.** End by presenting the kept changes + report; a human commits.

## Isolation

Run the whole session in a dedicated **git worktree** so the loop's churn never
touches the user's working tree:

```bash
git worktree add ../radar-autotune-<op> HEAD && cd ../radar-autotune-<op>
# ... the loop builds + edits here; the main checkout stays clean ...
# on finish (kept or abandoned): cd back, `git worktree remove` after the human
# has taken/approved the changes.
```

Within the worktree, apply/gate/keep-or-revert each candidate **in place** (git
`checkout --` to revert a failed candidate) — incremental rebuilds keep the loop
fast. (Parallel exploration across multiple worktrees is a future enhancement;
v0 is sequential.)

## The loop

Let `OP` be the operator stem and `BUILD=build-opt` (testing ON, perf + NVTX ON;
see `perf-optimize` prerequisites and the `perf-analyze` container guidance).

1. **Baseline.** Build; run `perf-analyze` and **save** its `analysis.json` and
   `diagnosis.json` as the baseline (the diagnosis is your bottleneck queue).
   Record the baseline steady median, output checksum, and green unit tests.
   Start an empty `session.json` (`mode: "autotune"`).

2. **Iterate** until a stop condition (below):
   1. **Pick** the top-severity actionable finding not yet tried this session
      (skip `info`-level and already-attempted findings).
   2. **Apply** the smallest behavior-preserving change implementing its
      recommended pattern (pool/ping-pong, gate unused work, cache a rebuild).
   3. **Gate** (rebuild → unit tests → checksum vs baseline → harness median vs
      current-best). **Keep** only if correctness holds, the checksum is
      identical, and the median improves beyond noise; otherwise `git checkout
      --` the candidate.
   4. **On keep:** append the change to `session.json` (finding, summary, files,
      diffstat, gates incl. per-size `speed_delta_pct`); update the current-best
      baseline; **re-run `perf-analyze`** so the next pick reflects the new
      bottleneck landscape (a fix often shifts what dominates).
   5. **On revert:** mark that finding tried-no-win; continue.

3. **Finish.** Remove all temporary instrumentation (see below), do a final
   `perf-analyze`, and render the cumulative report:
   ```bash
   python3 performance/agent/optimize_report.py \
     --baseline <baseline analysis.json> --final <final analysis.json> \
     --session session.json --out-dir <run-dir>
   ```
   Present the batch of kept changes + `optimization_report.{md,html}`
   (before→after per size, per-change gates, the before/after chart) for
   approval. **Do not land.**

## Stop conditions (whichever comes first)

- **Budget:** a max iteration count (default ~5) or a wall-time cap.
- **Dry:** `K` consecutive picks yield no keepable win (default `K = 2`).
- **Queue empty:** no actionable (non-`info`) findings remain.
- **Target:** an optional user-set goal (e.g. steady median ≤ X ms) is met.

Whichever fires, go to step 3. Always report what stopped the loop and any
findings left unaddressed (never silently drop remaining bottlenecks).

## Temporary instrumentation (must be removed)

To sharpen an ambiguous finding you MAY add temporary analysis code — extra NVTX
ranges (`RADAR_NVTX_PHASE` / `RADAR_NVTX_RANGE` from
`radar_common/sdk/nvtx.hpp`), scratch timers, debug prints. Mark each with a
`// PERF-SCRATCH` comment. **Remove all of it before the loop ends** — i.e.
after *all* iterations, not per-iteration (you may reuse instrumentation across
iterations). Before the final report, assert it is gone and rebuild:

```bash
grep -rn "PERF-SCRATCH" operators/ performance/   # must return nothing
```

## Notes / limits

- Per-candidate mechanics are exactly `perf-optimize`'s; this adds the outer
  loop, candidate selection, re-analysis, and the cumulative report.
- Needs the operator's `--checksum` sink + a `-L unit` test target wired (as
  `perf-optimize`).
- Re-analysis each kept iteration is what keeps the loop honest — it chases the
  *current* bottleneck, not a stale finding list.
