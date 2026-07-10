# FMCWRangeDopplerProcessorOp — Open Items

Deferred, agreed-upon work for this operator. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. Benchmark, then revisit whether the fused operator earns its keep

**Status:** agreed 2026-07-10, deferred until representative benchmarks exist.

The fused operator now fuses at the only seam the FFT barriers allow
(cuFFT in MatX v0.9.4): the clutter expression composes into the Doppler
pre-FFT pass (`with_clutter_expr` → the `run_doppler_stage` expression
overload), saving one full-cube write+read, one kernel launch, and one
async alloc/free per tick versus the split chain — plus the graph-level
savings of one node (no inter-operator message passing or intermediate
Holoscan tensor). The FFTs still dominate the tick, so the computational
win is expected to be modest.

What it costs to keep: a third operator to maintain (union-of-parameters
API surface, override plumbing, its own test suite including the
`BitExact*` parity gates), and the expression-overload machinery exists
primarily for its benefit.

**Decision to make after benchmarking** (`nsys profile` of fmcw_siso or a
dedicated benchmark at representative cube sizes, split chain vs fused,
clutter enabled):

- If the end-to-end gain is negligible at realistic workloads, consider
  retiring the operator (or demoting it to an example of stage
  composition) rather than carrying the API surface forward into the
  MIMO phase.
- If the gain is real, or once a MatX upgrade with `-DMATX_EN_MATHDX=ON`
  (cuFFTDx) lands — which would let the expression-fed FFTs fuse their
  elementwise pre-ops and widen the gap — keep it and record the numbers
  here.

Related: the split-vs-fused parity contract and stage-sharing design are
documented in `docs/planning/SIGPROC_PLAN.md` (item 6) and enforced by
`tests/cpp/test_fmcw_range_doppler_processor.cpp` (`BitExact*`).
