# FMCWRangeDopplerProcessorOp — Open Items

Deferred, agreed-upon work for this operator. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. Benchmark, then revisit whether the fused operator earns its keep

**Status:** open — **benchmark harness landed**; team keep/kill decision still
pending (do not encode a verdict in CI).

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

### Decision aid (run this; do not treat as an automatic gate)

**Primary tool:** `performance/cpp/perf_fmcw_split_vs_fused` (Python mirror:
`performance/python/perf_fmcw_split_vs_fused.py`).

Times two end-to-end Holoscan graphs with identical I/O:

| Variant | Graph |
| --- | --- |
| `split` | `adc → range → [clutter] → Doppler → sink` |
| `fused` | `adc → FMCWRangeDopplerProcessorOp → sink` |

Default matrix: `{noclt, msub} × {xs, md, lg, xl}` plus `md_ch4` noclt.
`--full` adds `sdel` and channel batches. Emits per-graph rows plus a
**comparison table** (`split_ms`, `fused_ms`, `delta_ms`, `pct_saved`,
`speedup`). Suggested artifact paths:

```bash
./build/performance/cpp/perf_fmcw_split_vs_fused \
  --output perf_results/perf_fmcw_split_vs_fused.json \
  --text   perf_results/perf_fmcw_split_vs_fused.txt
./build/performance/cpp/perf_fmcw_split_vs_fused --full \
  --output perf_results/perf_fmcw_split_vs_fused.full.json \
  --text   perf_results/perf_fmcw_split_vs_fused.full.txt
```

**Do not** decide from a naive sum of
`perf_fmcw_range_processor` + `perf_clutter_filter` +
`perf_fmcw_doppler_processor` medians — that omits inter-op Holoscan cost on
the split path (and was the gap this tool closes).

**Decision to make after reviewing the table** (and optional `nsys` of
`applications/fmcw_siso` split vs fused configs):

- If the end-to-end gain is negligible at realistic workloads, consider
  retiring the operator (or demoting it to an example of stage
  composition) rather than carrying the API surface forward.
- If the gain is real, or once a MatX upgrade with `-DMATX_EN_MATHDX=ON`
  (cuFFTDx) lands — which would let the expression-fed FFTs fuse their
  elementwise pre-ops and widen the gap — keep it and paste the numbers
  into this section.

Related: the split-vs-fused parity contract and stage-sharing design are
enforced by `tests/cpp/test_fmcw_range_doppler_processor.cpp` (`BitExact*`).
