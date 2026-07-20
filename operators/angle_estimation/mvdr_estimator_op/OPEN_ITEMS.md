# MVDREstimatorOp — open items

Status: **checklist-complete.** The full contract
(`OPERATOR_API.md` → MVDREstimatorOp) is implemented:

- sample covariance with `is_valid` masking, diagonal loading (`auto` / fixed),
  batched Hermitian inverse, Capon spectrum, dominant DOA;
- `max_sources` **[live]** + `doa_multi` (num_items, max_sources, 2) +
  `source_count` (num_items,) — the top-`max_sources` Capon-spectrum peaks
  (pass 0 = global Capon argmax so `doa_multi[·,0,·]` equals `aoa_estimates`;
  further peaks are strict 2-D local maxima, strongest first), unused rows NaN;
- `eigenvalues` (num_items, num_channels) — the ascending eigenvalue spectrum
  of the raw (pre-loading) sample covariance (batched `matx::eig`, NO_VECTOR).

All four optional outputs (`angle_spectrum`, `doa_multi`, `source_count`,
`eigenvalues`) carry `ConditionType::kNone`, so a consumer that only wants
`aoa_estimates` leaves them unwired without gating scheduling. They are always
emitted (see the eigenvalues perf note below).

## Spatial smoothing (settled — S1)

`spatial_smoothing` (+ `forward_backward`) is implemented as a front-end
transform (shared helper `radar_common/dsp/spatial_smoothing`): it detects the
array lattice, gathers the snapshot into overlapping sub-array looks (forward /
forward-backward), forms the averaged covariance on the reduced sub-array
geometry, and the covariance / eig / inverse / Capon path runs unchanged on the
`L` reduced channels. 1-D (ULA) and 2-D (URA) are both supported; the parameter
is a per-aperture-axis vector of sub-array **counts** (`[P]` / `[P_az, P_el]`).
See STATUS.md (S1 settled decision).

- **Non-uniform / holey arrays are rejected** (clear error) — they must be
  reduced to a uniform sub-array upstream (S3), or a custom smoother can be
  supplied as an upstream operator emitting a reduced `(snapshot, geometry)`
  pair.
- **Two team forks remain for ratification** (do not silently re-resolve): the
  integer means sub-array **count** (vs. length); and holey arrays route to S3
  rather than accepting in-operator explicit partitions. See TEAM_BLOCKED (L13).

The related "where does the source count come from" question (fixed vs.
MDL/AIC vs. CFAR/cluster seed) does **not** affect MVDR: `max_sources` is simply
the number of Capon peaks to report (a reporting capacity, not a model order).
That shared question still governs MUSIC/ESPRIT's `num_sources`.

## Representation notes

- **`diagonal_loading` encoding.** The contract types this param as "float or
  auto". Pending the shared descriptor/YAML story, it is encoded here as a
  `double`: a negative value selects auto (δ scaled to
  `kAutoLoadingFactor · trace(R̂)/C`), a non-negative value is the absolute δ.
  Revisit when Python bindings / YAML converters land so a literal `auto`
  string round-trips.
- **Auto-loading factor.** `kAutoLoadingFactor = 1e-2` is a first cut. It
  guarantees invertibility (including rank-deficient `K < C`) without
  dominating a well-conditioned covariance, but has not been swept against
  resolution / SNR; tune with the analytic simulator.

## Performance follow-ups

Profiled on an RTX PRO 6000 (Blackwell GB202), `perf_mvdr_estimator --full`.
`capon_reduce_kernel` is ~97% of GPU time (nsys `cuda_gpu_kern_sum`); the
covariance kernel and the batched `matx::inv` are ~1.6% / ~1.3%. GPU-metrics
sampling shows the Capon kernel is **latency-bound**, not memory-bound: DRAM
throughput ≈0%, tensor cores unused, SM issue-rate tiny while warps are
resident (they stall). Findings:

- **Staging `R̂⁻¹` in shared memory measured *slower*** (lg 5.3→8.7 ms) and was
  reverted. Every thread reads the same `R̂⁻¹[c,c2]` in lockstep, so the global
  reads already broadcast from L1/L2; staging only added a copy + `__syncthreads`
  and cost occupancy. Do not retry this.
- **Inner-loop ILP (four independent accumulators) landed** — breaks the serial
  reduction dependency chain so FMAs pipeline. Measured: xl 20.0→14.9 ms
  (−26%), lg ~−4%, md flat, no regressions.
- **Remaining ceiling is occupancy + the one-thread-per-beam mapping.** The
  kernel uses ~66–68 registers/thread (≈3 blocks/SM), and `lg` launches only
  ~2.7 blocks/SM. The larger structural wins (not yet done): a warp-cooperative
  reduction (a warp per beam → more warps to hide latency + shorter per-thread
  chains), or a cuBLAS batched GEMM `M = R̂⁻¹·A` followed by a cheap reduction
  (tile over beams to bound the `O(num_items·num_channels·num_beams)` `M`).
- Covariance + inverse scratch is `O(num_items · num_channels²)`; chunk the item
  axis if a workload needs wider apertures at high detection counts.
- **`eigenvalues` runs a batched `matx::eig` (heevd) every tick, even when the
  port is unwired.** Measured overhead vs. the Capon-only pipeline is small
  (lg 5.3→5.4 ms ≈ +2%, xl 14.9→15.6 ms ≈ +4%, md flat; cold start grows with
  the cached heevd workspace). `peak_extract_kernel` (multi-source) is
  negligible — it re-reads the resident spectrum. If a workload never consumes
  `eigenvalues`, gating the `eig` on the port's connectivity is the obvious
  win, but Holoscan gives no clean in-`compute()` "is this output connected"
  signal (the codebase idiom gates optional outputs on a *parameter*, e.g.
  FMCWRangeProcessor's `waveform`); revisit if the diagnostic cost matters.

## Inherited caveats

- **`is_valid` bool transport (HSDK 4.3).** Shares the CFAR / SnapshotGather
  bool-`DLPack` transport caveat; `is_valid` is accepted as bool or uint8.

## Python teardown double-free / SIGSEGV

**Status:** fixed in-tree.

Two MatX lifetime bugs stacked:

1. **`inv` plans keyed on `(A, A_inv)` device pointers.** Holoscan frees
   cov/rinv when `compute()` returns; cached plans outlive those keys.
2. **`MemTracker` leftovers.** Even after `GetCache().Clear`, ~64B of
   MatX-tracked temps remain. `~MemTracker` frees them with `cudaFreeAsync` at
   process exit — after CuPy has torn down CUDA under pytest → SIGSEGV
   (tests already PASSED).

**Fix:** `retire_matx_solver_caches()` at the end of each `run_mvdr_stage`:
`Clear` eig/inv caches, then drain `matx::GetAllocMap()` while CUDA is still
valid. `install_matx_cache_atexit()` after `import cupy` remains as backup.
`tests/python/conftest.py` imports `holoscan.radar` **before** CuPy —
CuPy-first makes MatX eig `Clear` abort on this stack.

Apps: import `holoscan.radar` **before** CuPy, then call
`holoscan.radar.install_matx_cache_atexit()` once. Do not register
`retire_matx_solver_caches()` from C++ `std::atexit`/`~Op` alone — too late
once CUDA is gone.

## Python mid-run double free in `retire_matx_solver_caches()` (fixed)

**Status:** fixed — was an import-order footgun, not a MatX keying bug.

`performance/python/utils/__init__.py` eagerly imported CuPy-using harness
modules. Any `from utils.bootstrap import …` therefore loaded **CuPy before
`holoscan.radar`**, which makes MatX eig/inv `GetCache().Clear` abort with
`double free or corruption` when the per-tick `retire_matx_solver_caches()`
runs (same CuPy-first rule as above). Pytest's `conftest.py` imports radar
before CuPy, so unit tests never hit it; `perf_mvdr_estimator.py` did.

**Fix:** keep `performance/python/utils/__init__.py` free of CuPy/Holoscan
imports so bootstrap cannot pull CuPy early. Scripts still do
`import holoscan.radar` then `import cupy` then `install_matx_cache_atexit()`.
