# MUSICEstimatorOp — open items

Status: **checklist-complete, C++ and Python.** The full contract
(`OPERATOR_API.md` → MUSICEstimatorOp) is implemented:

- sample covariance with `is_valid` masking, batched Hermitian eigendecomposition
  (`matx::eig`, `EigenMode::VECTOR` — ascending eigenvalues, column
  eigenvectors), per-item model-order resolution (fixed or EFT `auto`), the
  noise-subspace projector, the MUSIC pseudospectrum, and the dominant DOA;
- `num_sources` **[live]** (int or `auto`) + `source_count_pfa` **[live]**;
- `max_sources` **[live]** + `doa_multi` (num_items, max_sources, 2) +
  `source_count` (num_items,) — the top-`min(num_sources_i, max_sources)`
  pseudospectrum peaks (pass 0 = global argmax so `doa_multi[·,0,·]` equals
  `aoa_estimates`; further peaks are strict 2-D local maxima, strongest
  first), unused rows NaN;
- `eigenvalues` (num_items, num_channels) — the ascending eigenvalue spectrum
  of the sample covariance;
- `spatial_smoothing` + `forward_backward` reused verbatim from MVDR (same
  shared `radar_common/dsp/spatial_smoothing` front-end transform).

All four optional outputs carry `ConditionType::kNone`, matching MVDR.

**Test suite (40 C++ cases in `test_music_estimator.cpp`), including the
gaps found and closed by explicitly diffing against MVDR's 28-test suite
and `TESTING.md`'s requirements** (an initial 21-test pass was missing
several real, previously-untested code paths):

- `ResolvesCloselySpacedSourcesBeamscanCannot` — the specific capability
  `TESTING.md` names for this operator ("resolves two closely-spaced
  sources that Beamscan cannot"). Caught a test-methodology bug on the
  way: a single random-relative-phase snapshot's "spectrum" is a
  periodogram of one interference realization, not the array's resolution
  limit — it spuriously showed 2 lobes regardless of source separation.
  Fixed by feeding Beamscan the deterministic in-phase combination
  `a(az0) + a(az1)` (the textbook two-point-source resolution setup)
  instead of a snapshot slice.
- Live `[live]` overrides for both scan axes, geometry (including
  warn-once/cache reuse), model order/Pfa, spatial smoothing, and output
  capacity.
- Multi-item (`num_items > 1`) batch with independent per-item DOAs — the
  kernels are one-block-per-item; a single-item suite can't catch an item-
  indexing bug.
- 2-D single-snapshot input (`(num_items, num_channels)`, no explicit
  `num_snapshots` axis) and 2-D (URA) + forward-only (no
  backward-look) spatial smoothing — validated code paths that had no
  test exercising them.
- CUDA-accessible device, pinned-host, and managed input acceptance plus
  pageable-host rejection; deterministic output across repeated runs.
- `RejectsEmptyScanGrid`, `RejectsInvalidSnapshotRank`.

## Shared prerequisite: the EFT source-count helper (settled — S2)

`num_sources: auto` resolves via a new shared helper,
`radar_common/dsp/source_count.{hpp,cpp}` (host reference, CUDA-free) +
`source_count_device.{hpp,cu}` (batched per-item GPU kernel, target
`radar_common_source_count`) — built once here, for reuse by
Root-MUSIC/ESPRIT when they land, per STATUS.md S2 ("Settled; EFT helper +
operators are follow-on").

- **Fidelity note (read before "fixing" the thresholds).** The cited
  reference (Quinlan, Barbot, Larzabal & Haardt, EURASIP JASP 2007,
  doi:10.1155/2007/71953) calibrates its per-step acceptance thresholds
  η_P by **Monte Carlo** (~10^5 trials per `(P, N, Pfa)` triple — the same
  complexity class as this codebase's GO/SO/OS CFAR `alpha_lut`
  calibration). This was an explicit, discussed trade-off (not a silent
  simplification): this implementation instead uses the paper's own
  closed-form noise-eigenvalue-profile model (eq. 25-31) for the
  *prediction* step, but derives the accept/reject threshold from a
  first-order Gaussian approximation of the profile's fluctuation (eq.
  22-24) instead of Monte Carlo calibration. `source_count_pfa` is
  therefore a **target**, not a Monte-Carlo-guaranteed false-alarm rate.
  Tightening this to the exact MC-calibrated test later is a documented
  follow-up if the team wants exact Pfa control — it does not change the
  operator contract (`OPERATOR_API.md` does not pin the calibration
  method).
- **Per-item, on-device, no host round trip.** The device kernel takes the
  per-item valid-snapshot count `k_counts` (already computed by the
  covariance kernel) rather than a single scalar N, so ragged `is_valid`
  masking is handled correctly per item, and bakes in the "rank-deficient
  covariance (`K < num_channels`) falls back to 1" contract clause
  directly (no separate blanking pass needed for that case).
- **The "and warns" half of the rank-deficient fallback is only exact in
  the uniform case.** `MUSICEstimatorOp` warns once when *every* item is
  uniformly rank-deficient (no `is_valid`, or a snapshot shape with fewer
  snapshots than channels) — that check is free (known from the input
  shape on host, no device readback). When `is_valid` is connected and
  ragged (only *some* items fall below `num_channels` valid snapshots),
  those items still correctly fall back to `num_sources = 1` (the kernel's
  own per-item check), but the operator does not warn for that
  sub-case — would need a device→host reduction it does not currently pay
  for. Low severity (functionality is correct; only the diagnostic is
  incomplete) but flag if `source_count_pfa` tuning ever seems to silently
  not apply for a subset of items.

## Noise-subspace projector, not the covariance inverse

Unlike MVDR (which inverts the loaded covariance), MUSIC needs the noise
*eigenvectors* — `matx::eig(..., EigenMode::VECTOR)`, not `NO_VECTOR` — to
build the projector `P_n = Σ e_j e_jᴴ` over the `num_channels - num_sources`
smallest-eigenvalue eigenvectors. `music_reduce_kernel` computes the same
`aᴴ M a` quadratic form as MVDR's `capon_reduce_kernel` with `P_n` standing in
for `R̂⁻¹`, and shares its 4-way-ILP inner loop — but its shared-memory
strategy differs (it stages `P_n`, wave-gated), because profiling showed the
kernel is L1-bandwidth bound on this GPU where MVDR's note assumed
latency-bound; see Performance below.

- **The MUSIC denominator floor is a genuine peak, not noise — do not gate
  it to zero like Capon's.** `aᴴ P_n a → 0` is exactly what makes MUSIC's
  peaks sharp (unlike Capon, where the *whole point* of the adaptive
  inverse means a near-singular result is numerical trouble, not signal).
  `kMusicDenomFloor<T> = 1e-6` floors the division instead of gating it.
  **Caught in testing:** invisible (`direction_cosine` out-of-visible-region)
  beams carry an all-zero manifold entry, so `aᴴ P_n a == 0` for a
  *different* reason (no beam at all, not a peak) — `music_reduce_kernel`
  checks `aᴴa` first and reports those beams as zero power, matching the
  contract ("Invalid (direction-cosine invisible) beams are zero") before
  the peak-preserving floor logic ever runs. Not yet swept against SNR /
  aperture size; revisit with the analytic simulator if peak sharpness or
  invisible-region leakage becomes visible in a real scan.
- **`doa_multi` targets `num_sources`, not always `max_sources`** (the
  contract's `holds up to min(num_sources, max_sources) DOAs`) — a real
  semantic difference from MVDR's `doa_multi`/`source_count` (which always
  greedily reports up to `max_sources` Capon peaks, since Capon has no
  model-order concept). `music_peak_extract_kernel` takes the per-item
  resolved model order as its search target; `source_count` output is still
  "peaks actually found" (≤ that target), matching the shared-port
  contract's "All resolved DOAs" framing.

## Python bindings

`python/_music_estimator_op_bindings.cpp` mirrors MVDR's binding shape
exactly (`MUSICEstimatorOpF32`/`F64`, `retire_music_matx_caches`,
re-exported from `python/holoscan/radar/__init__.py`, and registered with
`install_matx_cache_atexit()` alongside MVDR's cache). `tests/python/
angle_estimation/test_music_estimator.py` has **38 behavioral tests**,
including deterministic output and all live overrides. The two
CUDA-memory-space construction checks (pinned host and managed) remain
C++-only because they validate DLPack transport rather than binding behavior.

The last 7 cases (`SpatialSmoothingRejectsNonUniformGeometry`,
`RejectsChannelMismatch`, `RejectsSnapshotDtypeMismatch`,
`RejectsPageableHostSnapshot`, `DirectionCosineMasksInvisibleRegion`,
`IsValidMaskExcludesWrongDoaSnapshots`, `AllInvalidMaskBlanksDoaAndSpectrum`)
were assumed blocked (a documented, pre-existing gap, mirroring language in
MVDR's own `OPEN_ITEMS.md`) — turned out not to be, once actually tried:

- **`is_valid` as an *input*** (the two `*MaskExcludes*`/`*Blanks*` tests)
  works fine from Python using **`uint8` instead of `bool`** for the mask
  array — the operator's contract already accepts either dtype, and `uint8`
  side-steps HSDK 4.3's bool-`DLPack` opacity entirely (that opacity is only
  proven to bite on the *output*/host-conversion side, via
  `NullSinkOp`/`drain_ports` — nothing in the codebase had actually tried
  bool-as-input, so the assumption it was blocked was never verified). This
  is a candidate to also close the identical gap noted in MVDR's own
  `OPEN_ITEMS.md`, if the team wants that (out of scope for this operator's
  own file to decide unprompted).
- **Pageable-host rejection** works from Python via `holoscan.as_tensor()`
  on a plain (non-CuPy) numpy array, bypassing `to_device_array`'s CuPy
  path deliberately.
- **Dtype mismatch** needed bypassing `run_music()`'s own dtype-coercion
  (it always casts the input to match `op_cls`, which makes constructing a
  genuine mismatch through that helper impossible) and calling
  `run_operator_multi()` directly instead.
- The remaining two (non-uniform-geometry rejection, direction-cosine
  masking) were straightforward, direct translations with no blocker at
  all — simply hadn't been ported yet.

## Inherited caveats (carried over from MVDR, unchanged)

- **`is_valid` bool transport (HSDK 4.3).** Shares the CFAR / SnapshotGather
  bool-`DLPack` transport caveat; `is_valid` is accepted as bool or uint8.
- **MatX eig plan lifetime.** Per tick, `clear_music_matx_caches()` clears the
  cached `eig` plan (keys are freed-at-return scratch pointers); the full
  `MemTracker` drain (`retire_music_matx_caches()`) runs once from the
  operator's `stop()`. MUSIC only ever calls `eig` (VECTOR mode), never `inv`,
  so there is no inverse-plan cache to retire. Root-MUSIC and ESPRIT reuse this
  same pair. **Do not** call `retire_music_matx_caches()` per tick: draining the
  process-global `GetAllocMap()` every tick frees sibling operators' cached FFT
  workspaces and corrupts them (see MVDR `OPEN_ITEMS.md` → "Per-tick
  `GetAllocMap()` drain corrupted sibling FFT operators"). Not yet exercised
  end-to-end under a pytest CuPy-import-order test the way MVDR's was; if a
  Python binding lands, mirror MVDR's `install_matx_cache_atexit()` guidance
  verbatim.
- **The teardown drain is process-global, and that is only safe at true
  teardown.** `retire_music_matx_caches()` empties the entire
  `matx::GetAllocMap()`, not just this operator's buffers — the same
  process-wide MemTracker every MatX consumer shares. This is intentional (it
  is the one point where CUDA is still valid and no operator is mid-compute)
  and correct for the current single-fragment usage, where Holoscan calls
  every operator's `stop()` at graph teardown with no further `compute()` on
  any operator and the Python `atexit` backup fires at interpreter shutdown
  (after `run()` returns). It is **not** safe if operators can stop at
  staggered times while siblings keep ticking — a multi-fragment graph whose
  fragments tear down independently, or an early `atexit`/manual `stop()` on
  one estimator while another still runs — because the first `stop()` would
  free the still-live siblings' cached workspaces (the exact per-tick failure
  mode the split above avoids, just moved to teardown). If such a topology is
  ever built, scope the drain to this operator's own allocations (track them
  explicitly) or reference-count the drain across live MatX consumers rather
  than emptying the global map on the first `stop()`. Flagged by CodeRabbit
  review on the fmcw_aoa_estimators PR; left as-is for the current single-
  fragment app because no staggered-stop topology exists yet.

## Performance

`performance/cpp/perf_music_estimator.cpp` + `performance/python/
perf_music_estimator.py` mirror MVDR's harness (same size ladder, inline
`[live]` `az_angles` override, `perf_music_estimator_smoke` ctest). The
default/full matrices also include an `xs` scenario that overrides every
functionally distinct live path together: az/el grids, geometry, fixed→auto
model order plus Pfa, output capacity, and spatial smoothing. Both harnesses
correctly use the `# isort: off` / `# isort: on` guard around `import
holoscan.radar` / `import cupy` — see `mvdr_estimator_op/OPEN_ITEMS.md`
"Regression: isort silently re-broke the import order" for why that guard
is load-bearing, not cosmetic, whenever a script calls
`install_matx_cache_atexit()`.

`num_sources` is **fixed at 1** across the size ladder (both harnesses), not
`auto` — this keeps the model order, and hence the noise-subspace projector's
cost, deterministic across repeats, isolating the covariance/eig/projector/
scan path from the (separately-kernelled, cheap) EFT source-count decision.
The all-live scenario covers the fixed→auto transition and Pfa override, but
does not constitute a model-order performance sweep; that remains a documented
gap below.

### Getting `nsys` working in a sandboxed/container environment

`nsys profile --stats=true` (the default flags) hung indefinitely with no
output in this environment. Root cause, confirmed via `nsys status -e`:
CPU-sampling collection (`perf_event_open`) is unavailable (no root, Linux
kernel paranoid level 4) — a common container restriction, unrelated to
the GPU trace itself. `nsys profile` sets up CPU sampling by default
alongside the CUDA trace, and hangs rather than failing fast when that
setup can't complete. **Fix:** disable it explicitly and scope the trace
to CUDA only:

```bash
nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --stats=true \
    -o /tmp/report --force-overwrite=true <binary> [args]
```

This is a reusable finding beyond MUSIC — the same hang (and the same fix)
would hit anyone profiling any operator's perf binary in a similar sandbox
(e.g. `perf_mvdr_estimator`, whose own `OPEN_ITEMS.md` "Performance
follow-ups" section presumably was written on a host where `nsys` worked
out of the box, since it has full per-kernel percentages already).

### Measured (RTX PRO 6000 Blackwell GB202, single precision, `--full`)

| size | channels | detections | median_ms (MUSIC) | median_ms (MVDR, same shape) |
| --- | --- | --- | --- | --- |
| xs | 4 | 64 | ~0.9 | ~3.3 |
| sm | 8 | 128 | ~1.0 | ~3.4 |
| md | 16 | 256 | ~1.6 | ~3.9 |
| lg | 32 | 512 | ~6.0 | ~8.3 |
| xl | 32 | 1024 | ~17.2 | ~18.0-18.4 |

MUSIC is *faster* than MVDR at every measured size (no diagonal-load or
batched-inverse step), including `xl` after the two fixes below (it was
~1.5x *slower* at `xl` before the ILP fix, then roughly tied, then faster
after the shared-`Pn` staging fix).

**Root cause found and fixed, not just measured.** The initial ~1.5x `xl`
slowdown was *not* what an earlier version of this note speculated
(`EigenMode::VECTOR` + the extra `projector_kernel`) — the `nsys`
per-kernel breakdown (once the hang above was worked around) showed both
of those at a combined ~0.6% of GPU time, nowhere near enough to explain
it:

```text
Time (%)   Kernel (xl, before fix)
98.4%      music_reduce_kernel<float>          <- the pseudospectrum scan
0.9%       music_covariance_kernel<float>
0.3%       batch_parallel_jacobi_algo1 (eig)   <- VECTOR-mode eig, negligible
0.3%       projector_kernel<float>              <- also negligible
0.1%       music_peak_extract_kernel<float>
```

`music_reduce_kernel` (98.4%) was missing the 4-way-independent-accumulator
ILP unrolling that MVDR's `capon_reduce_kernel` has — it had a single serial
`qr`/`qi` accumulator chain instead. Porting that identical, already-validated
optimization over (`music_stage.hpp::music_reduce_kernel`) measured:

- `xl`: 26.7-26.9 ms → 19.9-20.0 ms (**-25.8%**, C++ and Python agree)
- `lg`: 7.5 ms → 6.0-6.1 ms (**-20.6%**)

### Fix 2: shared-memory staging of `Pn` (`xl` -18.7%, wave-gated)

After the ILP fix, `nsys` still put `music_reduce_kernel` at ~98% of `xl`
GPU time, so it was profiled with `ncu` (which *does* work in this sandbox —
`--set basic`/named `--section`s only need GPU counters, not the CPU
`perf_event_open` that blocks `nsys`; skip the first 5 launches to get past
the discarded process-warmup grid before the real `(1024,1,1)` frame). The
`ncu` Speed-of-Light overturned the inherited assumption:

```text
Memory Throughput   72.2%   (L1/TEX 75%, L2 72%)   <- the limiter
Compute (SM)         7.4%
DRAM                 0.03%
Warp stalls: 81% "L1TEX scoreboard" (waiting on L1/L2 load results)
```

The kernel is **L1-bandwidth bound, not latency-bound** — `music_stage.hpp`'s
old comment (copied from MVDR's "shared was slower, kernel is latency-bound,
reads already cached" note) is **wrong for this GPU**; that MVDR note was
evidently taken on different hardware. `Pn` (num_channels² complex = 8 KB at
32 ch) is re-read from L1 for every one of the thousands of beams, and for a
32-channel array `Pn` reads outnumber manifold reads ~32:1, so they dominate
L1 traffic. **Fix:** cooperatively stage `Pn` into shared once per block, read
it from there in the beam loop (`music_reduce_kernel<T, Staged>`). `ncu`
confirmed the mechanism: memory throughput *rose* to 85% (now L2-bound on the
unavoidable per-beam manifold reads), achieved occupancy went *up* (6.89 →
8.04 active warps/scheduler — staging did **not** cost occupancy here), and
duration dropped.

Measured (same-machine staged vs. forced-non-staged, 3 repeats):

- `xl`: 21.0 ms → 17.2 ms (**-18.7%**)
- `lg`: 6.0 ms → 6.4 ms (**+6.6%**, a *regression*)
- xs/sm/md: within noise (`Pn` is tiny at ≤16 ch)

**Why `lg` regresses, and the gate.** On 188 SMs at ~5 resident blocks/SM,
one full occupancy wave is ~940 blocks. `xl` (1024 items = 1024 blocks) spans
> 1 wave → the GPU is saturated and bandwidth-bound, so staging pays. `lg`
(512) is only ~half a wave → occupancy/latency-starved, so the staging
`__syncthreads` is pure overhead with no bandwidth payoff. The launch
therefore **gates staging on `num_items >= one full occupancy wave`**, where
the wave size is computed at runtime from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count — no hard-coded
size threshold, so it adapts to the GPU (a smaller GPU fills at fewer items
and would stage `lg`). Also gated on `Pn` fitting a 40 KB shared budget
(falls back to the global-read path for large channel counts). Verified: the
gate picks non-staged for `lg` (6.0 ms, no regression) and staged for `xl`
(17.2 ms). All 31 C++ tests pass on a temporarily-forced staged path (unit
grids are all < 1 wave, so they normally exercise only the non-staged path).

### A negative result worth recording: per-beam visibility precompute

Before profiling with `ncu`, the invisible-beam guard (`a_norm` = ‖a‖² per
beam, item-independent) looked like obvious redundant work — recomputed once
per (item, beam) instead of once per beam. Precomputing it into a per-beam
mask kernel and reducing the hot loop to a byte lookup measured **no
improvement** (`xl` 20.07 → 20.35 ms, `music_reduce_kernel` 17.35 → 17.79 ms
at the kernel level — i.e. a wash/slight regression). Reason, confirmed by the
`ncu` run above: the kernel is L1-bandwidth/latency bound, the `a[c]` reads
were already L1-cached (reused by the main loop), and the extra arithmetic
filled otherwise-idle issue slots — removing "free" work frees nothing. It was
reverted. Lesson: profile the *limiter* before optimizing; a plausible
FLOP/redundancy argument was simply wrong about what bounds this kernel.

## Not yet done / deliberately deferred

- **Denominator floor tuning.** `kMusicDenomFloor<T> = 1e-6` is a first cut
  (mirrors MVDR's `kAutoLoadingFactor` in spirit: "guarantees no inf/nan
  without dominating a well-posed case", not swept for resolution/SNR
  sensitivity).
- **`num_sources` is not swept in the perf harness.** Both `perf_music_estimator`
  tools fix it at 1 for determinism (see Performance above); a scenario
  varying model order (hence noise-subspace dimension / projector cost)
  would be a natural follow-up, especially once `max_sources` > 1 usage
  patterns emerge.
- **`music_reduce_kernel` is now L2-bound on the per-beam manifold reads**
  (`ncu` shows 85% memory throughput after `Pn` staging). The manifold is
  read identically by every item/block; a scheme that reuses manifold loads
  across items (e.g. several items per block, or tiling the beam grid) could
  relieve L2, but adds a cross-block/argmax-combine complexity that is not
  clearly worth it — deferred, not attempted.
- **Staging is not applied below one occupancy wave** (see Performance). MVDR
  could plausibly benefit from the same staging on its `xl` case (its
  "shared was slower" note predates this L1-bound finding); re-profiling MVDR
  is a separate follow-up in its own operator.
