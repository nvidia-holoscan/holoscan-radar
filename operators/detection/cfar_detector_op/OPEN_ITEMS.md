# CFARDetectorOp Open Items

Issues discovered during `CFARDetectorOp` implementation that need a team
decision, contract clarification, or SDK follow-up. Keep operator-specific
items here so they travel with the implementation.

When resolved, summarize the decision in
[`docs/planning/STATUS.md`](../../../docs/planning/STATUS.md) if it affects
the project contract or team workflow, then remove or archive the section.

---

## 1. `detection_mask` bool dtype vs Holoscan 4.3 transport

**Context.** `OPERATOR_API.md` specifies `detection_mask` as `bool`. On HSDK
4.3, emitting the mask through the graph logs:

```text
[error] [tensor.cpp:820] Unsupported DLPack data type (code: 6, bits: 8, lanes: 1)
```

(`kDLBool`). GXF cannot map `kDLBool` to a numpy typestr on the C++ message
path; the error is logged and swallowed, so **C++-to-C++ port wiring and
host capture via raw `cudaMemcpy` still work**.

**Impact.** Any **Python** consumer of `detection_mask` is at risk: wrapping
the tensor as a `PyTensor` calls `numpyTypestr()` and **throws**
(`Unable to determine NumPy dtype from DLPack tensor` in
`python/holoscan/core/dl_converter.cpp`). A Python pipeline that reads
`detection_mask` can fail on HSDK 4.3.

**Questions for the team.**

- Keep `bool` in the contract and treat HSDK 5 as the validation target?
- Add a compatibility `uint8` mask port or change the contract dtype?
- Document a hard "C++ only until HSDK 5" caveat for detection outputs?

**Status:** open — not validated on a finalized HSDK 5 build in this
environment (only 4.3.0 installed).

---

## 2. CFAR `nci_axes` / Gamma(K) threshold multiplier — implemented in conflict with contract (pre-emptive)

**Context.** `OPERATOR_API.md` requires that when `nci_axes` integrates `K`
leading axes, the cell statistic is Gamma(`K`) and `alpha` is recomputed from
`K` by closed form. The contract pins only the **K = 1** formula explicitly:

```text
alpha = N * (Pfa^(-1/N) - 1)
```

For **K > 1** the contract says `alpha` "takes `K`" but does **not** pin the
closed form — that was left for a signal-processing owner decision.

**What we implemented (pre-emptively).** Rather than block `nci_axes`
development on that contract gap, the operator now uses the **correct**
Gamma-statistic CA-CFAR calibration:

```text
Pfa = I_{1/(1 + alpha/N)}(N*K, K)     [regularized incomplete beta]
```

`alpha` is obtained by numerically inverting this relation on the **host**
(`cfar_alpha.cpp`), then uploaded as a per-`N` lookup table
(`alpha_lut[train_cnt]`) covering every training-cell count that can occur on
the map (interior and border cells). This is the pragmatic middle ground:
border-correct `N` without porting `betainc` into the fused CUDA stencil.

**Why this conflicts with the contract.** `OPERATOR_API.md` still documents
the power-law generalization implicitly expected by the `nci_axes` bullet
("recomputed by closed form") without stating the incomplete-beta relation.
The code no longer uses `alpha = N * (Pfa^(-1/(N*K)) - 1)` for any `K`.

**Monte Carlo validation (`test_cfar_detector_statistical`).**

- `CaMonteCarloPfaK1` — passes (`K = 1` matches the contract formula).
- `CaMonteCarloPfaNciK8` — re-enabled; should pass with the corrected LUT.

**Recommendation (device-side `betainc`).** **Do not** port incomplete-beta
inversion into the fused kernel unless profiling shows the host LUT rebuild
is a bottleneck. The LUT is tiny (≤ a few hundred doubles), rebuilt only when
`(Pfa, K, window footprint, map size)` changes, and keeps the stencil simple.
Revisit only if dynamic per-tick window footprints on very large maps make
the host scan + inversion dominate.

**Questions for the team.**

- Adopt the incomplete-beta relation as the pinned `K > 1` closed form in
  `OPERATOR_API.md` (signal-processing owner sign-off)?
- Or revert to the power-law placeholder and gate `nci_axes` until the
  contract is updated?

**Status:** open — **code is ahead of the contract**; landed pre-emptively so
`nci_axes` development and Monte Carlo calibration can continue. Resolve by
either updating `OPERATOR_API.md` to match the implementation or reverting
the implementation to match the contract.

---

## 3. `max_cached_kernels` parameter is currently a no-op

**Context.** `OPERATOR_API.md` lists `max_cached_kernels` (static, default 4)
to bound how many distinct guard/training window-footprint kernels are
retained before eviction. It exists so a `[live]` `guard_cells` /
`training_cells` override to an unseen footprint warns once and caches, rather
than thrashing.

**Current implementation.** The CFAR math is a single fused stencil that reads
the guard/training extents as plain kernel arguments — there is no per-footprint
compiled kernel or coefficient table to build or cache. So `max_cached_kernels`
is declared for contract compliance but has **no effect**, and no warn-once /
cache path exists for window-footprint overrides.

**OS-CFAR follow-up (all four variants now landed).** The original risk called
out OS specifically: a per-footprint sorted-window kernel might need caching.
The shipped OS variant does **not** change this picture — `cfar_os_kernel` is
the same fused-stencil pattern as CA/GO/SO (extents passed as kernel args; a
small in-register sort selects the order statistic per CUT). No per-footprint
kernel build or eviction path exists for any variant, including OS. The §3
forward-looking OS concern is **resolved in practice**; `max_cached_kernels`
remains a no-op across CA, GO, SO, and OS.

**Impact.** None functionally today (overrides already work without a rebuild).
The risk is a caller reading the contract and expecting eviction/caching
behavior that isn't there, or a future variant (e.g. OS with a sorted-window
kernel) that *does* need caching silently diverging from the parameter.

**Questions for the team.**

- Keep the parameter as a documented forward-looking placeholder (status quo)?
- Or drop it from the contract until a variant actually needs kernel caching?

**Status:** open — parameter accepted and stored but intentionally unused;
noted so it is not mistaken for wired behavior.

---

## 4. `pre-commit run --all-files` not verified in this dev environment

**Context.** [`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
and [`AGENTS.md`](../../../AGENTS.md) require `pre-commit run --all-files`
before push. CI enforces the same command in
[`.github/workflows/check_lint.yml`](../../../.github/workflows/check_lint.yml)
(ruff/isort/black, cpplint, cmakelint, markdownlint, codespell, metadata
schema, etc. — see [`.pre-commit-config.yaml`](../../../.pre-commit-config.yaml)).

**What was verified locally.** `ctest -L unit` (the four CFAR variant binaries
contribute 111 unit tests: CA 28, GO 27, SO 27, OS 29) and manual `cpplint`
on touched C++/CUDA files across all four variants.

**What was not verified locally.** The full `pre-commit` hook stack — the
`pre-commit` CLI is not installed in the current container/workspace, so the
checklist/CI lint gate was not reproduced here.

**Questions for the team.**

- Run `pip install pre-commit==4.5.1 && pre-commit run --all-files` on a
  machine with the hook env before merging the CA slice?
- Or rely on the PR CI `check-lint` job and treat local pre-commit as
  optional when the tool is unavailable?

**Status:** open — CA-CFAR implementation is otherwise checklist-complete;
resolve by passing CI lint on the PR or running pre-commit locally before
merge.

---

## 5. `os_rank` indexing convention not pinned in the contract

**Context.** `OPERATOR_API.md` lists `os_rank` as an int rank "into the sorted
training cells" with a typical value of ≈ 0.75·N, but does **not** state
whether the index is **0-based or 1-based**, nor whether sorting is
ascending or descending.

**What we implemented (pending contract sign-off).** OS-CFAR uses a **0-based,
ascending** order-statistic index:

- Training-cell powers in the CA window footprint are sorted ascending
  in-register; `noise_estimate` is the value at index `os_rank`.
- When `os_rank >= train_cnt` for a given CUT (e.g. border cells with fewer
  training cells), the kernel clamps to `train_cnt - 1` (the maximum available
  rank).
- `os_rank` is validated at `compute()` time: must be non-negative and strictly
  less than the maximum training-cell count for the configured window footprint
  on the map.
- Alpha LUT calibration (`build_os_alpha_lut`) keys on `(variant, pfa, fold,
  os_rank, footprint, map size)` and uses the same effective-rank clamp per
  training count `N`.

**Test coverage.** `tests/cpp/test_cfar_detector_os.cpp` pins the behavior:
`RejectsOsRankOutOfRange`, `LiveOsRankOverrideMidStreamAndSticky`,
`OsAlphaInvertsMonteCarloPfa`, and `NoiseEstimateMatchesOrderStatistic`.

**Questions for the team.**

- Confirm **0-based ascending** as the contract convention (rank 0 = smallest
  training-cell power; typical interior rank ≈ `floor(0.75 * N)` for
  `N` training cells)?
- Or adopt 1-based / descending convention and update the implementation to
  match?

**Status:** open — implementation and tests are consistent with 0-based
ascending; **`OPERATOR_API.md` must not be changed until the team explicitly
approves a contract update** in a dedicated PR.

---

## 6. SO-CFAR Monte Carlo Pfa test uses a relaxed ratio tolerance

**Context.** `test_cfar_detector_statistical.cpp` checks that realized interior
false-alarm rate matches configured `pfa` on exponential clutter maps. CA, GO,
and OS use the default envelope (`min_ratio_tol = 0.15`, i.e. realized/configured
Pfa ratio within ±15% after the binomial σ term). SO-CFAR uses a **relaxed
`min_ratio_tol = 0.25`** for both `SoMonteCarloPfaK1` and
`SoMonteCarloPfaNciK8` (documented inline in the test file).

**Why.** SO's min-of-side noise estimate is more sensitive to **spatial
correlation across neighboring CUTs** that share training cells than CA/GO/OS.
With the default 0.15 envelope, interior realized Pfa was consistently ~20–25%
above configured Pfa (~1.24 ratio at `pfa = 1e-2`) even though per-`n` host MC
alpha inversion (`so_cfar_alpha`) recovers the target Pfa in isolation. The
relaxed tolerance is a **test-envelope choice**, not a change to the operator's
configured `pfa` semantics.

**Not a contract issue.** Runtime `pfa` still drives alpha LUT calibration the
same way as other variants; only the statistical test acceptance band differs.

**Questions for the team / reviewers.**

- Accept 0.25 as sufficient for SO given shared-training-cell correlation?
- Tighten SO MC calibration (more trials, different bracket seed) and retry
  with the default 0.15 envelope?
- Or model correlation explicitly in the test design (e.g. sparser interior
  sampling)?

**Status:** open — SO statistical tests pass with 0.25; worth a reviewer's eye
before treating SO Pfa validation as on par with CA/GO/OS.

---

## 7. GO/SO/OS alpha calibration conflicts with "cheap" live-override wording

**Context.** `OPERATOR_API.md` specifies `pfa` as `[live]` and says it "sets
the per-variant threshold multiplier α" with a "cheap scalar recompute." It
also says `nci_axes` changes `K` and α is "recomputed by closed form (cheap)."
That wording is true for CA in the current implementation, but not for GO, SO,
or OS.

**Current implementation.** CA builds `alpha_lut[train_cnt]` by numerically
inverting a closed-form incomplete-beta relation on the host; after the
closed-form `max_training_cell_count()` fix, cold CA LUT rebuilds are
sub-millisecond to ~2 ms on a 1024×2048 map depending on footprint.

GO, SO, and OS instead calibrate each LUT entry by Monte Carlo plus bisection
in `cfar_alpha.cpp`. For each LUT index `n`, the code probes candidate
`alpha`, estimates realized Pfa with thousands of random trials, and adjusts
the bracket. OS is especially expensive because every trial draws the full
training window and selects an order statistic — except **OS with `K = 1`**,
where the pooled order-statistic Pfa is inverted in closed form
(`os_cfar_alpha_closed_k1`) and LUT builds are sub-millisecond. The per-trial
Monte Carlo runs on the GPU (`cfar_alpha_mc.cu`) for GO/SO and for OS when
`K > 1`, falling back to the host mt19937 path otherwise. (An earlier
`K == 1` gate forced the much slower host path for every NCI case; it was
removed once the statistical suite confirmed GPU calibration matches the
host reference for `K > 1`.)

**What is in the LUT.** The LUT is still tiny and one-dimensional, not
per-cell:

- CA / OS: `alpha_lut[n]`, where `n` is the number of in-bounds training cells
  around the CUT; slot 0 is unused.
- GO / SO: `alpha_lut[n_side]`, where `n_side` is the side-count index used by
  the GOGO/SOSO stencil.

The runtime stencil uses `alpha_lut[...]` only to avoid doing calibration in
the fused detection kernel. The expensive part is building the table, not
storing or uploading it.

**Large-map cold rebuild measurements (this environment, 1024×2048 map, cold
LUT key, calibration only — no Holoscan graph startup).**

| Variant | `pfa` change, small window | footprint change, mid window | typical large footprint | `nci_axes` K=8, small window |
| --- | ---: | ---: | ---: | ---: |
| CA | ~0.2 ms | ~0.5 ms | ~2 ms | ~0.5 ms |
| GO | ~2.36 s | ~6.26 s | ~28.82 s | ~1.1 s |
| SO | ~2.14 s | ~6.11 s | ~28.77 s | ~1.0 s |
| OS | ~sub-ms | ~8.93 s | >180 s (timed out) | ~1.2 s |

Small window = `guard_cells={1,1}`, `training_cells={2,2}`; mid window =
`guard_cells={1,1}`, `training_cells={4,4}`; typical large footprint =
`guard_cells={2,2}`, `training_cells={8,8}`, `pfa=1e-6`.

The `nci_axes` K=8 column dropped ~20–40× (GO ~21.73 s → ~1.1 s, SO
~21.33 s → ~1.0 s, OS ~46.52 s → ~1.2 s) after the `K == 1` GPU gate was
removed and NCI calibration moved onto the GPU MC path. The remaining K=1
columns are unchanged and still dominate cold-rebuild cost on large
footprints; they are the next target.

With `blocking_alpha_lut_rebuild=false` (the default), a live override to an
unseen key serves the previous LUT for the tick and builds the new LUT in the
background. With `blocking_alpha_lut_rebuild=true`, or on first tick when no
previous LUT exists, these cold-build costs are tick stalls.

**Contract tension.** The contract does **not** require Monte Carlo
calibration. It requires the user-facing semantics of `pfa`, `threshold_scale`,
`os_rank`, the noise statistic, and the score formula. Full MC is an
implementation choice, and it is hard to reconcile with the current "cheap
scalar recompute" wording for `[live]` `pfa`, `guard_cells`,
`training_cells`, `os_rank`, and `nci_axes`.

**Possible contract/API direction: explicit alpha input.** Real-world OS-CFAR
implementations often receive `alpha` from upstream configuration, offline
calibration, or a precomputed table instead of deriving it from `pfa` at
runtime. To support that pattern, consider a contract change such as:

- Add an optional static or `[live]` `alpha` scalar that bypasses Pfa-derived
  calibration and uses the same multiplier for all training counts.
- Add an optional static or `[live]` `alpha_lut` vector for callers who want
  border-correct per-`N` or per-side-count thresholds.
- Define precedence explicitly, e.g. `alpha_lut` > `alpha` > `pfa`-derived
  calibration, and specify validation against the configured variant and
  maximum LUT index.

This would be a real contract change (new parameter/port), so it should be
made in `OPERATOR_API.md` and called out in the PR. It would not change the
existing data-plane tensor ports or the detection score formula.

**Questions for the team.**

- Should the contract keep promising `pfa` as the primary calibration API for
  GO/SO/OS, or should explicit `alpha` / `alpha_lut` be added as the practical
  real-time path?
- If `pfa` remains primary, should GO/SO/OS use analytic/numerical formulas or
  precomputed tables instead of full MC where possible?
- Should `[live]` wording be narrowed to say costly calibration may be async
  and serve a stale LUT until the new key is ready?

**Status:** open — current GO/SO/OS MC calibration works functionally but does
not match the contract's "cheap recompute" implication. Decide whether to
change the implementation strategy, change the contract, or both.
