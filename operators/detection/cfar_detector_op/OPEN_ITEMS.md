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

## 2. CFAR `nci_axes` / Gamma(K) threshold multiplier — **implemented in
conflict with contract (pre-emptive)**

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

**What was verified locally.** `ctest -L unit` (52 tests) and manual `cpplint`
on touched C++/CUDA files in the CA slice.

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

## 5. GO-CFAR `nci_axes` / Gamma(K) calibration — **K = 1 validated; K > 1
deferred**

**Context.** GO-CFAR uses a 2-D GOGO noise model (greatest-of on range sides,
then Doppler sides, then max across axes). `alpha` is calibrated on the host
via Monte Carlo inversion (`cfar_alpha.cpp`), bracketed from the CA closed-form
seed. `GoMonteCarloPfaK1` validates realized Pfa ≈ configured `pfa` for
homogeneous clutter with `K = 1`.

**What is not yet covered.** There is no `GoMonteCarloPfaNciK8` statistical
test (unlike CA). GO NCI integration reuses the same Gamma(`K`) CUT statistic
and applies `go_cfar_alpha(pfa, n_side, K)` when building the LUT, but the
Monte Carlo simulator was validated only for `K = 1` in this slice.

**Questions for the team.**

- Add `GoMonteCarloPfaNciK8` before claiming GO contract completeness for NCI
  Pfa, or explicitly defer alongside §2's contract gap?

**Status:** open — GO unit NCI tests cover shape/detectability; statistical Pfa
for `K > 1` is not yet validated.
