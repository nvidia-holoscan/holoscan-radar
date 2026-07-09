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

## 2. CFAR `nci_axes` / Gamma(K) threshold multiplier

**Context.** `OPERATOR_API.md` requires that when `nci_axes` integrates `K`
channels (or leading axes), the cell statistic is Gamma(`K`) and `alpha` is
recomputed from `K` by closed form.

**Current implementation** uses:

```text
alpha = N * (Pfa^(-1/(N*K)) - 1)
```

where `N` is the per-cell training-cell count. This **matches the K = 1**
(exponential) CA-CFAR formula. For **K > 1**, the correct false-alarm
probability under CA-CFAR on integrated Gamma statistics involves a
**Beta / incomplete-beta** relationship, not this power-law inversion.
Unit tests check detect / no-detect at hand-picked levels; they do **not**
verify realized Pfa.

**Impact.** Detectability and false-alarm calibration for `nci_axes != []`
may be wrong even when detections "look reasonable" in synthetic tests.

**Questions for the team.**

- Pin the exact `alpha(K, N, Pfa)` formula in `OPERATOR_API.md`
  (signal-processing owner)?
- Implement incomplete-beta inversion (or a cited approximation) and add
  Monte Carlo Pfa tests per `TESTING.md`?
- Defer non-empty `nci_axes` in code until the formula is settled (revert or
  gate behind a feature flag)?

**Status:** open — implementation landed with the approximate formula;
Monte Carlo Pfa test explicitly deferred.

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
