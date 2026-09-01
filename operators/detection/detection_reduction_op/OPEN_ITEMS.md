# DetectionReductionOp Open Items

Issues discovered during `DetectionReductionOp` implementation. HSDK 5 / EA
migration items are indexed in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream `[live] representative` / `score_method` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + revert-to-Params”;
Performance → inline live-override measurement (when Graph-ported).

**Class:** Wiring settled (L16 / L17, 2026-08-21); Graph test coverage still
deferred on EA. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 and
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Decision.** `representative` / `score_method` / `feature_weights` join the
`OnAligned{labels, detection_idx, points}` trigger with `Exact{}` and
`on_missing SafeDefault{}`, each declared unconditionally. Wiring is the
application's only statement, as it is module-wide. A matched override
applies to that tick only; a tick with no match uses the Params value. The application stamps the frame's transmit time on both the override
and the aligned tensors.

**The `expect_*` gate is retired.** It existed only because the SDK once
rejected an unwired non-trigger `Exact{}` entry under an aligned trigger with
`TEMPORAL_INPUT_UNCONNECTED`; the SDK admits it as of pin `f0cb828d4`
([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11), so the three
parameters are gone. The live suite now wires all three ports in every case,
including cases that never emit on one — that is also the assertion that an
enabled-but-silent port does not stall the aligned trigger.

**Unit coverage landed (2026-08-28).** The enum-edge fix (§12) unblocked all
three tests, and they are the module's first operator-level assertions of a
wired override under an `OnAligned` trigger:

- `LiveRepresentativeAppliesForOneTick` — two points in one cluster with
  `feature_weights = (1, 0)`, chosen so `kFirst` keeps row 0 and `kPeak` keeps
  row 1. The reported representative index therefore names the policy, and the
  third tick shows the revert to the Params policy after the override tick.
- `LiveScoreMethodAppliesForOneTick` — with `w = (1, 1)`, `kWeightedSum` ranks
  row 0 above row 1 (7 vs 6) while `kWeightedL2` reverses it (5 vs 6), so the
  peak row names the score method.

Params authoring remains covered separately
(`RepresentativeConfigurableFromParams` / `ScoreMethodConfigurableFromParams`).

**Status:** closed. The test-coverage gap is closed, the contract wiring is
settled, and the `expect_*` gate is retired.

---

## 2. `[live] feature_weights` rides a host config Tensor

**Blocks checklist:** none — the `feature_weights` override is wired.

**Class:** Durable HSDK 5 (`PublishedPayload` admission). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<double>>` is rejected at compile time.
`feature_weights` length equals `num_features` from the caller-assembled
`points` matrix, so a fixed POD like `SnapshotWindowOverride` does not fit
either.

**Mapping.** `feature_weights` is an `Input<Tensor>` carrying a rank-1
pageable-host float64 vector, validated on arrival (rank, host memory,
contiguity, dtype, and a `MaxTensorExtents.max_features` extent bound) and
converted to `std::vector<double>` per tick, with no state kept. Like the
enum lives its contract entry is unconditional (§1); an arriving sample makes
`max_features` a required extent and is rejected without it. Graph coverage:
`LiveFeatureWeightsExactMatchChangesSelection`,
`LiveFeatureWeightsSkewedCaptureMissesExact`,
`LiveFeatureWeightsRevertsToParamsAfterMiss`,
`LiveFeatureWeightsRevertsToParamsAfterMiss`,
`LiveFeatureWeightsRejectsMalformedLength`,
`LiveFeatureWeightsRejectsBoundOverflow`.

**Next.** Nothing owed unless an owning-container payload family lands, at
which point the config Tensor can collapse back to the logical vector type.

**Status:** settled — the payload form differs from the logical Params type;
behavior conforms.

---

## 3. Optional `points` is Exact in the temporal contract (EA)

**Blocks checklist:** Tests → unconnected `points` with `representative=first`;
`RejectsPeakWithoutPoints` Graph path.

**Class:** HSDK 5 EA gap — no optional input (same family as FeatureAssembly
optional gathers / Beamscan `is_valid`). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11.

**What fails.**

- `SafeDefault` / `LatestWithin` on optional Tensor `points` still yields
  `TEMPORAL_INPUT_UNCONNECTED` when left unwired under this operator's
  `OnAligned` trigger: `on_missing` governs a declared *and connected*
  input.
- Omitting `points` from `contract()` while wiring it makes the sample
  unavailable in `compute()` and kills the second activation with
  `RUNTIME_OPERATOR_REPLAY_MISMATCH` (peak/medoid Graph paths fail).
- The unwired-tolerance the `[live]` overrides rely on (§1, §11) does not
  transfer: it covers a *non-trigger* entry, and `points` is a trigger port,
  so omitting or relaxing its entry would change what the operator triggers
  on, not just whether an override is selected.

**Interim.**

- `contract()` uses `OnAligned(labels, detection_idx, points)` + `Exact` on
  all three. Graph tests always wire `points` (dummy zeros for first-only).
- Runtime still validates “points required for peak/medoid” when a sample is
  absent.
- `RejectsPeakWithoutPoints` is `GTEST_SKIP`’d.

**Next.** Restore optional-Tensor recipe (unconnected first-only + wired
peak/medoid) on a newer EA / GA; unskip the Graph test.

**Status:** open — EA; Graph peak/medoid covered with always-wired points.

---

## 4. Exact-sized `rep_idx` forces a scalar device→host sync

**Context.** Contract requires `rep_idx` sized to `num_clusters`. Host
allocators need the count before `allocate_tensor`, so compute syncs a
scalar cluster count (and max-label) each tick.

**Status:** open — same class of cost as CFAR exact-sized `detection_idx`.

---

## 5. Dense label assumption

**Context.** Implementation treats non-noise labels as integers in
`[0, max_label]` and allocates `max_label+1` slots (cap 2^20). Gaps are
skipped when compacting. Huge sparse label ids are rejected.

**Status:** open — matches expected `ClusterDBSCANOp` output; revisit if
labels are remapped differently.

---

## 6. Custom `ScoreFn` template path

**Context.** Contract `score_method=custom` uses a compile-time device
functor (`DetectionReductionOp<T, ScoreFn>`). Implemented: custom peak
kernels are header templates; default `ScoreFn = NoCustomScore` rejects
`custom` only when `representative=peak`; callers with a real functor must
explicitly instantiate the op in a CUDA TU (`detection_reduction_op_impl.cuh`
is installed with the operator headers). Graph unit coverage for a non-default
`ScoreFn` is deferred with the HSDK 5 custom-test harness (default-path
`RejectsCustomWithoutScoreFn` covers the NoCustomScore reject).

**Status:** done (default path); custom Graph test deferred.

---

## 7. Algorithm text previously said MatX; implementation is custom CUDA

**Context.** Locked algorithm line mentioned MatX segmented reduction; this
slice uses custom CUDA atomics / compact (CUB/MatX segmented reduce can
replace later). `OPERATOR_API.md` now notes the custom-CUDA implementation.
Non-template kernels compile into `holoscan::radar_common_reduce` (public
link target); operators must not recompile `group_by_label.cu`.

**Status:** documented — keep open until a MatX/CUB rewrite lands.

---

## 8. Per-tick scratch `make_device_tensor` allocations

**Context.** Each compute tick allocates temporary device buffers
(`best_rows`, score-bit scratch, centroid sums, packed working sets, …)
rather than pooling/reusing across ticks like CFAR's dense-buffer pool.
Steady-state cost includes allocator traffic.

**Status:** accepted for now (push-review dismissed) — pool when a second
consumer or a hot production path makes it worth the complexity.

---

## 9. Representative `rep_idx` under ties is order-dependent (benign, not a defect)

**This is a known, by-design behavior, not a bug — no fix is required.** It is
documented here only because it affects bit-exact reproducibility.

**Context.** The per-cluster representative (medoid) is chosen by a device
reduction (argmin of distance-to-center) over the cluster's real detections.
When two or more detections are equidistant from the center they are all
equally-valid medoids, and which one wins the reduction depends on GPU thread /
reduction order. So under a tie, `rep_idx` — and thus a cluster's reported
representative range/velocity — can differ by one bin between runs of the
*same* binary on the same input. Cluster counts, membership, sizes, and
centroids are unaffected (the centroid is a sum, computed by `ClusterReportOp`,
not here); only the representative pick flips. Observed while diffing
example-app output (`fmcw_simo` / `fmcw_mimo`): one cluster's "rep range"
flipped by a single range bin across repeated runs.

**Why it's not a correctness issue.** The contract is "reduce each cluster to a
valid representative (medoid)"; under a tie there is more than one valid medoid,
and returning any of them is contract-correct. The representative is a
convenience pointer to a real cell for reporting / marking — a one-bin flip
between two equidistant detections is the same answer. Nothing downstream
(detection, clustering, the cluster's physical range/velocity story) changes.

**Only surfaces in:** bit-exact reproducibility (golden-file regression diffs,
pixel-identical screenshots). Mitigation there: compare the deterministic
fields and exclude the medoid `rep_*` values, or add a deterministic tie-break
(e.g. keep the lowest linear detection index) to the reduction.

**Status:** accepted / by-design — documented for reproducibility only.

---

## 10. Performance harness (4.x; Graph port deferred)

`performance/cpp/perf_detection_reduction.cpp` still uses the HSDK 4.x
Application helpers. Graph-port the perf binary (cold/steady + deferred
live-override timing per §1 / §2) in a follow-up, mirroring
`perf_clutter_filter`.

**Status:** open — deferred; not required to land the C++ HSDK 5 unit port.

---

## 11. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_detection_reduction_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.
