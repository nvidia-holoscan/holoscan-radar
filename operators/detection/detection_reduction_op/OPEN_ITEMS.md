# DetectionReductionOp Open Items

Issues discovered during `DetectionReductionOp` implementation. HSDK 5 / EA
migration items are indexed in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream sticky `[live] representative` / `score_method` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”;
Performance → inline live-override measurement (when Graph-ported).

**Class:** HSDK 5 EA limitation (not a contract change). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares sticky `Input<RepresentativePolicy> representative` /
  `Input<ScoreMethod> score_method` and resolves via `resolve_sticky_override`.
- Both live ports are omitted from `contract()` so unwired graphs compile
  (same pattern as ClutterFilter `method` / ClusterDBSCAN scalars).
- `LiveRepresentativeStickyOverride` / `LiveScoreMethodStickyOverride` are
  `GTEST_SKIP`’d.
- Params authoring covers selection (`RepresentativeConfigurableFromParams` /
  `ScoreMethodConfigurableFromParams`).

**Next.** Restore sticky temporal policy + Graph sticky unit tests on a newer
EA / GA (shared recipe with ClutterFilter §1).

**Status:** open — EA; do not treat as checklist-complete until closed.

---

## 2. `[live] feature_weights` is Params-only (unbounded vector)

**Blocks checklist:** Tests → live `feature_weights` mid-stream sticky override.

**Class:** Durable HSDK 5 (`PublishedPayload` admission). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<double>>` is rejected at compile time.
`feature_weights` length equals `num_features` from the caller-assembled
`points` matrix and is unbounded, so a fixed POD like
`SnapshotWindowOverride` does not fit.

**Interim.** `feature_weights` stays on `DetectionReductionParams` only
(`set_params` / YAML). `LiveFeatureWeightsStickyOverride` is `GTEST_SKIP`’d.
Params coverage remains (`FeatureWeightsConfigurableFromParams`). Scalar /
enum live ports (`representative` / `score_method`) still declare sticky
Inputs (§1).

**Next.** Prefer `Input<Tensor>` (or a durable schema / bounded POD once a
module max-feature count is pinned) and restore sticky live + Graph tests.

**Status:** open — ported compute path; live vector port deferred.

---

## 3. Optional `points` is Exact in the temporal contract (EA)

**Blocks checklist:** Tests → unconnected `points` with `representative=first`;
`RejectsPeakWithoutPoints` Graph path.

**Class:** HSDK 5 EA limitation (same family as FeatureAssembly optional
gathers / Beamscan `is_valid`). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 / §7.

**What fails.**

- `SafeDefault` / `LatestWithin` on optional Tensor `points` still yields
  `TEMPORAL_INPUT_UNCONNECTED` when left unwired.
- Omitting `points` from `contract()` while wiring it makes the sample
  unavailable in `compute()` (peak/medoid Graph paths fail).

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

## 4. Dense label assumption

**Context.** Implementation treats non-noise labels as integers in
`[0, max_label]` and allocates `max_label+1` slots (cap 2^20). Gaps are
skipped when compacting. Huge sparse label ids are rejected.

**Status:** open — matches expected `ClusterDBSCANOp` output; revisit if
labels are remapped differently.

---

## 5. Custom `ScoreFn` template path

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

## 6. Algorithm text previously said MatX; implementation is custom CUDA

**Context.** Locked algorithm line mentioned MatX segmented reduction; this
slice uses custom CUDA atomics / compact (CUB/MatX segmented reduce can
replace later). `OPERATOR_API.md` now notes the custom-CUDA implementation.
Non-template kernels compile into `holoscan::radar_common_reduce` (public
link target); operators must not recompile `group_by_label.cu`.

**Status:** documented — keep open until a MatX/CUB rewrite lands.

---

## 7. Per-tick scratch `make_device_tensor` allocations

**Context.** Each compute tick allocates temporary device buffers
(`best_rows`, score-bit scratch, centroid sums, packed working sets, …)
rather than pooling/reusing across ticks like CFAR's dense-buffer pool.
Steady-state cost includes allocator traffic.

**Status:** accepted for now (push-review dismissed) — pool when a second
consumer or a hot production path makes it worth the complexity.

---

## 8. Representative `rep_idx` under ties is order-dependent (benign, not a defect)

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

## 9. Performance harness (4.x; Graph port deferred)

`performance/cpp/perf_detection_reduction.cpp` still uses the HSDK 4.x
Application helpers. Graph-port the perf binary (cold/steady + deferred
live-override timing per §1 / §2) in a follow-up, mirroring
`perf_clutter_filter`.

**Status:** open — deferred; not required to land the C++ HSDK 5 unit port.

---

## 10. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_detection_reduction_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.
