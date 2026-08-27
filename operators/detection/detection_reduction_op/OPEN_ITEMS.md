# DetectionReductionOp Open Items

Issues discovered during `DetectionReductionOp` implementation. HSDK 5 / EA
migration items are indexed in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream `[live] representative` / `score_method` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”;
Performance → inline live-override measurement (when Graph-ported).

**Class:** Wiring settled (L16 / L17, 2026-08-21); Graph test coverage still
deferred on EA. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 and
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Decision.** `representative` / `score_method` / `feature_weights` join the
`OnAligned{labels, detection_idx, points}` trigger with `Exact{}` and
`on_missing SafeDefault{}`, each gated by `expect_<name>` (default false).
The gate is required here and not module-wide: under an `OnAligned` trigger
the SDK still rejects an unwired non-trigger `Exact{}` entry with
`TEMPORAL_INPUT_UNCONNECTED`
([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11), so
`expect_*=true` requires the matching `add_flow` and `expect_*=false` omits
the entry and forbids wiring; wiring without the flag is
`RUNTIME_OPERATOR_REPLAY_MISMATCH`. `sticky_<name>` (default true) latches
the last Exact match; false reverts to Params on a miss. The application
stamps the frame's transmit time on both the override and the aligned
tensors.

**Still deferred.**

- `LiveRepresentativeStickyOverride` / `LiveScoreMethodStickyOverride` are
  still `GTEST_SKIP`’d — the Graph harness for a mid-stream enum override
  alongside a three-way `OnAligned` join is not written yet.
- Params authoring covers selection (`RepresentativeConfigurableFromParams` /
  `ScoreMethodConfigurableFromParams`).

**Next.** Port the Graph live-override harness (the pattern
`test_fmcw_range_processor.cpp` uses for `nfft`) to this triple-in fixture
and unskip.

**Status:** open — test coverage only; the contract wiring is settled.

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
converted to `std::vector<double>` before it reaches sticky state. Like the
enum lives it is gated by `expect_feature_weights` (§1), and that flag also
makes `max_features` a required extent. Graph coverage:
`LiveFeatureWeightsExactMatchChangesSelection`,
`LiveFeatureWeightsSkewedCaptureMissesExact`,
`LiveFeatureWeightsStickyKeepsLastExactMatch`,
`LiveFeatureWeightsEphemeralRevertsToParams`,
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
- The `expect_*` recipe that settled the `[live]` overrides (§1) does not
  transfer: `points` is a trigger port, so gating its contract entry on a
  flag would also change what the operator triggers on, not just whether an
  override is selected.

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
