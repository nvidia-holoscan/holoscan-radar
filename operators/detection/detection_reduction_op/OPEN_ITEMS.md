# DetectionReductionOp Open Items

Issues discovered during `DetectionReductionOp` implementation.

---

## 1. Exact-sized `rep_idx` forces a scalar device→host sync

**Context.** Contract requires `rep_idx` sized to `num_clusters`. Host
allocators need the count before `make_device_tensor`, so compute syncs a
scalar cluster count (and max-label) each tick.

**Status:** open — same class of cost as CFAR exact-sized `detection_idx`.

---

## 2. Dense label assumption

**Context.** Implementation treats non-noise labels as integers in
`[0, max_label]` and allocates `max_label+1` slots (cap 2^20). Gaps are
skipped when compacting. Huge sparse label ids are rejected.

**Status:** open — matches expected `ClusterDBSCANOp` output; revisit if
labels are remapped differently.

---

## 3. Custom `ScoreFn` template path

**Context.** Contract `score_method=custom` uses a compile-time device
functor (`DetectionReductionOp<T, ScoreFn>`). Implemented: custom peak
kernels are header templates; default `ScoreFn = NoCustomScore` rejects
`custom` only when `representative=peak`; callers with a real functor must
explicitly instantiate the op in a CUDA TU (`detection_reduction_op_impl.cuh`
is installed with the operator headers).

**Status:** done.

---

## 4. Algorithm text previously said MatX; implementation is custom CUDA

**Context.** Locked algorithm line mentioned MatX segmented reduction; this
slice uses custom CUDA atomics / compact (CUB/MatX segmented reduce can
replace later). `OPERATOR_API.md` now notes the custom-CUDA implementation.
Non-template kernels compile into `holoscan::radar_common_reduce` (public
link target); operators must not recompile `group_by_label.cu`.

**Status:** documented — keep open until a MatX/CUB rewrite lands.

---

## 5. Per-tick scratch `make_device_tensor` allocations

**Context.** Each compute tick allocates temporary device buffers
(`best_rows`, score-bit scratch, centroid sums, packed working sets, …)
rather than pooling/reusing across ticks like CFAR's dense-buffer pool.
Steady-state cost includes allocator traffic.

**Status:** accepted for now (push-review dismissed) — pool when a second
consumer or a hot production path makes it worth the complexity.

---

## 6. Representative `rep_idx` under ties is order-dependent (benign, not a defect)

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
