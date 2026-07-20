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

## 6. Representative `rep_idx` medoid tie-break is run-to-run non-deterministic

**Context.** The per-cluster representative (medoid) is chosen by a device
reduction over the cluster's real detections. When two detections are
equidistant from the centroid, the winner depends on GPU reduction / thread
order, so `rep_idx` — and thus a cluster's reported representative
range/velocity — can differ by one bin between runs of the *same* binary on
the same input. Cluster counts, point counts, and centroids are stable; only
the representative selection flips. Observed while diffing example-app output
(`fmcw_simo` / `fmcw_mimo`): a single cluster's "rep range" flipped by one
range bin across repeated runs.

**Status:** accepted for the representative field — it does not affect
detection or clustering results. When bit-exact output is needed (e.g.
regression diffing of the example apps), compare the deterministic fields and
exclude the medoid `rep_*` values, or add a deterministic tie-break (e.g.
lowest linear detection index) to the reduction.
