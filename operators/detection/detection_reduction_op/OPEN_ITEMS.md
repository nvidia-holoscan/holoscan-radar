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
`custom` at runtime; callers with a real functor must explicitly
instantiate the op in a CUDA TU (see `detection_reduction_op_impl.cuh`).

**Status:** done.

---

## 4. Algorithm text previously said MatX; implementation is custom CUDA

**Context.** Locked algorithm line mentioned MatX segmented reduction; this
slice uses custom CUDA atomics / compact (CUB/MatX segmented reduce can
replace later). `OPERATOR_API.md` now notes the custom-CUDA implementation.

**Status:** documented — keep open until a MatX/CUB rewrite lands.

---

## 5. Per-tick scratch `make_device_tensor` allocations

**Context.** Each compute tick allocates temporary device buffers
(`best_rows`, score-bit scratch, centroid sums, packed working sets, …)
rather than pooling/reusing across ticks like CFAR's dense-buffer pool.
Steady-state cost includes allocator traffic.

**Status:** accepted for now (push-review dismissed) — pool when a second
consumer or a hot production path makes it worth the complexity.
