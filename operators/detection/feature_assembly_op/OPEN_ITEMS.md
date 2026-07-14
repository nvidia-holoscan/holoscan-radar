# FeatureAssemblyOp Open Items

Issues discovered during `FeatureAssemblyOp` implementation that need a team
decision or are staged follow-ups. Keep operator-specific items here.

---

## 1. Shared gather primitive with SnapshotGatherOp

**Context.** `OPERATOR_API.md` notes that FeatureAssembly and SnapshotGather
should share a per-detection gather primitive (scalars/lookups here; channel
vectors there).

**Current implementation.** FeatureAssembly has its own CUDA assemble kernel.
No shared library helper yet.

**Status:** staged — extract when `SnapshotGatherOp` lands so the shared
API has a second consumer.

---

## 2. Dense `points` ping-pong

**Context.** `OPERATOR_DEVELOPMENT.md` recommends ping-pong for dense emitted
buffers. `points` is `(num_detections, num_features)` — typically small
(sparse detections).

**Status:** deferred — measure with `performance/cpp/perf_feature_assembly`
across the detection-count ladder before adding lifetime state. Prefer
evidence over guessing. Note: assemble currently stream-synchronizes each
tick to surface device-side OOB index errors; that sync dominates small-N
steady time and is separate from the `points` alloc question.

---

## 3. Algorithm text vs implementation (MatX → custom CUDA)

**Context.** The locked algorithm note historically said “Pure gather /
1-D lookup (MatX)”. The implementation uses a fused custom CUDA assemble
kernel (sparse irregular gathers are a poor MatX fit), matching CFAR’s
“benchmark; fuse when MatX is a poor fit” lesson.

**Status:** resolved in contract — `OPERATOR_API.md` FeatureAssembly
algorithm / carried-over notes now say custom CUDA. Keep this section as
a breadcrumb for SnapshotGather when sharing the gather primitive.
