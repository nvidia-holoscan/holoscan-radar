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
evidence over guessing.
