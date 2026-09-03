# FeatureAssemblyOp — open items for team review

Concrete questions / follow-ups for `FeatureAssemblyOp`. HSDK 5 / EA migration
items are indexed in [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).
Settled v1 pins (static `features`, explicit physical-versus-raw tokens,
Params-derived temporal join, rank-2 `detection_idx`, ctor-sized workspace,
error-flag sync before emit) live in the README.

---

## 1. Shared gather primitive with SnapshotGatherOp

**Context.** `OPERATOR_API.md` notes that FeatureAssembly and SnapshotGather
should share a per-detection gather primitive (scalars/lookups here; channel
vectors there).

**Current implementation.** FeatureAssembly has its own CUDA assemble kernel.
SnapshotGather is ported with its own gather path; no shared library helper
yet.

**Status:** staged — extract when both consumers are stable on HSDK 5.
