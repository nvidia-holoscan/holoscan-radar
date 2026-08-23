# FeatureAssemblyOp — open items for team review

Concrete questions / follow-ups for `FeatureAssemblyOp`. HSDK 5 / EA migration
items are indexed in [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Optional gather Inputs omitted from temporal contract

**Blocks checklist:** Tests → optional `range_bins` / `velocity_bins` /
`detection_score` / `aoa_estimates` / `az_angles` / `el_angles` connected
Graph paths.

**Class:** HSDK 5 EA gap — no optional input (same family as Beamscan optional
`is_valid`). See [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11.

**What fails.**

- Putting optional Tensor Inputs in `contract()` with `SafeDefault` /
  `LatestWithin` yields `TEMPORAL_INPUT_UNCONNECTED` when left unwired
  (apps routinely omit unused gather sources); `on_missing` governs a
  declared *and connected* input.
- Wiring those ports via DualIO / multi-source while they are omitted from
  `contract()` yields `RUNTIME_OPERATOR_REPLAY_MISMATCH` on EA.
- The `expect_*` recipe that settled the module's POD `[live]` overrides
  does not transfer: this operator has no `[live]` parameters, and the
  gather ports' presence already follows from the static `features` list,
  so a per-port flag would state the same fact a third time.

**Interim.**

- Optional gather ports stay registered in `setup()`; `compute()` uses
  `receive_optional_value`.
- Ports are omitted from `contract()` so unconnected graphs compile
  (trigger = `OnEach{detection_idx}` only).
- Graph unit coverage for optional-connected paths is deferred
  (`OptionalGatherGraphWiringDeferred` skipped). Stage / kernel launches in
  `test_feature_assembly.cu` cover physical lookup, score gather, AoA,
  dense az/el, and OOB error codes.

**Next.** Restore optional-Tensor temporal recipe (SafeDefault that admits
unconnected ports, or OnAligned when wired) on a newer EA / GA; unskip Graph
wiring tests.

**Status:** open — EA; do not treat as checklist-complete until closed.

---

## 2. Shared gather primitive with SnapshotGatherOp

**Context.** `OPERATOR_API.md` notes that FeatureAssembly and SnapshotGather
should share a per-detection gather primitive (scalars/lookups here; channel
vectors there).

**Current implementation.** FeatureAssembly has its own CUDA assemble kernel.
SnapshotGather is ported with its own gather path; no shared library helper
yet.

**Status:** staged — extract when both consumers are stable on HSDK 5.

---

## 3. Dense `points` ping-pong

**Context.** `OPERATOR_DEVELOPMENT.md` recommends ping-pong for dense emitted
buffers. `points` is `(num_detections, num_features)` — typically small
(sparse detections).

**Status:** deferred — measure with `performance/cpp/perf_feature_assembly`
across the detection-count ladder before adding lifetime state. Prefer
evidence over guessing. Note: assemble currently stream-synchronizes each
tick to surface device-side OOB index errors; that sync dominates small-N
steady time and is separate from the `points` alloc question.

---

## 4. Performance harness (4.x; Graph port deferred)

`performance/cpp/perf_feature_assembly.cpp` still uses the HSDK 4.x
Application helpers. Graph-port the perf binary (cold/steady) in a follow-up,
mirroring `perf_clutter_filter`. Until then, a full `cmake --build` of
`perf_feature_assembly` will fail against the HSDK 5 operator header — build
/ test the unit target only.

**Status:** open — deferred; not required to land the C++ HSDK 5 unit port.

---

## 5. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_feature_assembly_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.
