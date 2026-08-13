# BeamscanEstimatorOp open items

Issues for `BeamscanEstimatorOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Graph mid-stream sticky `[live] peak_interpolation` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”;
Performance → inline live-override measurement (when Graph-ported).

**Class:** HSDK 5 EA limitation (not a contract change). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares `Input<PeakInterpolation> peak_interpolation` and
  resolves sticky overrides in `compute()` via `resolve_sticky_override`.
- `peak_interpolation` is omitted from `contract()` so unwired graphs compile
  (same pattern as ClutterFilter `method` / SnapshotGather `snapshot_window`).
- `BeamscanEstimatorOpTest.LivePeakInterpolationStickyOverride` is
  `GTEST_SKIP`’d.
- Params authoring covers selection
  (`PeakInterpolationConfigurableFromParams`).

**Next.** Restore sticky temporal policy + Graph sticky unit test on a newer
EA / GA (shared recipe with ClutterFilter §1).

**Status:** open — EA; do not treat as checklist-complete until closed.

---

## 2. Non-POD / variable-length `[live]` params are Params-only

**Blocks checklist:** Tests → live `az_angles` / `el_angles` /
`array_geometry` / `array_taper` mid-stream sticky overrides.

**Class:** Durable HSDK 5 (`PublishedPayload` admission) + our deferred
work. Same recipe as `PhaseShiftBeamformerOp` — see
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<double>>`, `Input<ArrayGeometry>`, and
`Input<WindowSpec>` are not valid PublishedPayloads (vectors are owning
containers; descriptors embed vectors / optionals and are not ChannelPod /
schema-identity channel types). Unbounded scan grids cannot use a
fixed-size POD like `SnapshotWindowOverride`.

**Interim.** Those parameters remain on `BeamscanParams` only (set via
`set_params`). Live Graph override tests are `GTEST_SKIP`’d. Params
coverage remains (`ArrayTaperConfigurableFromParams`, scan-grid / DOA
tests). `angle_convention` stays static (Params only) per OPERATOR_API.md.
Enum live `peak_interpolation` still has a sticky Input (§1).

**Next.** Prefer Tensor live ports for angle grids and/or ChannelPod /
schema wrappers for descriptors once a module-wide recipe lands (§9);
then restore sticky live ports + Graph tests.

**Status:** open — ported compute path; live descriptor/vector ports
deferred.

---

## 3. Optional `is_valid` Graph wiring

**Blocks checklist:** Tests → optional `is_valid` connected Graph paths.

**Class:** HSDK 5 EA limitation (related to §1 temporal / binding gaps).

**What fails.**

- Omitting `is_valid` from `contract()` lets unwired graphs compile
  (`TEMPORAL_INPUT_UNCONNECTED` otherwise; `SafeDefault` alone is not
  enough on EA — same as ClutterFilter live ports).
- Wiring `is_valid` via DualIO while it is omitted from the contract
  yields `RUNTIME_OPERATOR_REPLAY_MISMATCH`.

**Interim.** Port remains registered in `setup()`; `compute()` uses
`receive_optional_value`. Graph tests
`AcceptsValidIsValidMaskOn3DSnapshot` /
`RejectsInvalidIsValidMaskOn3DSnapshot` are `GTEST_SKIP`’d. Direct
`CheckIsValidMask*` unit tests still cover mask validation.

**Next.** Restore optional-Tensor temporal recipe (SafeDefault that
admits unconnected ports, or OnAligned when wired) on a newer EA / GA.

**Status:** open — EA.

---

## 4. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_beamscan_estimator_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.

---

## 5. Performance harness (4.x; Graph port deferred)

`performance/cpp/perf_beamscan_estimator.cpp` still uses the HSDK 4.x
Application helpers. Graph-port the perf binary (cold/steady + deferred
live-override timing per §1) in a follow-up, mirroring `perf_clutter_filter`.
Until then, a full `cmake --build` of `perf_beamscan_estimator` will fail
against the HSDK 5 operator header — build/test the unit target only.
