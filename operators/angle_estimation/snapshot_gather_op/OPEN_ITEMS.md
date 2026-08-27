# SnapshotGatherOp open items

Issues for `SnapshotGatherOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Graph mid-stream sticky `[live] snapshot_window` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”.

**Class:** HSDK 5 EA limitation (not a contract change). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares `Input<SnapshotWindowOverride> snapshot_window`
  (POD — see [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9;
  Params remain `std::vector<int64_t>`) and resolves sticky overrides in
  `compute()` via `resolve_sticky_override`.
- `snapshot_window` is omitted from `contract()` so unwired graphs compile
  (same pattern as ClutterFilter `method`).
- `SnapshotGatherOpTest.LiveSnapshotWindowStickyOverride` is `GTEST_SKIP`’d.
- `perf_snapshot_gather` measures an inline `snapshot_window` POD transition
  with the supported single-clock / exact-contract / single-partition shape.
- Params authoring covers window selection
  (`SnapshotWindowConfigurableFromParams`, window/edge-mode tests).
- Edge modes (`doppler_edge` / `range_edge`) remain static Params only.

**Next.** Restore sticky temporal policy + Graph sticky unit test on a newer
EA / GA (shared recipe with ClutterFilter §1).

**Status:** open for the sticky unit test; perf override timing is restored.

---

## 2. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_snapshot_gather_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.

---

## 3. `is_valid` mask dtype vs Holoscan DLPack transport (settled)

`is_valid` emits `uint8` (0/1), byte-identical to the mask the kernel already
writes and NumPy-visible — aligned with `CFARDetectorOp.detection_mask`
(settled — STATUS.md L1). The prior `kDLBool` port threw in Python on 4.3;
consumers still accept `bool` or `uint8`.

---

## 4. Multi-look snapshot axis transposition (fixed)

**Symptom.** Multi-look gather (`num_snapshots > 1`) recovered wrong angles
because the kernel wrote `(N, K, C)` while the contract / consumers index
`(N, C, K)`.

**Fix.** Write `out[c * num_snapshots + snap_slot]` (channel-major,
snapshot-minor). Gather tests index `(N, C, K)`.

**Status:** fixed. Aligns code to `OPERATOR_API.md`; no contract change.

---

## 5. Performance harness (closed)

`performance/cpp/perf_snapshot_gather.cpp` uses the HSDK 5 Graph dual-Tensor
source and reports cold / steady / inline `snapshot_window` override timing.

**Status:** closed.
