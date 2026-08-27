# SnapshotGatherOp open items

Issues for `SnapshotGatherOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Graph mid-stream `[live] snapshot_window` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”.

**Class:** Test / harness gap. The override wiring itself is settled
(L16 / L17) — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md) and
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Settled wiring.**

- `Input<SnapshotWindowOverride> snapshot_window` (POD — §9; Params remain
  `std::vector<int64_t>`) is registered unconditionally in `setup()` and
  joins the `OnAligned` data trigger with `Exact{}` +
  `on_missing SafeDefault{}` when `expect_snapshot_window` is set. The gate
  is what the `OnAligned` trigger requires: an unwired non-trigger `Exact{}`
  entry is still `TEMPORAL_INPUT_UNCONNECTED` there (§11), where an
  `OnEach`-triggered operator declares its override entries unconditionally
  and lets wiring decide.
- `sticky_snapshot_window` (default true) latches the last Exact match;
  false reverts to Params on a miss. `compute()` resolves through
  `resolve_expected_override`.
- The trigger stays `OnAligned{rd_maps, detection_idx}`; the override rides
  the same activation, so a wired application must stamp the frame's
  transmit time on the override as well as the cube.
- `perf_snapshot_gather` measures an inline `snapshot_window` POD transition
  with the supported single-clock / exact-contract / single-partition shape.

**Still owed.** `SnapshotGatherOpTest.LiveSnapshotWindowStickyOverride` is
`GTEST_SKIP`’d — no Graph test drives a mid-stream override here yet (the
`FMCWRangeProcessorOp` `nfft` Exact tests are the worked example). Params
authoring covers window selection
(`SnapshotWindowConfigurableFromParams`, window/edge-mode tests). Edge modes
(`doppler_edge` / `range_edge`) remain static Params only.

**Status:** open — do not treat the mid-stream/sticky test clause as
checklist-complete until a Graph test drives it. Perf override timing is
restored (§8).

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
