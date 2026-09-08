# BeamscanEstimatorOp open items

Issues for `BeamscanEstimatorOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Graph mid-stream `[live] peak_interpolation` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + revert-to-Params”;
Performance → inline live-override measurement (when Graph-ported).

**Wiring (settled — L16 / L17).** `contract()` unconditionally joins
`Input<PeakInterpolation> peak_interpolation` to the data trigger with
`Exact{}` and `on_missing SafeDefault{}`, and `compute()` resolves it with
`resolve_live_override`. The application enables the override by wiring the
port and disables it by leaving it unwired.
A matched override applies to that tick only; a tick with no match uses
the Params value. The application stamps one transmit
time on both the override and the snapshot — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Still open (EA test harness).**

- `BeamscanEstimatorOpTest.LivePeakInterpolationAppliesForOneTick` is
  `GTEST_SKIP`’d; the descriptor / scan-grid override tests do run
  mid-stream, so the gap is this enum port specifically.
- Params authoring covers selection
  (`PeakInterpolationConfigurableFromParams`).

**Next.** Add the Graph mid-stream + revert-to-Params test, modeled on
`FMCWRangeProcessorOp`’s `nfft` Exact cases.

**Status:** open — test coverage only; the wiring decision is settled.

---

## 3. Optional `is_valid` Graph wiring

**Blocks checklist:** Tests → optional `is_valid` connected Graph paths.

**Class:** HSDK 5 EA gap — no optional input
([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11).

**What fails.**

- Omitting `is_valid` from `contract()` lets unwired graphs compile
  (`TEMPORAL_INPUT_UNCONNECTED` otherwise; `SafeDefault` alone is not
  enough on EA).
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
