# ClutterFilterOp Open Items

Issues for `ClutterFilterOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Graph mid-stream `[live] method` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”;
Performance → inline live-override measurement.

**Class:** HSDK 5 EA limitation (not a contract change). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Wiring (settled, L16 / L17).** `Input<ClutterFilterMethod> method` joins
the data trigger with `Exact{}` and `on_missing SafeDefault{}` unconditionally,
so the override's `capture_time` must match the cube's. Wiring the port is the
application's only obligation: an unwired entry reads as no override for that
tick. `sticky_method` (default true) latches the last Exact match; false
reverts to Params on a miss. `compute()` resolves via
`resolve_live_override`. See
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Still deferred.**

- `ClutterFilterOpTest.LiveMethodOverrideMidStreamAndSticky` is
  `GTEST_SKIP`’d; Params authoring covers method selection
  (`MethodConfigurableFromParams`).
- `perf_clutter_filter` measures setup / cold / steady only; notes
  `live_override=deferred_ea`.

**Next.** Add the Graph mid-stream unit test + `override_ms` in the perf
harness on a newer EA / GA.

**Status:** open — EA harness gap; do not treat as checklist-complete until
closed.

---

## 2. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_clutter_filter_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 pilot.

---

## 3. Physical output port rename (`range_profiles_out`)

**Class:** Durable HSDK 5 (unique in/out names). Documented in
`OPERATOR_API.md` and [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md)
§2.

**Status:** accepted interim for the EA port; revisit module-wide naming
before dropping `[EA HSDK5 ONLY]` fences.
