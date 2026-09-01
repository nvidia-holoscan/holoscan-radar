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

**Blocks checklist:** Tests → “[live] overrides: mid-stream + revert-to-Params”;
Performance → inline live-override measurement.

**Class:** was an HSDK 5 EA limitation (never a contract change). `method` is
enum-typed, and an `enum class` Graph edge failed runtime binding until the SDK
fix on pin `f0cb828d4` — see
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §12.

**Wiring (settled, L16 / L17).** `Input<ClutterFilterMethod> method` joins
the data trigger with `Exact{}` and `on_missing SafeDefault{}` unconditionally,
so the override's `capture_time` must match the cube's. Wiring the port is the
application's only obligation: an unwired entry reads as no override for that
tick. A tick with no match uses the Params method. `compute()` resolves via
`resolve_live_override`. See
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Unit coverage landed (2026-08-28).** With the enum edge fixed,
`LiveMethodOverrideAppliesForOneTick` and
`LiveMethodRevertsToParamsAfterMiss` are real tests. They read the method from
the output's chirp count — mean subtraction preserves all `kChirps` rows,
single-delay drops one, double-delay drops two — so the assertion names the
method without inspecting parameters: an override mid-stream changes the shape
for that tick, and the tick after a present override reverts to the Params
method.

**Still deferred.**

- `perf_clutter_filter` measures setup / cold / steady only and notes
  `live_override=deferred_ea`. The blocker is gone, so this is now ordinary
  harness work: `perf_graph.hpp` still skips enum sidebands
  ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §8).

**Next.** Add `override_ms` for `method` in the perf harness.

**Status:** open only for perf `override_ms`; the unit-test gap is closed.

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
