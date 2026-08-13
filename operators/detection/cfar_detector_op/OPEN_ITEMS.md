# CFARDetectorOp Open Items

Issues discovered during `CFARDetectorOp` implementation. HSDK 5 / EA
migration items are indexed in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream sticky `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky” for
`pfa` / `threshold_scale` / `os_rank` / `max_detections`; Performance →
inline live-override measurement (when Graph-ported).

**Class:** HSDK 5 EA limitation (not a contract change). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares sticky `Input<double> pfa` /
  `Input<double> threshold_scale` / `Input<int64_t> os_rank` /
  `Input<int64_t> max_detections` and resolves via `resolve_sticky_override`.
- Live ports are omitted from `contract()` so unwired graphs compile.
- `LiveThresholdScaleOverrideMidStreamAndSticky` /
  `LivePfaOverrideMidStreamAndSticky` /
  `LiveMaxDetectionsOverrideMidStreamAndSticky` are `GTEST_SKIP`’d.
- Params authoring covers selection (`ThresholdScaleConfigurableFromParams` /
  `PfaConfigurableFromParams` / `MaxDetectionsConfigurableFromParams`).

**Next.** Restore sticky temporal policy + Graph sticky unit tests on a newer
EA / GA (shared recipe with ClutterFilter §1).

**Status:** open — EA; do not treat as checklist-complete until closed.

---

## 2. `[live]` vector params are Params-only (unbounded / pair vectors)

**Blocks checklist:** Tests → live `guard_cells` / `training_cells` /
`nci_axes` / `alpha_lut` mid-stream sticky override.

**Class:** Durable HSDK 5 (`PublishedPayload` admission). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<…>>` is rejected at compile time.
`guard_cells` / `training_cells` are length-2 pairs (could become a fixed
POD later); `nci_axes` / `alpha_lut` are unbounded.

**Interim.** Those four stay on `CFARParams` only (`set_params` / YAML).
`LiveGuardCellsOverrideMidStreamAndSticky` /
`LiveTrainingCellsOverrideMidStreamAndSticky` /
`LiveNciAxesOverrideMidStreamAndSticky` are `GTEST_SKIP`’d. Params coverage
remains (`GuardCellsConfigurableFromParams` /
`TrainingCellsConfigurableFromParams`). POD scalar live Inputs (§1) still
declared.

**Next.** Prefer fixed POD for (doppler, range) pairs and `Input<Tensor>` (or
a durable schema) for `alpha_lut` / `nci_axes`; restore sticky live + Graph
tests.

**Status:** open — ported compute path; live vector ports deferred.

---

## 3. Exact-sized `detection_idx` — deferred device-side fast path

Exact-sized `detection_idx` forces a per-tick scalar detection-count
device→host sync before allocation (the host allocator needs
`num_detections` first). Exact sizing is the locked contract and is
**ratified**.

**Deferred optimization (not v1).** If profiling shows the per-tick D2H sync
bounds latency, add a fixed/max-capacity `detection_idx` + scalar
`num_detections` port as an **opt-in** device-side fast path — added
alongside, not replacing, the exact-size default.

**Status:** open — same as pre-port item 8.

---

## 4. SO-CFAR Monte Carlo Pfa test uses a relaxed ratio tolerance

**Context.** `test_cfar_detector_statistical.cpp` checks that realized
interior false-alarm rate matches configured `pfa`. SO-CFAR uses a
**relaxed `min_ratio_tol = 0.25`** (vs 0.15 for CA/GO/OS) because SO's
min-of-side noise estimate is more sensitive to spatial correlation across
neighboring CUTs that share training cells. See pre-port notes.

**Status:** open — statistical suite deferred on HSDK 5 Graph (§5); tolerance
question remains when that suite is re-enabled.

---

## 5. Variant / alpha_lut / statistical Graph test suites

**Blocks checklist:** none — Graph suites landed.

**Class:** Was our deferred work (sequencing); now closed on HSDK 5.

**Landed.** `test_cfar_detector_{go,so,os,alpha_lut}.cpp` rewritten to the
Graph five-out harness (`CFARParams` + `run_single_in_five_out_operator_graph_result`);
registered next to the CA suite with label `unit`. Live mid-stream sticky /
vector-live Graph cases remain `GTEST_SKIP` under HSDK5_PORT §1 / §9 (same
as CA). `test_cfar_detector_statistical.cpp` uses the same Graph harness
with multi-frame inputs and is registered with label `statistical` only
(`ctest -L statistical`; excluded from `ctest -L unit`).

**Status:** closed for phase-2 Graph unit coverage; §1/§9 skips stay until
phase 5.

---

## 6. Performance harness (4.x; Graph port deferred)

`performance/cpp/perf_cfar_detector.cpp` still uses the HSDK 4.x Application
helpers. Graph-port the perf binary (cold/steady + deferred live-override
timing per §1 / §2) in a follow-up, mirroring `perf_clutter_filter`.

**Status:** open — deferred; not required to land the C++ HSDK 5 unit port.

---

## 7. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_cfar_detector_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.

---

## 8. GO/SO/OS `detail::` α calibration helpers — Python white-box gap

**Status:** accepted / unclosable by design — leave the helpers in
`detail::`, keep white-box calibration regressions C++-only; do not treat
missing Python ports of those tests as unfinished bindings work. (Pre-port
item 9.)
