# CFARDetectorOp Open Items

Issues discovered during `CFARDetectorOp` implementation. HSDK 5 / EA
migration items are indexed in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky” for
`pfa` / `threshold_scale` / `os_rank` / `max_detections`.

**Class:** Wiring settled (L16 / L17, 2026-08-21); Graph test coverage still
deferred on EA. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 and
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Decision.** `pfa` / `threshold_scale` / `os_rank` / `max_detections` join
the `rd_maps` trigger with `Exact{}` and `on_missing SafeDefault{}`, and the
entries are unconditional. `sticky_<name>` (default true) latches the last
Exact match; false reverts to Params on a miss. The application stamps the
frame's transmit time on both the override and the cube. Wiring a port is
the only thing that enables its override: an unwired entry compiles
(warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`) and reads as no
override, while wiring a port absent from `contract()` is
`RUNTIME_OPERATOR_REPLAY_MISMATCH`.

**Still deferred.**

- `LiveThresholdScaleOverrideMidStreamAndSticky` /
  `LivePfaOverrideMidStreamAndSticky` /
  `LiveMaxDetectionsOverrideMidStreamAndSticky` (and the OS suite's
  `LiveOsRankOverrideMidStreamAndSticky`) are still `GTEST_SKIP`’d — the
  Graph harness for a mid-stream POD override on this five-output operator
  is not written yet.
- `perf_cfar_detector` measures an inline `threshold_scale` transition with
  the supported single-clock / exact-contract / single-partition shape.
- Params authoring covers selection (`ThresholdScaleConfigurableFromParams` /
  `PfaConfigurableFromParams` / `MaxDetectionsConfigurableFromParams`).

**Next.** Port the Graph live-override harness (the pattern
`test_fmcw_range_processor.cpp` uses for `nfft`) to the five-out CFAR
fixture and unskip.

**Status:** open — test coverage only; the contract wiring is settled.
Isolated `pfa` / `os_rank` / `max_detections` perf is still owed;
`threshold_scale` perf timing is restored (§8).

---

## 2. `[live]` vector params ride PODs and config Tensors

**Blocks checklist:** none — the vector lives are wired.

**Class:** Durable HSDK 5 (`PublishedPayload` admission). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<…>>` is rejected at compile time, so the
contract's vector `[live]` tags cannot map to the logical Params type
directly.

**Mapping.** `guard_cells` / `training_cells` are length-2 pairs and ride
padding-free `CfarCellWindowOverride` PODs converted by `from_wire`.
`nci_axes` / `alpha_lut` ride rank-1 pageable-host config Tensors validated
on arrival (rank, host memory, contiguity, dtype, and a per-variant maximum
extent). Sticky state owns the converted `std::vector`, so no wire storage
leaks into operator state. Graph coverage:
`LiveGuardCellsOverrideMidStreamAndSticky`,
`LiveAlphaLutOverrideStickyAndClearing`,
`RejectsMalformedLiveNciAxesTensor`, `RejectsMalformedLiveAlphaLut`.

**Next.** Nothing owed unless an owning-container payload family lands, at
which point the POD / config-Tensor wrappers can collapse back to the
logical vector types.

**Status:** settled — the payload form differs from the logical Params type;
behavior conforms.

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

## 6. Performance harness (closed for Graph + scalar timing)

`performance/cpp/perf_cfar_detector.cpp` uses the HSDK 5 Graph five-output
path and reports cold / steady / inline `threshold_scale` override timing.
Other scalar live inputs, and the POD / config-Tensor vector lives, still
need isolated rows.

**Status:** closed for the Graph port and representative scalar override.

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
