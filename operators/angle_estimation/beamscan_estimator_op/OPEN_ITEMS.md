# BeamscanEstimatorOp open items

## 1. `AngleConvention::kCustom` boundary mapping

**Context.** `OPERATOR_API.md` documents a `custom` preset with explicit axis
fields. `boundary_to_canonical` / `canonical_to_boundary` throw
`invalid_argument` for `preset=custom` until the team defines the mapping rules.

**Status:** open — deferred until a consumer needs non-preset conventions.

## 2. `is_valid` bool dtype vs Holoscan 4.3 transport

**Context.** Beamscan accepts optional `is_valid` with `kDLBool`, aligned with
`SnapshotGatherOp` and `CFARDetectorOp` `detection_mask`. See
`snapshot_gather_op/OPEN_ITEMS.md` §1 for the HSDK 4.3 transport caveat.

**Status:** open — team-wide bool-mask policy. `check_is_valid_mask` copies
device masks to the host before inspection (required for graph-wired masks).
`AcceptsValidIsValidMaskOn3DSnapshot` / `RejectsInvalidIsValidMaskOn3DSnapshot`
graph-test the wired exclusion path via a relay operator. Python consumers of
`kDLBool` tensors remain at risk until HSDK 5 (see SnapshotGather OPEN_ITEMS).

## 3. Performance harness (landed)

`performance/cpp/perf_beamscan_estimator.cpp` characterizes steady-state Bartlett
scan cost and inline `[live] az_angles` override latency (finer grid → steering
rebuild). The `perf_beamscan_estimator_smoke` ctest (label `performance`) runs a
tiny subset and verifies a report is emitted.
