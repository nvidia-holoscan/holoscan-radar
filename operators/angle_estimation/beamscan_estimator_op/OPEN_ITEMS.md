# BeamscanEstimatorOp open items

## 1. `AngleConvention::kCustom` boundary mapping

**Context.** `OPERATOR_API.md` documents a `custom` preset with explicit axis
fields. `boundary_to_canonical` / `canonical_to_boundary` threw
`invalid_argument` for `preset=custom` until the team defined the mapping rules.

**Resolved (L11, 2026-07-22).** Implemented the generic mapping (chosen over
deferring): `custom` builds an orthonormal frame from the signed-axis fields and
maps `(az, el)` → `cos(el)cos(az)·boresight + cos(el)sin(az)·azimuth +
sin(el)·elevation` (with the `zenith` polar step), inverted by projection. The
three axes must be distinct principal axes (mutually orthogonal) — a degenerate
config throws `invalid_argument` where the steering table is built. Every named
preset is a specific `custom` config, which is the correctness anchor:
`test_angle_convention.cpp` asserts `custom` reproduces each preset (both
directions), round-trips, and rejects degenerate axes. Shared with
`PhaseShiftBeamformerOp` §1. See STATUS.md → Settled decisions (L11).

**Status:** resolved — `custom` implemented + tested; contract mapping stated.

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

## 4. Steering-weight build moved to GPU (landed)

The `O(num_beams·num_channels)` phase evaluation now runs in a kernel via the
shared `build_steering_weights_device`
(`radar_common/dsp/steering_weights_device.hpp`,
`holoscan::radar_common_steering`), replacing the serial host trig loop that
dominated the `[live]` override spike at scale. Beamscan requests
`BeamOrder::kElMajor` + `SteeringLayout::kChannelMajor` + `conjugate`, so the
builder emits `Wᶜ` `(num_channels, num_beams)` with beams el-major — exactly
the reduce kernel's beam decode and the `X · Wᶜ` GEMM operand, no host reorder.
See `phase_shift_beamformer_op/OPEN_ITEMS.md` §5.

**Status:** landed — shared with `PhaseShiftBeamformerOp`.
