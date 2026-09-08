# PhaseShiftBeamformerOp open items

Issues for `PhaseShiftBeamformerOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 2. Dense RDA-cube memory footprint

**Context.** The output cube is `(num_az, num_el, num_doppler,
num_range_bins)` complex, allocated per tick (the output is owned downstream).
For a fine scan grid over a full RD map this is large (e.g. a 61×31 grid over
128×256 cells is ~250 MB in complex64), which is inherent to the dense path —
the contract notes a dense MVDR/MUSIC cube would be prohibitive, so conventional
beamforming is the only dense estimator. No tiling / streaming over the beam
grid is implemented; callers size the scan grid and RD extent sensibly (the
caller's footgun, per the module's `[live]` philosophy).

**Status:** open — revisit if a real deployment needs a beam-grid tiling /
chunked-emit variant to bound peak memory.

---

## 3. Required `angle_convention` has no unset representation

**Contract.** `AngleConvention` is required with no default on every operator
that consumes or emits angles.

**Current representation.** `PhaseShiftBeamformerParams` contains an
`AngleConvention` value rather than an optional or a type with an unset
sentinel. Default construction therefore produces the valid
`radar_boresight` convention, and `compute()` cannot distinguish omission from
an explicit request for that preset. This is shared by the angle-estimator
Params types, not specific to this operator.

**Decision needed.** Choose a module-wide required-parameter representation:
an optional Params field, an explicit unset enum value, or a schema-level
presence mechanism. Do not make one estimator interpret the shared descriptor
differently.

**Status:** open — module-wide contract representation.

---

## 4. Inline live-override performance timing

The C++ Graph tests cover all four live payload families and their one-tick
revert behavior. The performance harness still reports only cold and steady
Params-driven runs; it does not co-emit a host-Tensor or wire-schema override
mid-session and populate `override_ms` / `override_delta_ms`.

**Status:** open — ordinary performance-harness work, no SDK blocker.
