# PhaseShiftBeamformerOp open items

Issues for `PhaseShiftBeamformerOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. `array_geometry` overrides have no in-module producer

**Blocks checklist:** Performance → inline live-override measurement (when
Graph-ported). The ports and their mid-stream Graph coverage are in place.

**Contract.** `OPERATOR_API.md` marks `array_geometry`, `az_angles`,
`el_angles`, and `array_taper` as sticky `[live]`; `angle_convention`
remains STATIC.

**Ports (settled — L16 / L17 / L18).** `contract()` unconditionally joins
`Input<WireArrayGeometry> array_geometry`, `Input<WireWindowSpec>
array_taper`, and the rank-1 pageable-host float64 config Tensors
`az_angles` / `el_angles` to the data trigger with `Exact{}` +
`on_missing SafeDefault{}`; the application enables an override by wiring
the port, and `sticky_*` (default true) latches the last match —
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).
`compute()` resolves each with `resolve_live_override_via`, validating the
Tensors against `max_az` / `max_el` from `MaxTensorExtents`. Logical Params
types are unchanged; the `ConfigCache` steering rebuild keys on the
resolved geometry / grid / taper either way.

Graph coverage: `LiveAzGridStickyOverride`, `LiveElGridStickyOverride`,
`LiveArrayGeometryStickyOverride`, `LiveArrayTaperStickyOverride`,
`LiveOverridesRejectMalformedAndBounds`. Params authoring still covers
selection (`ScanGridConfigurableFromParams`,
`ArrayTaperConfigurableFromParams`).

**Still open.** `VirtualArrayAssemblyOp` / `SubarraySelectionOp` emit
geometry as a packed float64 `(N + 1, 3)` Tensor (§10), not the
wire-schema type this port consumes (§9). The representations differ, so a
runtime virtual geometry cannot drive this override directly — it has to
come from an app-side wire-schema producer. The generated wire objects are
also C++-only, so Python drives the parameter through Params.

**Next.** Settle one `ArrayGeometry` transport so a descriptor output can
feed a descriptor override port. Document any OPERATOR_API physical-type
divergence.

**Status:** open — `ArrayGeometry` transport decision.

---

## 2. Dense RDA-cube memory footprint

**Context.** The output cube is `(..., num_az, num_el, num_doppler,
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

## 3. Beam-axis enumeration (implementation note, not a team question)

The GEMM emits beams az-major (`beam = ia * num_el + ie`) so the output buffer
aliases the `(..., num_az, num_el, D, R)` cube with no post-permute. The device
steering builder (`radar_common/dsp/steering_weights`) is asked for
`BeamOrder::kAzMajor` +
`SteeringLayout::kBeamMajor` + `conjugate`, so the weights land directly in
that order/layout — no host reorder. Recorded here only to explain the beam
order; not an open decision.

---

## 4. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_phase_shift_beamformer_op_bindings.cpp` is still the
HSDK 4.x `Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.
