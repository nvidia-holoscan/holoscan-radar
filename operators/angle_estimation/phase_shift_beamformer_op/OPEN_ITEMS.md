# PhaseShiftBeamformerOp open items

Issues for `PhaseShiftBeamformerOp` that block full
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
completion on the HSDK 5 EA branch, or that need follow-up after EA. Cross-cut
SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

When an item closes, summarize in [`STATUS.md`](../../../docs/planning/STATUS.md)
if it affects the module, then remove or archive the section here.

---

## 1. Live `[live]` Input ports for geometry / scan grid / taper

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”;
Performance → inline live-override measurement (when Graph-ported).

**Class:** Durable HSDK 5 `PublishedPayload` admission (§9). The temporal
side is settled (L16 / L17): enabled `[live]` ports join the data trigger
with `Exact{}` + `on_missing SafeDefault{}`, gated by a per-port `expect_*`
(default false) with `sticky_*` (default true) latching the last match —
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md). The
only blocker left here is that none of these parameters has an admissible
payload type, so there is no port to declare.

**Contract.** `OPERATOR_API.md` marks `array_geometry`, `az_angles`,
`el_angles`, and `array_taper` as sticky `[live]`; `angle_convention`
remains STATIC.

**Why Params-only on EA.**

- `Input<std::vector<double>>` is rejected (`az_angles` / `el_angles`).
- `ArrayGeometry` and `WindowSpec` own `std::vector` (and
  `std::optional`) members — not ChannelPod / schema payloads — so
  `Input<ArrayGeometry>` / `Input<WindowSpec>` fail compile.
- Unbounded scan grids cannot use a fixed-size POD like
  `SnapshotWindowOverride`.

**Interim.**

- Operator uses `Operator<PhaseShiftBeamformerParams>` + `set_params`;
  no live Input ports are declared, so `contract()` carries only
  `OnEach{rd_maps}` and there are no `expect_*` / `sticky_*` fields.
- `ConfigCache` steering rebuild still keys on the Params-derived
  geometry / grid / taper.
- Graph sticky override tests
  (`LiveAzGridStickyOverride`, `LiveElGridStickyOverride`,
  `LiveArrayGeometryStickyOverride`, `LiveArrayTaperStickyOverride`) are
  `GTEST_SKIP`’d.
- Params authoring covers selection
  (`ScanGridConfigurableFromParams`, `ArrayTaperConfigurableFromParams`).

**Next.** Choose a durable live transport (Tensor for scan grids; nested
ChannelPod / schema for `ArrayGeometry` / `WindowSpec`, or flatten), then
declare each port unconditionally in `setup()` and add its `Exact{}` +
`SafeDefault{}` contract entry behind `expect_*`, and unskip the Graph
tests with the port wired. Document any OPERATOR_API physical-type
divergence.

**Status:** open — do not treat as checklist-complete until live Inputs
land.

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
