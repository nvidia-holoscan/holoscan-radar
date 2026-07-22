# PhaseShiftBeamformerOp open items

## 1. `AngleConvention::kCustom` boundary mapping

**Context.** `OPERATOR_API.md` documents a `custom` preset with explicit axis
fields. `boundary_to_canonical` / `canonical_to_boundary` (used by
`build_steering_table`) threw `invalid_argument` for `preset=custom` until the
team defined the mapping rules. Shared with `BeamscanEstimatorOp` OPEN_ITEMS §1.

**Resolved (L11, 2026-07-22).** The generic `custom` mapping is implemented in
`angle_convention.cpp` (see `BeamscanEstimatorOp` OPEN_ITEMS §1 and STATUS.md
L11); `build_steering_table` now works for `preset=custom`. A degenerate axis
config throws `invalid_argument` at table-build time. Pinned by
`test_angle_convention.cpp` (preset-equivalence + round-trip + rejection).

**Status:** resolved — `custom` implemented + tested (shared mapping).

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

## 3. Beam-axis enumeration (implementation note, not a team question)

The GEMM emits beams az-major (`beam = ia * num_el + ie`) so the output buffer
aliases the `(..., num_az, num_el, D, R)` cube with no post-permute. The device
steering builder (see §5) is asked for `BeamOrder::kAzMajor` +
`SteeringLayout::kBeamMajor` + `conjugate`, so the weights land directly in
that order/layout — no host reorder. Recorded here only to explain the beam
order; not an open decision.

## 4. Performance harness (landed)

`performance/cpp/perf_phase_shift_beamformer.cpp` characterizes steady-state
dense beamforming cost (channels × beam grid × RD cells) and inline
`[live] az_angles` override latency (finer grid → steering rebuild). The
`perf_phase_shift_beamformer_smoke` ctest (label `performance`) runs a tiny
subset and verifies a report is emitted. Baselines saved under
`perf_results/perf_phase_shift_beamformer{,.full}.{txt,json}`.

## 5. Steering-weight build moved to GPU (landed)

**Context.** Profiling the `[live]` override showed the serial host trig loop in
`build_steering_table` dominated the override spike at scale (host build alone
≈0.97 ms at the `lg` override grid — larger than the *entire* post-move override
delta of ≈0.70 ms; the gap widens at `xl`). The
`O(num_beams·num_channels)` phase evaluation now runs in a kernel via the shared
`build_steering_weights_device` (`radar_common/dsp/steering_weights_device.hpp`,
compiled into `holoscan::radar_common_steering`). Only the per-beam direction
vectors + validity (`O(num_beams)`, the convention-dependent part) and the taper
are computed host-side and uploaded — a *smaller* H2D than the full complex
table it replaced. The builder emits weights directly in the requested beam
order / layout / conjugation, so both angle operators consume it with no host
reorder. The host `build_steering_table` is retained as the unit-test reference
(`tests/cpp/common/test_steering_weights_device.cpp` asserts device == host).

**Status:** landed — shared by `PhaseShiftBeamformerOp` and
`BeamscanEstimatorOp`.

## 6. Dense-path az/el assembly gap (FeatureAssemblyOp) — team question

**Context.** On the dense path the RDA-cube CFAR emits `detection_idx` with the
scan-grid `(az_bin, el_bin)` as leading *preserved-batch* columns
(`(az_bin, el_bin, doppler_bin, range_bin)`), and a rank-4 `detection_score`.
But `FeatureAssemblyOp` sources the `kAz` / `kEl` features **only** from its
`aoa_estimates` input port (angle *values*, the sparse-path form produced by
`BeamscanEstimatorOp`); it treats the leading `detection_idx` columns purely as
opaque batch coordinates for gathering `detection_score` and never maps them
through the beam grid to angles. So the dense path cannot feed angle features to
clustering without an adapter.

`applications/fmcw_dense_simo` works around this with an app-local
`CubeAoaLookupOp` (reads the `(az_bin, el_bin)` columns, looks them up in the
az/el scan grids, emits `aoa_estimates`). That keeps the data plane as plain
tensors and requires no contract change, but it is per-app glue.

**Question for the team.** Should the dense az/el-bin → angle mapping be a
sanctioned shared operator (e.g. promote `CubeAoaLookupOp` into
`operators/detection/` or `common/`), or should `FeatureAssemblyOp` optionally
accept az/el grids and resolve angles from the preserved batch columns itself?
Either is an `OPERATOR_API.md` contract change — do not resolve silently.

**Resolved (S4, 2026-07-21).** `FeatureAssemblyOp` was extended (chosen over a
standalone shared op): new `az_bin` / `el_bin` feature tokens + optional
`az_angles` / `el_angles` grid inputs resolve az/el from the preserved
`(az_bin, el_bin)` `detection_idx` columns — symmetric with its existing
range/velocity bin lookups. The dense path feeds angle features to clustering
with no per-app adapter; `fmcw_dense_simo` / `fmcw_mimo` can drop their local
`CubeAoaLookupOp` at leisure (the workaround still works). See STATUS.md S4 and
`OPERATOR_API.md` → FeatureAssemblyOp.

**Status:** resolved — extended `FeatureAssemblyOp`, implemented + tested.
