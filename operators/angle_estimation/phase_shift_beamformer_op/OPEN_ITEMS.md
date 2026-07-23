# PhaseShiftBeamformerOp open items

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
steering builder (`radar_common/dsp/steering_weights`) is asked for
`BeamOrder::kAzMajor` +
`SteeringLayout::kBeamMajor` + `conjugate`, so the weights land directly in
that order/layout — no host reorder. Recorded here only to explain the beam
order; not an open decision.
