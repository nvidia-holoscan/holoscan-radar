# DopplerNotchOp Open Items

## 1. Determine whether the operator should emit a notch mask

**Status:** deferred from the initial operator contract on 2026-09-16. The
first pass emits only the filtered RD tensor.

A hard or tapered post-Doppler notch creates a deterministic low-power region.
CFAR training windows that cross its boundary may underestimate interference
and raise realized Pfa. Before adding a public `notch_mask` output, measure
hard-null and cosine-transition behavior at the notch interior and edges for
CA, GO, SO, and OS CFAR. Compare against the existing pre-Doppler
`ClutterFilterOp` methods so the decision is based on observed statistical
impact rather than assuming post-Doppler filtering is unique.

If a mask is justified, decide whether it marks only the full-rejection
stopband or also some/all of the transition, whether it is a rank-1 Doppler
mask or a broadcastable RD mask, and whether it is a causal sideband emitted
before `rd_maps_out`. Coordinate the contract with
`CFARDetectorOp`'s generic exclusion-mask investigation; do not add a
notch-specific CFAR port.
