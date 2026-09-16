# CFARDetectorOp Open Items

## 1. Add a generic detection and training exclusion mask

**Status:** deferred 2026-09-16; not required by the initial
`DopplerNotchOp` contract.

Investigate a generic validity/exclusion input for cells that CFAR must not use
as CUTs or training samples. Potential producers and uses include a future
`DopplerNotchOp.notch_mask`, known interference or saturation excision,
calibration-invalid bins, sensor blind zones, and algorithm-owned guard bands.
The surface must remain mechanism-neutral rather than naming Doppler notches.

Before locking a contract, settle:

- accepted shapes and named-axis broadcasting, including rank-1 Doppler masks
  versus masks matching the post-NCI CFAR plane;
- same-tick temporal pairing and missing-mask behavior;
- whether masked cells are excluded as CUTs, training cells, or both;
- dynamic training-cell counts and Pfa-derived alpha selection for CA;
- independent side counts for GO/SO and rank semantics when OS training cells
  are removed;
- edge behavior, minimum surviving training count, NMS interaction, and output
  values for excluded CUTs;
- mask dtype and whether multiple exclusion producers require an explicit
  composition operator.

Characterize realized Pfa at mask boundaries and ordinary map edges before
implementation. Any new input is an additive contract change to
`OPERATOR_API.md`.
