# TDMSparseCompensationOp

Detection-indexed TDM compensation for sparse angle pipelines.

The operator consumes uncompensated `(M_tx, N_rx, doppler, range)` maps and
exact-sized `(doppler_bin, range_bin)` detections. For each detection and
configured RD-neighborhood cell it applies the same firing-slot-aware phase
correction as `TDMDenseCompensationOp`, assembles the same virtual-channel
ordering as `VirtualArrayAssemblyOp`, and emits estimator-ready snapshots.

Outputs are:

- `snapshot`: `(num_detections, num_virtual, num_snapshots)`
- `is_valid`: uint8 edge-validity mask
- `array_geometry`: tick-coherent `WireArrayGeometry`

The geometry wavelength comes from the effective build-time or `[live]`
waveform, so it remains coherent with the correction table. Geometry and mask
are published before `snapshot` for downstream Exact joins.

This is a TDM-only bridge. Sparse DDM processing changes the detector because
TX separation occurs in Doppler bands; it remains a separate future operator.
See `docs/planning/OPERATOR_API.md` for the locked contract and
`docs/planning/WORKFLOWS.md` for dense and sparse graph placement.
