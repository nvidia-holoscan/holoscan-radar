# FMCW MIMO TDM example application

Sparse **angle-then-cluster** MIMO-TDM pipeline demonstrating the contract path in
[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md) § MIMO — TDM:

```text
MimoTdmAdcSourceOp (N_rx, M·L, samples)
  → MIMODeinterleaveOp → Range → Clutter → Doppler
  → MIMOCoherentCompensationOp → VirtualArrayAssemblyOp
  → rd_maps (num_virtual, D, R)
  → CFAR (nci_axes: [0]) → SnapshotGather → BeamscanEstimator
  → FeatureAssembly [range, velocity, az, el]
  → ClusterDBSCAN → DetectionReduction → ClusterReportOp
```

`VirtualArrayAssemblyOp.array_geometry` is wired into `BeamscanEstimatorOp` as a
`[live]` override. Detection runs on a non-coherent sum over virtual channels.

## Build

```bash
cmake -S . -B build -DBUILD_ALL=ON
cmake --build build --target fmcw_mimo
```

## Run

```bash
./build/applications/fmcw_mimo/cpp/fmcw_mimo
# or with a custom config:
./build/applications/fmcw_mimo/cpp/fmcw_mimo /path/to/fmcw_mimo.yaml
```

Set `headless: true` or `visualize: false` in the YAML for log-only runs on
hosts without a display.

## Configuration

[`cpp/fmcw_mimo.yaml`](cpp/fmcw_mimo.yaml) highlights:

- **`tx_array`** — TX count and element spacing (wavelengths) along the azimuth
  aperture; default is 2 TX separated by half a wavelength.
- **`rx_array`** — 4×4 RX URA (same convention as `fmcw_simo`).
- **`sim.physical_chirp_repetition_interval`** — on-wire inter-chirp spacing for
  the interleaved ADC simulator.
- **`waveform.chirp_repetition_interval`** — per-TX effective PRI (`M × physical`)
  passed to range/Doppler/compensation operators.
- **`mimo_deinterleave.tx_schedule`** — TDM firing pattern (`num_tx` required).
- **`virtual_array_assembly.overlap_policy: keep_all`** — retain all `M·N` virtual
  channels for the demo (dedup collapses coincident positions when desired).

The default scenario uses 2 TX × 16 RX → 32 virtual channels and three moving
point targets.
