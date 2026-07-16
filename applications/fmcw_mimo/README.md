# FMCW MIMO example application

Sparse **angle-then-cluster** MIMO pipeline (TDM or DDM) selected by
`mimo_coherent_compensation.multiplexing_scheme` in the YAML config.

## TDM (default)

[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md) § MIMO — TDM:

```text
MimoTdmAdcSourceOp (N_rx, M·L, samples)
  → MIMODeinterleaveOp → Range → Clutter → Doppler
  → MIMOCoherentCompensationOp → VirtualArrayAssemblyOp → ...
```

Config: [`cpp/fmcw_mimo.yaml`](cpp/fmcw_mimo.yaml)

## DDM

[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md) § MIMO — DDM:

```text
MimoDdmAdcSourceOp (N_rx, num_chirps, samples)
  → Range → Doppler → MIMOCoherentCompensationOp
  → ClutterFilterOp (post-demux) → VirtualArrayAssemblyOp → ...
```

Config: [`cpp/fmcw_mimo_ddm.yaml`](cpp/fmcw_mimo_ddm.yaml)

Downstream from virtual-array assembly is identical (CFAR → gather → beamscan
→ cluster).

## Build

```bash
cmake -S . -B build -DBUILD_ALL=ON
cmake --build build --target fmcw_mimo
```

## Run

```bash
# TDM (default config staged next to the binary)
./build/applications/fmcw_mimo/cpp/fmcw_mimo

# DDM
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm.yaml

# or an absolute path to any config
./build/applications/fmcw_mimo/cpp/fmcw_mimo /path/to/config.yaml
```

Set `visualize: false` or `headless: true` for log-only runs without a display.

## Configuration notes

| Topic | TDM (`fmcw_mimo.yaml`) | DDM (`fmcw_mimo_ddm.yaml`) |
| --- | --- | --- |
| Slow-time | `cube.chirps_per_tx` per TX; wire length `M·L` | `cube.num_chirps`; all TX every chirp |
| PRI | `sim.physical_*` for simulator; `waveform.*` = `M × physical` for operators | `waveform.chirp_repetition_interval` = on-wire PRI |
| Marshaling | `mimo_deinterleave.tx_schedule` | `mimo_coherent_compensation.ddm_code` |
| Clutter | before Doppler | after compensation (demuxed block) |

Shared: `tx_array`, `rx_array`, targets, beamscan grid, `virtual_array_assembly.overlap_policy: keep_all`.
