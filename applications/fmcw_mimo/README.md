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

### DDM — non-linear `phase_codes`

Same DDM chain, but `ddm_code` uses non-linear codes
(`generate_phase_codes: zadoff_chu` expands to distinct Zadoff-Chu roots —
low periodic crosstalk, still non-linear so the correlate path runs).
`MIMOCoherentCompensationOp` takes the conjugate-correlation path (full Doppler
axis out; `guard_bands` must be 0). Use `generate_phase_codes: sin` only to
intentionally stress high-crosstalk leakage.

**Doppler window:** the phase-codes demo keeps `doppler_window: none`. A Hann
(or other taper) before correlate breaks Zadoff-Chu periodic correlation and
tends to fill the RD map with a pedestal; linear DDM gather can use Hann.

Config: [`cpp/fmcw_mimo_ddm_phase_codes.yaml`](cpp/fmcw_mimo_ddm_phase_codes.yaml)

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

# DDM (linear-phase / doppler_offsets)
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm.yaml

# DDM (non-linear phase_codes / correlate demux)
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm_phase_codes.yaml

# or an absolute path to any config
./build/applications/fmcw_mimo/cpp/fmcw_mimo /path/to/config.yaml
```

Set `visualization.enabled: false` or `headless: true` for log-only runs without
a display. When visualization is on, the RD map shown is the non-coherent
integration over the virtual-channel axis — the surface CFAR thresholds
(`nci_axes: [0]`) — not a single virtual channel.

## Configuration notes

| Topic | TDM (`fmcw_mimo.yaml`) | DDM (`fmcw_mimo_ddm.yaml`) | DDM-NL (`fmcw_mimo_ddm_phase_codes.yaml`) |
| --- | --- | --- | --- |
| Slow-time | `cube.chirps_per_tx` per TX; wire length `M·L` | `cube.num_chirps`; all TX every chirp | same as DDM |
| PRI | `sim.physical_*` for simulator; `waveform.*` = `M × physical` for operators | `waveform.chirp_repetition_interval` = on-wire PRI | same |
| Marshaling | `mimo_deinterleave.tx_schedule` | `ddm_code.doppler_offsets` (gather) | `ddm_code` / `generate_phase_codes: zadoff_chu` (correlate) |
| Clutter | before Doppler | after compensation (demuxed block) | same as DDM |
| Doppler out | per-TX chirps | `N/M − guard` bins | full `N` bins |

Shared: `tx_array`, `rx_array`, targets, beamscan grid, `virtual_array_assembly.overlap_policy: keep_all`.

## Python

A Python variant with the same graph lives at
[`python/fmcw_mimo.py`](python/fmcw_mimo.py) (`config.py` + `custom_ops.py`
colocated, same layering as the C++ side — see
[`python/README.md`](python/README.md)).
