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

**[EA HSDK5 ONLY]** On the `hsdk-5.0` branch Holoviz is absent from the EA
package, so the window is drawn by the Xlib stand-in `X11RdViewerOp`
(`radar_app/viz/`): RD map with cluster-colored detections and representative
crosshairs, legend, rolling range/velocity plots, and a boresight az/el scatter
of the virtual-array beamscan angles. X11 only — an `Xvfb` `DISPLAY` suffices.
See [`HSDK5_PORT.md`](../../docs/planning/HSDK5_PORT.md) §6.

## Configuration notes

| Topic | TDM (`fmcw_mimo.yaml`) | DDM (`fmcw_mimo_ddm.yaml`) | DDM-NL (`fmcw_mimo_ddm_phase_codes.yaml`) |
| --- | --- | --- | --- |
| Slow-time | `cube.chirps_per_tx` per TX; wire length `M·L` | `cube.num_chirps`; all TX every chirp | same as DDM |
| PRI | `sim.physical_*` for simulator; `waveform.*` = `M × physical` for operators | `waveform.chirp_repetition_interval` = on-wire PRI | same |
| Marshaling | `mimo_deinterleave.tx_schedule` | `ddm_code.doppler_offsets` (gather) | `ddm_code` / `generate_phase_codes: zadoff_chu` (correlate) |
| Clutter | before Doppler | after compensation (demuxed block) | same as DDM |
| Doppler out | per-TX chirps | `N/M − guard` bins | full `N` bins |

Shared: `tx_array`, `rx_array`, targets, beamscan grid, `virtual_array_assembly.overlap_policy: keep_all`.

### Scene

All three configs use the same five targets, mirroring the
[`fmcw_aoa_estimators`](../fmcw_aoa_estimators/README.md) scene. Four of the
five move **non-radially** — each carries a transverse velocity of
0.32–0.46 m/s on top of its radial one, so its az/el sweep over the 25 s run;
target 3 is the single purely radial reference. Both inbound targets sweep
toward boresight, since an inbound target's angular rate goes as `v_t / r` and
one sweeping outward would leave the ±0.3 rad scan grid.

Three are well separated in speed (−6.7 / +1.2 / +8.5 m/s); the other two each
shadow one of them through a close pass, 1.83 m/s faster than the target they
shadow — resolvable in Doppler, but inside the 2.0 m/s `cluster_dbscan.epsilon`,
so the pair can merge. One closes a 15 m gap and coincides near t ≈ 8 s, the
other a 27 m gap near t ≈ 15 s. The first pair is split by 0.316 rad in azimuth,
the second by only 0.023 rad — so one pass is separable by angle and the other
is not.

Two constraints differ from the SIMO apps and set the speeds:

- Every radial *starts* on a **Doppler bin center** (a multiple of the
  ~0.6083 m/s bin). Non-radial motion means the radial rate no longer holds
  still, but the drift is under 0.2 bins across the whole run and monotone, so a
  representative reads as a gently sloping line rather than the frame-to-frame
  flicker an off-center radial parked on a bin edge would produce.
- The **unambiguous velocity span is halved**. TDM stretches the effective PRI
  by `num_tx`; DDM splits the 128-bin Doppler axis into one 64-bin band per TX
  (less `ddm_code.guard_bands` at each edge). Both land at ±19.5 m/s, so the
  ±14-bin (±8.5 m/s) ladder stays well clear of the fold.

## Python

A Python variant with the same graph lives at
[`python/fmcw_mimo.py`](python/fmcw_mimo.py) (`config.py` + `custom_ops.py`
colocated, same layering as the C++ side — see
[`python/README.md`](python/README.md)).
