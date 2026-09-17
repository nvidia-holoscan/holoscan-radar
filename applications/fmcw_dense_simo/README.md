# FMCW SIMO — dense angle path (`fmcw_dense_simo`)

A lean C++ demo that exercises `PhaseShiftBeamformerOp` end-to-end: the **dense**
angle path, where a beam is formed for *every* range–Doppler cell to build a
Range–Doppler–Angle (RDA) cube, and detection runs on that cube. It runs with an
optional Holoviz display or fully headless (log-only).

This is a sibling to [`simo_detect_then_beamform`](../fmcw/simo_detect_then_beamform), which demonstrates the
**sparse** path (detect first on the RD map, estimate angle only at detections
via `BeamscanEstimatorOp` + `SnapshotGatherOp`). The two are kept separate on
purpose: the dense path diverges from the sparse path in both the detect+angle
middle *and* the reporting layer — dense CFAR emits a 4-column `detection_idx`
`(az_bin, el_bin, doppler_bin, range_bin)` and a rank-4 `detection_score`,
whereas the shared `radar_app` report/visualization ops assume the sparse
2-column layout. Rather than branch those stable ops, this demo carries its own
small, dense-aware bridge and log ops.

## Pipeline

```text
SimoAdcSourceOp                 (simulated SIMO ADC cube: N_rx, chirps, samples)
  -> FMCWRangeProcessorOp
  -> ClutterFilterOp
  -> FMCWDopplerProcessorOp      (rd_maps: ..., doppler, range)
  -> PhaseShiftBeamformerOp      (rda_cube: az, el, doppler, range)
  -> CFARDetectorOp              (nci_axes: [] -> az/el ride as batch;
                                  detection_idx = az_bin, el_bin, dopp, range)
  -> AnglePeakPickOp             (one detection per RD cell: the peak (az,el))
  -> CubeAoaLookupOp  (bridge)   (az_bin/el_bin -> aoa_estimates angle values)
  -> FeatureAssemblyOp           (points: [range, velocity, az, el])
  -> ClusterDBSCANOp
  -> DetectionReductionOp        (medoid representative per cluster)
  -> DenseClusterReportOp        (logs cluster centroids + representatives)
```

CFAR's `detection_mask` / `detection_score` / `noise_estimate` outputs are
unused here; each is attached to a small `EntityDrainOp` so CFAR has a
downstream on every port and is scheduled.

## Display

With `visualization.enabled: true` (and a reachable display, or `headless: true` for
offscreen rendering) the app opens a Holoviz window reusing the app-agnostic
`radar_app/viz` operators unchanged:

- **Left:** the range–Doppler map (turbo colormap) with CFAR detections drawn
  as cluster-colored dots. This app detects on the *beamformed* RDA cube
  (`nci_axes: []`), so `CFARDetectorOp.cfar_power` is a per-beam power cube
  `(num_az, num_el, num_doppler, num_range)` rather than a single map.
  `RdAngleMaxOp` collapses the beam grid by taking each RD cell's **max over
  all beams**, so the image is the detector's own power domain projected the
  same way the detections are: `AnglePeakPickOp` collapses each detected RD
  cell to its *peak* beam, so the overlay is already a max-over-beams
  projection. Both therefore live in the same domain, and a target anywhere on
  the grid is drawn over the cell that actually detected it.

  Slicing one fixed beam (`RdChannelSliceOp` with `leading_index` set to the
  grid cell nearest boresight) was the previous behavior. It kept the image a
  physical beam, but put it in a different projection from the overlay, so an
  off-boresight target was drawn over a cell the beam pattern had attenuated —
  and a target out near the ±0.3 rad grid edge sat in the null and vanished
  from the image while still being detected. `RdChannelSliceOp` is still the
  right choice when you specifically want one beam.

  The cost of the max is a higher noise floor: the max of N beams of noise sits
  above one beam's mean. For an n-beam grid of exponential-distributed power
  the rise is about `10·log10(H_n)` dB — ~7.6 dB at 13×13 — though less in
  practice, since adjacent beams are correlated across the mainlobe.
  `RdMapToImageOp`'s per-column median normalization absorbs part of it,
  because the median rises too.
- **Right (stacked):** range-vs-time and velocity-vs-time rolling plots of the
  per-cluster representative, and a boresight azimuth/elevation scatter of those
  same per-cluster representatives (the angle information the dense path adds) —
  one colored marker per cluster centroid rather than the raw detections.

The Doppler stage still emits per-channel `rd_maps (N_rx, D, R)`, so
`RdChannelSliceOp` / `RdMapToImageOp` / `RollingPlotOp` / `BoresightViewOp` apply
directly. The one adaptation is `TrailingIdxColumnsOp`, which slices the dense
4-column `detection_idx` / `rep_idx` down to their trailing
`(doppler_bin, range_bin)` columns so the RD-plane overlay/plot ops see the
sparse 2-column form they expect — the shared viz ops are untouched. The
boresight scatter reuses `CubeAoaLookupOp` on the reduction's `rep_idx` (one row
per cluster): its angles feed the scatter and its `row_index` output doubles as
the per-marker cluster label, so the boresight shows one centroid per cluster.

**[EA HSDK5 ONLY]** On the `hsdk-5.0` branch Holoviz is absent from the EA
package, so the window is drawn by the Xlib stand-in `X11RdViewerOp`
(`radar_app/viz/`) — same RD map, overlays, legend, rolling plots, and
boresight panel, X11 only and satisfied by an `Xvfb` `DISPLAY`. One deliberate
difference: the X11 boresight plots *every* detection (cluster-colored, like
the RD overlay) from `CubeAoaLookupOp`'s per-detection angles, rather than one
marker per cluster representative. See
[`HSDK5_PORT.md`](../../docs/planning/HSDK5_PORT.md) §6.

If visualization is enabled but no display is reachable, the app logs a warning
and
runs headless automatically (disable that probe with `skip_display_check:
true`). The `DenseClusterReportOp` cluster log runs in both modes.

### App-local operators

- **`AnglePeakPickOp`** — collapses the cube's many per-beam detections to one
  per range–Doppler cell: the `(az, el)` beam with the largest CFAR score (the
  beamformed-power peak, i.e. the target's true DOA). CFAR runs independently on
  each beam's RD plane (`nci_axes: []`), so it has no cross-beam view — a strong
  target trips the threshold in *many* beams (its broad mainlobe plus
  sidelobes), scattering raw detections across the whole angular field of view.
  Peak-picking is the dense analog of the sparse path's "one angle per
  detection" and is what makes the boresight scatter and RD overlay tight and
  clustering unambiguous. Its peaked `detection_idx` drives the entire
  downstream (angle lookup, features, clustering, reduction, display).
- **`CubeAoaLookupOp`** — the dense-path bridge. `FeatureAssemblyOp` reads
  azimuth/elevation from an `aoa_estimates` port, not from the RDA cube's
  preserved `(az_bin, el_bin)` batch columns. This op maps those detection bins
  through the scan grids to physical angles and emits `aoa_estimates`
  `(num_points, 2)`. It is a plain-tensor add-on (per the data-plane invariant),
  not a shared type or an operator contract change.
- **`DenseClusterReportOp`** — a dense-aware clone of `radar_app`'s
  `ClusterReportOp`, handling the 4-column `points` / `rep_idx` layout.
- **`EntityDrainOp`** — a local copy of `radar_app`'s `NullSinkOp` (raw Entity
  receive; keeps the app self-contained without linking the sparse report ops).

## Configuration notes

See [`cpp/fmcw_dense_simo.yaml`](cpp/fmcw_dense_simo.yaml).

- **Beam grid is deliberately modest** (13 × 13 over ±0.3 rad). The dense path
  materializes the full `num_az × num_el × num_doppler × num_range` cube, so a
  fine grid over a full RD map is memory-heavy — see the `PhaseShiftBeamformerOp`
  `OPEN_ITEMS.md`.
- **`cfar_detector.nci_axes: []`** leaves the `(az, el)` axes as batch slices, so
  CFAR runs a 2-D detector on each beam's RD plane and widens `detection_idx` to
  four columns.
- **`phase_shift_beamformer.array_taper`** (Taylor, 35 dB) suppresses spatial
  sidelobes. Without it, a strong target leaks above the CFAR threshold in nearly
  every beam (untapered ULA sidelobes are only ~-13 dB).
- **`noise_snr_db: -28`** (per-sample SNR for a unit-amplitude target). The
  coherent processing gain of the range + Doppler FFTs and the beamformer is
  ~63 dB, so a modest per-sample SNR still yields a strong integrated peak; this
  value keeps the integrated dynamic range in a regime where CFAR fires on the
  mainlobe region but not on deep sidelobes.
- **`cluster_dbscan.epsilon: [3, 2, 0.3, 0.3]`** uses a wide azimuth/elevation
  radius so the near-mainlobe detection cloud of a single target (all at the same
  range/velocity, spread across neighboring beams) collapses into one cluster.
  The medoid representative then lands on the true direction of arrival
  (grid-snapped).
- **`targets`** — the same five-target scene as
  [`fmcw_aoa_estimators`](../fmcw_aoa_estimators/README.md), so the dense and
  sparse paths can be compared on identical input. Four of the five move
  non-radially (transverse velocity 0.32–0.46 m/s), so their az/el sweep over
  the 25 s run; target 3 is the single purely radial reference. Three are well
  separated in speed (−6.5 / +1.5 / +9 m/s); the other two each shadow one of
  them through a close pass, 1.5 m/s faster than the target they shadow,
  coinciding at t ≈ 10 s and t ≈ 18 s. Note that the wide 0.3 rad angular
  epsilon above is the same order as the 0.318 rad azimuth split of the first
  pair, so this app is expected to merge close passes that the sparse apps keep
  apart — the clustering radius, not the array, is the limit here. The sweeping
  angles also mean a target crosses beam-grid cells during the run, so its
  grid-snapped representative steps rather than holding one cell.

## Build & run

```bash
cmake -S . -B build -DAPP_fmcw_dense_simo=ON
cmake --build build --target fmcw_dense_simo -j"$(nproc)"
./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo \
    [path/to/config.yaml]   # defaults to the installed fmcw_dense_simo.yaml
```

Expected: each frame logs `dense: <N> detections -> 3 clusters`, one per target,
with cluster centroids/representatives near the configured target range,
velocity, azimuth, and elevation.

## Python

A Python variant with the same graph lives at
[`python/fmcw_dense_simo.py`](python/fmcw_dense_simo.py) (`config.py` +
`custom_ops.py` colocated, same layering as the C++ side — see
[`python/README.md`](python/README.md)).
