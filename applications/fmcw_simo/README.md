# FMCW SIMO example application

Sparse **angle-then-cluster** SIMO pipeline demonstrating the contract path in
[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md) § SIMO:

```text
SimoAdcSourceOp (N_rx, chirps, samples)
  → Range → Clutter → Doppler → rd_maps (N_rx, D, R)
  → CFAR (nci_axes: [0]) → SnapshotGather → BeamscanEstimator
  → FeatureAssembly [range, velocity, az, el]
  → ClusterDBSCAN → DetectionReduction → ClusterReportOp
```

Detection runs on a non-coherent sum over RX (`nci_axes: [0]`). Angle estimation
uses per-channel snapshots gathered at each CFAR cell. Clustering weights az/el
heavily via `reduction.feature_weights`.

## Build

```bash
cmake -S . -B build -DBUILD_ALL=ON
cmake --build build --target fmcw_simo
```

## Run

```bash
./build/applications/fmcw_simo/cpp/fmcw_simo
# or with a custom config:
./build/applications/fmcw_simo/cpp/fmcw_simo /path/to/fmcw_simo.yaml
```

Set `headless: true` or `visualization.enabled: false` in the YAML for log-only
runs on hosts without a display.

## Display

With `visualization.enabled: true` and `plots.enable: true` the Holoviz window
shows the non-coherently integrated range-Doppler map on the left — the same
surface CFAR thresholds, since it runs `nci_axes: [0]` over the RX axis, so the
picture and the overlaid detections come from the same data — with CFAR
detections drawn as cluster-colored dots, and three stacked panels on the right:

- **Range (m) vs time** and **Velocity (m/s) vs time** — rolling per-cluster
  representative tracks.
- **Boresight az/el (deg)** — the view looking out along the array boresight:
  each detection is a cluster-colored `+` at its beamscan `(azimuth, elevation)`
  estimate (azimuth left-right, elevation bottom-up). Axis extents come from the
  `beamscan_estimator` grid; a single-value elevation grid is padded so the axis
  is visible.

## Configuration

[`cpp/fmcw_simo.yaml`](cpp/fmcw_simo.yaml) defines:

- **`rx_array`** — a URA (uniform rectangular array) in the y-z plane,
  `az_elements` (along y, azimuth aperture) x `el_elements` (along z, elevation
  aperture), `spacing` as a fraction of the carrier wavelength. A 2D aperture is
  what makes both azimuth and elevation observable — a single line array (any
  axis or diagonal) has only 1D aperture and cannot separate the two. The
  channel count (`az_elements * el_elements`) must match the beamscan.
- **`targets`** — point targets by initial `(range, azimuth, elevation)` plus a
  `velocity: [vx, vy, vz]` vector (m/s) in the array frame (x = boresight,
  y = azimuth aperture, z = elevation aperture). The y/z components change the
  target's angles over time, so targets move on all three axes; both angles are
  resolved by the URA.
- **`cfar_detector.nci_axes: [0]`** — integrate RX before thresholding.
- **`cluster_dbscan.epsilon`** — four values for `[range, velocity, az, el]`.

The default scenario is an 8x8 URA with three targets moving on all three axes;
their range, azimuth, and elevation all sweep over the run while staying inside
the beamscan grid.

## Python

A Python variant with the same graph lives at
[`python/fmcw_simo.py`](python/fmcw_simo.py) (`config.py` colocated, same
layering as the C++ side — see [`python/README.md`](python/README.md)).
