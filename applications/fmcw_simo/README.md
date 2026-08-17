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

**[EA HSDK5 ONLY]** On the `hsdk-5.0` branch Holoviz is absent from the EA
package, so the same window — RD map, axes, cluster-colored detections and
representative crosshairs, legend, the two rolling plots, and the boresight
scatter — is drawn by the Xlib stand-in `X11RdViewerOp` (`radar_app/viz/`):
X11 only, no Vulkan, and an `Xvfb` `DISPLAY` is enough. The viewer presents on
the Holoscan compute thread, so it is a live-look tool rather than a
performance reference. See
[`HSDK5_PORT.md`](../../docs/planning/HSDK5_PORT.md) §6.

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

The default scenario is an 8x8 URA with five targets, matching the
[`fmcw_aoa_estimators`](../fmcw_aoa_estimators/README.md) scene so the two apps
are directly comparable. Four of the five move **non-radially** — each carries a
transverse velocity of 0.32–0.46 m/s on top of its radial one, so its az/el
sweep over the 25 s run; target 3 is the single purely radial reference. Both
inbound targets sweep toward boresight, since an inbound target's angular rate
goes as `v_t / r` and one sweeping outward would leave the ±0.3 rad scan grid.

Three are well separated in speed (−6.5 / +1.5 / +9 m/s); the other two each
shadow one of them through a close pass, 1.5 m/s faster than the target they
shadow — 2.5 velocity bins, still inside the 2.0 m/s `cluster_dbscan.epsilon`,
so the pair can merge. Transverse motion is small next to these ranges, so the
radial rates drift under 0.2 Doppler bins and both passes still coincide in
range and Doppler:

- **Pair A** (target 4 shadows target 2, coinciding at t ≈ 10 s) is separated by
  0.318 rad in azimuth, past the array's 0.253 rad first null — angle alone can
  carry it through the crossing.
- **Pair B** (target 5 shadows target 1, coinciding at t ≈ 18 s) differs by only
  0.025 rad in azimuth, far inside the first null, so this array cannot split it
  and every cell reports a blended angle. Elevation differs by 0.057 rad, just
  past the clustering epsilon.

## Python

A Python variant with the same graph lives at
[`python/fmcw_simo.py`](python/fmcw_simo.py) (`config.py` colocated, same
layering as the C++ side — see [`python/README.md`](python/README.md)).
