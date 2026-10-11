# SIMO

Detect-then-beamform FMCW SIMO example. It performs CFAR after receiver-power integration, gathers coherent snapshots only at detected range-Doppler cells, estimates angle, clusters neighboring detections, and reports one representative per cluster.

```mermaid
flowchart LR
    Source[Sim ADC source] --> Range[Range processor] --> Clutter[Clutter filter] --> Doppler[Doppler processor]
    Doppler --> CFAR[CFAR detector]
    Doppler --> Gather[Snapshot gather]
    CFAR --> Gather --> Beamscan[Beamscan estimator] --> Features[Feature assembly]
    CFAR --> Features --> Cluster[DBSCAN] --> Reduce[Detection reduction] --> Report[Cluster report]
    Features --> Cluster
    CFAR --> Dashboard[FmcwDashboardOp]
    Features --> Dashboard
    Range --> Dashboard
    Doppler --> Dashboard
    Dashboard --> Image[RgbaImageBridgeOp] --> Holoviz[HSDK 5 HolovizOp]
```

## Run from the host

From the repository root, the CLI builds the app and runs it in a container:

```bash
./holoscan-radar run simo --run-args=simo_smoke.yaml
```

This runs the two-frame headless smoke preset. Run
`./holoscan-radar run simo` on a host with a display for the normal preset.

The normal preset enables relative monostatic free-space range loss with a 100 m reference. The smoke preset intentionally omits it, uses the constant-amplitude analytic model, and emits a warning.

Simulator-backed sources default `app.scene_visibility` to `front_half_space`, with radar-local `+x` as boresight. Select `frustum` with explicit horizontal/vertical FOVs in radians, or `all` to restore isotropic target synthesis. This binary coverage gate is separate from relative range loss and does not model antenna gain.

## Python Graph

The HSDK 5 Python provider uses the same C++ YAML loader, operator ports,
device bindings, report sink, and optional dashboard as this C++ application.
Run from the repository root after building or installing the Python wheel:

```python
from holoscan.radar.applications.simo import build_graph

workflow = build_graph("applications/fmcw/simo/config/simo_smoke.yaml")
workflow.run()
```

`build_graph()` returns a `BuiltGraph`; call `.run()` to stop the Graph after
the configured report frame count. The canonical import path is
`holoscan_radar.applications.`simo`.
