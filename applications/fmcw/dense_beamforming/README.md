# Dense beamforming

Beamform-then-detect FMCW example. This implementation happens to use a SIMO
array, but the application is named for the processing approach it demonstrates. It forms the full range-Doppler-angle cube, applies N-D CFAR peak suppression, resolves physical coordinates, and reports the retained detections directly.

```mermaid
flowchart LR
    Source[Sim ADC source] --> Range[Range processor] --> Clutter[Clutter filter] --> Doppler[Doppler processor]
    Doppler --> Beamform[Phase-shift beamformer] --> CFAR[CFAR detector] --> Features[Feature assembly] --> Report[Detection report]
    Range --> Features
    Doppler --> Features
    Beamform --> Features
    CFAR --> Dashboard[FmcwDashboardOp]
    Features --> Dashboard
    Range --> Dashboard
    Doppler --> Dashboard
    Dashboard --> Image[RgbaImageBridgeOp] --> Holoviz[HSDK 5 HolovizOp]
```

## Run from the host

From the repository root, the CLI builds the app and runs it in a container:

```bash
./holoscan-radar run dense_beamforming --run-args=dense_beamforming_smoke.yaml
```

This runs the two-frame headless smoke preset. Run
`./holoscan-radar run dense_beamforming` on a host with a display for the normal preset.

The normal preset enables relative monostatic free-space range loss with a 100 m reference. The smoke preset intentionally omits it, uses the constant-amplitude analytic model, and emits a warning.

Simulator-backed sources default `app.scene_visibility` to `front_half_space`, with radar-local `+x` as boresight. Select `frustum` with explicit horizontal/vertical FOVs in radians, or `all` to restore isotropic target synthesis. This binary coverage gate is separate from relative range loss and does not model antenna gain.

## Python Graph

The HSDK 5 Python provider uses the same C++ YAML loader, operator ports,
device bindings, report sink, and optional dashboard as this C++ application.
From the repository root, the host CLI builds and runs the colocated
`python/` graph with the pinned development image:

```bash
holoscan run dense_beamforming --language python --img holoscan-radar-dev:hsdk5 \
  --no-docker-build --run-args=dense_beamforming_smoke.yaml
```

After building or installing the Python wheel, call the graph directly:

```python
from holoscan.radar.applications.dense_beamforming import build_graph

workflow = build_graph("applications/fmcw/dense_beamforming/config/dense_beamforming_smoke.yaml")
workflow.run()
```

`build_graph()` returns a `BuiltGraph`; call `.run()` to stop the Graph after
the configured report frame count. The canonical import path is
`holoscan_radar.applications.dense_beamforming`.
