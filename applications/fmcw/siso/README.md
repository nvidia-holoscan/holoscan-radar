# SISO

Baseline FMCW SISO example. It forms a range-Doppler map, applies CFAR, groups neighboring detections in physical range and velocity, and reports one medoid representative per cluster.

```mermaid
flowchart LR
    Source[Sim ADC source] --> Range[Range processor] --> Clutter[Clutter filter] --> Doppler[Doppler processor]
    Doppler --> CFAR[CFAR detector] --> Features[Feature assembly] --> Cluster[DBSCAN] --> Reduce[Detection reduction] --> Report[Cluster report]
    Range --> Features
    Doppler --> Features
    CFAR --> Reduce
```

The optional dashboard displays the range-Doppler heatmap and rolling range and velocity histories. SISO has no angle products, so the spatial and angle-history panels are omitted.

## Build and run

```bash
cmake -S . -B build -DBUILD_ALL=OFF -DAPP_siso=ON
cmake --build build --target siso
./build/applications/fmcw/siso/cpp/siso
```

The executable defaults to `config/siso.yaml`. Pass `config/siso_smoke.yaml` as the first argument for the two-frame, headless single-target smoke scene.
