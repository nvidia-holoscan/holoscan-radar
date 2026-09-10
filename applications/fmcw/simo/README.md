# FMCW SIMO applications

This application presents the two conventional-beamforming SIMO workflows by
their processing order. The names avoid overloading *sparse* and *dense*, which
can also describe array geometry, recovery algorithms, or MIMO processing.

## Workflows

`fmcw_simo_detect_then_beamform` detects on the non-coherently integrated
range-Doppler map, gathers only detected channel snapshots, and then runs
Beamscan:

```mermaid
flowchart LR
    ADC[ADC source] --> Range[Range processor]
    Range --> Clutter[Clutter filter]
    Clutter --> Doppler[Doppler processor]
    Doppler -- RD maps --> CFAR[CFAR<br/>NCI over RX]
    Doppler -- RD maps --> Gather[Snapshot gather]
    CFAR -- detection indices --> Gather
    Gather --> Beamscan[Beamscan estimator]
    CFAR --> Features[Feature assembly]
    Range -- range bins --> Features
    Doppler -- velocity bins --> Features
    Beamscan -- AoA estimates --> Features
    Features --> DBSCAN[DBSCAN]
    Features --> Reduction[Detection reduction]
    CFAR --> Reduction
    DBSCAN --> Reduction
    Features --> Report[Cluster report]
    DBSCAN --> Report
    Reduction --> Report
    CFAR -- CFAR power --> Heatmap[RD heatmap adapter]
    CFAR -- detection indices --> Geometry[Cluster geometry adapter]
    DBSCAN -- labels --> Geometry
    Reduction -- representatives --> Geometry
    Heatmap --> Holoviz[HSDK 5 Holoviz]
    Geometry -- white ovals and crosses --> Holoviz
```

This is the workflow called the *sparse angle path* in
[`WORKFLOWS.md`](../../../docs/planning/WORKFLOWS.md).

`fmcw_simo_beamform_then_detect` beamforms every range-Doppler cell before
detection:

```mermaid
flowchart LR
    ADC[ADC source] --> Range[Range processor]
    Range --> Clutter[Clutter filter]
    Clutter --> Doppler[Doppler processor]
    Doppler --> Beamformer[Phase-shift beamformer]
    Beamformer --> CFAR[CFAR per angle plane]
    CFAR --> Features[Feature assembly]
    Beamformer -- angle grids --> Features
    Range -- range bins --> Features
    Doppler -- velocity bins --> Features
    Features --> DBSCAN[DBSCAN]
    Features --> Reduction[Detection reduction]
    CFAR --> Reduction
    DBSCAN --> Reduction
    Features --> Report[Cluster report]
    DBSCAN --> Report
    Reduction --> Report
    CFAR -- CFAR power --> Heatmap[RD max projection<br/>and RGBA adapter]
    CFAR -- detection indices --> Geometry[Cluster geometry adapter]
    DBSCAN -- labels --> Geometry
    Reduction -- representatives --> Geometry
    Heatmap --> Holoviz[HSDK 5 Holoviz]
    Geometry -- white ovals and crosses --> Holoviz
```

This is the workflow called the *dense angle path* in `WORKFLOWS.md`.

Both applications report clustered detections and their selected
representatives to the log. When visualization is enabled, the stock HSDK 5
Holoviz sink shows the CFAR power projected onto the range-Doppler plane,
clustered detections as white ovals, and representatives as white crosses.
The fresh adapters in `cpp/viz/` do only the conversions required by the
current native image and geometry layer formats; they do not use the legacy
`radar_app/viz` implementation.

## Layout

- `cpp/detect_then_beamform/` and `cpp/beamform_then_detect/` contain only
  graph entry points.
- `cpp/config_schema.hpp` declares the typed app-owned HSDK parameter schema.
- `cpp/config.hpp` decodes app/operator parameters and owns exact pool sizing.
- `config/default.yaml` is the full scenario; `config/smoke.yaml` is a short
  validation run.
- `python/` is an intentionally empty package stub. Python applications are
  deferred until the HSDK 5 Python surface is ready.

The former sparse `applications/fmcw_simo` application has been removed.
`applications/fmcw_dense_simo` remains temporarily as a behavior-comparison
reference, but CMake does not register or package it.

## Build and run

```bash
cmake -S . -B build -DBUILD_ALL=OFF -DAPP_simo=ON
cmake --build build --target fmcw_simo_detect_then_beamform
cmake --build build --target fmcw_simo_beamform_then_detect

./build/applications/fmcw/simo/cpp/fmcw_simo_detect_then_beamform
./build/applications/fmcw/simo/cpp/fmcw_simo_beamform_then_detect
```

The HoloHub CLI project name follows the leaf application directory:

```bash
./holohub run simo detect_then_beamform --local
./holohub run simo beamform_then_detect --local
```

`default.yaml` enables the display and therefore needs a working `DISPLAY`;
`smoke.yaml` disables it. From the host, the HSDK 5 development image can be
selected explicitly:

```bash
HOLOSCAN_CLI_SEARCH_PATH=applications/fmcw/simo/cpp \
  ./holohub run simo detect_then_beamform \
  --base-img holoscan-radar-hsdk5-ea:<pin> --as-root --no-docker-build
```

Each executable defaults to the staged `default.yaml` and accepts a config path
as its sole argument. For a short smoke:

```bash
./build/applications/fmcw/simo/cpp/fmcw_simo_detect_then_beamform \
  applications/fmcw/simo/config/smoke.yaml
./build/applications/fmcw/simo/cpp/fmcw_simo_beamform_then_detect \
  applications/fmcw/simo/config/smoke.yaml
```

## Configuration

The YAML files put app-owned waveform, scene, run, array, and `visualization`
settings under `app:`. Visualization controls the window size/title, requested
display rate, relative dB range, oval radius, and representative cross size.
Operator blocks remain separate:
`detect_then_beamform` and `beamform_then_detect` own their incompatible
settings, while the range/clutter/Doppler front end is shared. HSDK
`config_io::load()` parses each file once; schema-backed decoding produces
`PreparedParams<T>` values that retain source evidence through
`OperatorHandle::set_params()`. In particular, detection-first CFAR requires
`nci_axes: [0]`, while beamforming-first CFAR requires `nci_axes: []`.

The RX geometry and sparse CFAR `nci_fold` are derived from `app:` and supplied
as decode baselines. They must not also be authored in operator blocks.

Angle grids are also variant-local. The full-cube workflow's grid directly
multiplies its RDA allocation, so its configured grid and
`MaxTensorExtents` must remain intentionally bounded.

Both display-enabled workflows have been exercised end to end against the
currently pinned `main-5x` SDK. The Holoviz sink runs in its own execution
partition and consumes drop-oldest queues so rendering never backpressures the
radar pipeline.
