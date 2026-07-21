# FMCW SIMO dense — Python demo

Python Holoviz demo mirroring [`../cpp/fmcw_dense_simo.cpp`](../cpp/fmcw_dense_simo.cpp).
Exercises the dense angle path: PhaseShiftBeamformer → CFAR on the RDA cube →
peak-pick → angle lookup → cluster.

## Prerequisites

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

## Run

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py [yaml]
```

Default config: [`../cpp/fmcw_dense_simo.yaml`](../cpp/fmcw_dense_simo.yaml).

Headless smoke (2 frames, no Holoviz):

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py \
  applications/fmcw_dense_simo/python/fmcw_dense_simo_smoke.yaml
```

## Layout

Same layering as the C++ side (see [`../README.md`](../README.md) and
[`../../common/README.md`](../../common/README.md)):

- `fmcw_dense_simo.py` — the graph only: `FmcwDenseSimoApp.compose()` plus a
  `main()` that calls the shared `common.python.driver.run_fmcw_app()`. Every
  operator block flows through Holoscan's `from_config()`.
- `config.py` — colocated, app-owned: `AppConfig` (inherits
  `demo_config.CommonRunConfig`) + `load_config()`.
- `custom_ops.py` — colocated, app-specific: `AnglePeakPickOp`,
  `CubeAoaLookupOp`, `TrailingIdxColumnsOp` (dense-path bridges), and
  `DenseClusterReportOp` (kept local: the Python `ClusterReportOp` binding
  doesn't expose `az_grid` / `el_grid` / `cadence` as read-write the way the
  C++ struct fields do — a binding gap, not fixed here).
- Reusable across all four demos, in `applications/common/python/`: the
  point-target simulator (`sim.py`), the shared ADC source
  (`ops.SimAdcSourceOp`), the entry-point scaffolding (`driver.py`), and the
  YAML-parsing helpers (`demo_config.py`).
