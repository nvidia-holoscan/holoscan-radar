# FMCW SIMO — Python demo

Python Holoviz demo mirroring [`../cpp/fmcw_simo.cpp`](../cpp/fmcw_simo.cpp). The
graph wires sparse-path angle estimation (SnapshotGather + Beamscan), detection,
clustering, and optional visualization from `holoscan.radar`.

## Prerequisites

Build the module with Python bindings enabled:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Holoscan SDK and a CUDA-capable GPU are required.

## Run

From the repository root:

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_simo/python/fmcw_simo.py [yaml]
```

Default config: [`../cpp/fmcw_simo.yaml`](../cpp/fmcw_simo.yaml).

Headless smoke (2 frames, no Holoviz):

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_simo/python/fmcw_simo.py \
  applications/fmcw_simo/python/fmcw_simo_smoke.yaml
```

`fmcw_simo.py` inserts `build/python/lib/holoscan` into `holoscan.__path__` when
`HOLOSCAN_RADAR_BUILD_DIR` is set.

## Layout

Same layering as the C++ side (see [`../README.md`](../README.md) and
[`../../common/README.md`](../../common/README.md)):

- `fmcw_simo.py` — the graph only: `FmcwSimoApp.compose()` plus a `main()`
  that calls the shared `common.python.driver.run_fmcw_app()`. Every operator
  block flows through Holoscan's `from_config()` (no manual per-app parsing).
- `config.py` — colocated, app-owned: `AppConfig` (inherits
  `demo_config.CommonRunConfig`) + `load_config()`. SIMO has no
  `custom_ops.py` — it uses the shared `SimAdcSourceOp` (rank-3 cube) as-is.
- Reusable across all four demos, in `applications/common/python/`: the
  point-target simulator (`sim.py`), the shared ADC source / port-drain
  operators (`ops.py`), the entry-point scaffolding (`driver.py`), and the
  YAML-parsing helpers (`demo_config.py`).
