# FMCW SISO — Python demo

Python Holoviz demo mirroring [`../cpp/fmcw_siso.cpp`](../cpp/fmcw_siso.cpp). The
graph wires the same signal-processing, detection, clustering, and optional
visualization operators from `holoscan.radar`.

## Prerequisites

Build the module with Python bindings enabled (same as the pytest suite):

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Holoscan SDK and a CUDA-capable GPU are required.

## Run

From the repository root:

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
# or prepend build/python/lib to holoscan path as conftest does
python applications/fmcw_siso/python/fmcw_siso.py [yaml]
```

Default config: [`../cpp/fmcw_siso.yaml`](../cpp/fmcw_siso.yaml).

Headless smoke (2 frames, no Holoviz):

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_siso/python/fmcw_siso.py \
  applications/fmcw_siso/python/fmcw_siso_smoke.yaml
```

`fmcw_siso.py` inserts `build/python/lib/holoscan` into `holoscan.__path__` when
`HOLOSCAN_RADAR_BUILD_DIR` is set so `import holoscan.radar` resolves to the
build-tree extensions.

## Layout

Same layering as the C++ side (see [`../README.md`](../README.md) and
[`../../common/README.md`](../../common/README.md)):

- `fmcw_siso.py` — the graph only: `FmcwSisoApp.compose()` plus a `main()`
  that calls the shared `common.python.driver.run_fmcw_app()`.
- `config.py` — colocated, app-owned: `AppConfig` (inherits
  `demo_config.CommonRunConfig`) + `load_config()`. SISO has no
  `custom_ops.py` — it uses the shared `SimAdcSourceOp` (rank-2 cube) as-is.
- Reusable across all four demos, in `applications/common/python/`: the
  point-target simulator (`sim.py`), the shared ADC source
  (`ops.SimAdcSourceOp`), the entry-point scaffolding (`driver.py`), and the
  YAML-parsing helpers (`demo_config.py`).
