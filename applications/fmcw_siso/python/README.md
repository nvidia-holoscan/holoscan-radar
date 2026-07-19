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
