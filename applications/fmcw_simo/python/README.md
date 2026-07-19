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
