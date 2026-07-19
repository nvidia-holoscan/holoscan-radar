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

App-local Python ops: `SimoAdcSourceOp`, `AnglePeakPickOp`, `CubeAoaLookupOp`,
`DenseClusterReportOp`, `TrailingIdxColumnsOp`.
