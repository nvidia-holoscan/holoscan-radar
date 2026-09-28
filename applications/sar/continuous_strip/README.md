# Continuous Strip SAR application

The scroll stage slices reader batches at the emission cadence and places each stream boundary on the last slice. Backprojection scrolls its lattice; the mosaic accumulates emitted windows.

```mermaid
flowchart LR
    R[CPHDReaderOp] --> RC[SARRangeCompressionOp]
    R -- positions / reference / boundary --> S[ScrollGeometryStagerOp]
    RC -- profiles --> S
    S -- profiles / positions / reference / boundary --> A[ApertureManagerOp]
    S -- geometry --> B[BackprojectionOp: scroll]
    A -- profiles / positions / control --> B
    B --> M[SarMosaicOp]
    M --> Report[Image report]
    M -. dump .-> Dump[TensorDumpOp]
    M -. visualize .-> Raster[CUDA SAR raster]
    Raster --> Viz[HolovizOp]
```

Build and run in the pinned HSDK 5 GPU image:

```bash
cmake -S . -B build -DBUILD_ALL=OFF -DHOLOSCAN_RADAR_BUILD_PYTHON=OFF -DHOLOSCAN_RADAR_BUILD_TESTING=OFF -DAPP_continuous_strip=ON
cmake --build build --target continuous_strip
./build/applications/sar/continuous_strip/cpp/continuous_strip continuous_strip.yaml /path/to/collection.cphd
```

The default [`continuous_strip.yaml`](config/continuous_strip.yaml) ships beside the executable.
The first argument may be a YAML path or a CPHD path; a second CPHD path
replaces the YAML `cphd.path`. `visualize: true` opens Holoviz when a usable
X11 display exists. `headless: true` renders offscreen. `dump: true` writes
complex `.npy` images through TensorDumpOp.

CTest generates one CPHD fixture in C++ and checks image counts and dimensions
for this executable and its headless visualization path. Real collection and
interactive display validation remain open in the untracked root `OPEN_ITEMS.md`.
The six reusable SAR operators are documented under [`operators/sar/`](../../../operators/sar/README.md)
and the CPHD reader under [`operators/io/`](../../../operators/io/README.md).
