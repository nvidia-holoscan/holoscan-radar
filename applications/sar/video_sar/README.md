# Video Sar SAR application

A retained pulse window emits at a configured cadence. The manager sends signed eviction weights and backprojection continues the image across frames.

```mermaid
flowchart LR
    R[CPHDReaderOp] --> RC[SARRangeCompressionOp]
    R -- positions / reference --> E[Optional ECEF to scene ENU]
    R -- boundary --> A[ApertureManagerOp: retained window]
    RC -- profiles --> A
    E --> A
    A -- profiles / positions / control / signed weight --> B[BackprojectionOp]
    B --> Report[Image report]
    B -. dump .-> Dump[TensorDumpOp]
    B -. visualize .-> Raster[CUDA SAR raster]
    Raster --> Viz[HolovizOp]
```

Build and run in the pinned HSDK 5 GPU image:

```bash
cmake -S . -B build -DBUILD_ALL=OFF -DHOLOSCAN_RADAR_BUILD_PYTHON=OFF -DHOLOSCAN_RADAR_BUILD_TESTING=OFF -DAPP_video_sar=ON
cmake --build build --target video_sar
./build/applications/sar/video_sar/cpp/video_sar video_sar.yaml /path/to/collection.cphd
```

The default [`video_sar.yaml`](config/video_sar.yaml) ships beside the executable.
The first argument may be a YAML path or a CPHD path; a second CPHD path
replaces the YAML `cphd.path`. `visualize: true` opens Holoviz when a usable
X11 display exists. `headless: true` renders offscreen. `dump: true` writes
complex `.npy` images through TensorDumpOp.

CTest generates one CPHD fixture in C++ and checks image counts and dimensions
for this executable and its headless visualization path. Real collection and
interactive display validation remain open in the untracked root `OPEN_ITEMS.md`.
The six reusable SAR operators are documented under [`operators/sar/`](../../../operators/sar/README.md)
and the CPHD reader under [`operators/io/`](../../../operators/io/README.md).
