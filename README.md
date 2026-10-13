# Holoscan Radar

GPU-accelerated FMCW and SAR signal-processing operators for NVIDIA Holoscan.
Operators are implemented in C++/CUDA on MatX and compose through typed ports,
without pipeline-specific glue between processing stages.

Holoscan Radar targets HSDK 5. Pull the staging SDK image before opening the dev
container or using the Holoscan CLI. This pre-release image requires
`nvstaging` as your user org in NGC Catalog. Setup and the current packaging
limitation are documented in [`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md).

## Quick start

Install the development CLI as described in [`DEVELOPER.md`](DEVELOPER.md#holoscan-cli),
then use the `./holoscan-radar` wrapper from the repository root. It selects the
staging image and runs the project build and application in the named container:

```bash
docker pull nvcr.io/nvstaging/holoscan/holoscan:v5.0.0.7-cuda13
./holoscan-radar build-container --img holoscan-radar-dev:hsdk5
./holoscan-radar build siso --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --configure-args=-DBUILD_ALL=OFF \
  --configure-args=-DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
  --configure-args=-Dholoscan_DIR=/usr/lib/cmake/holoscan
./holoscan-radar run siso --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --configure-args=-DBUILD_ALL=OFF \
  --configure-args=-DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
  --configure-args=-Dholoscan_DIR=/usr/lib/cmake/holoscan \
  --run-args=siso_smoke.yaml
```

The `--run-args=siso_smoke.yaml` example runs two headless frames. Omit that
flag to use the SISO executable's default `siso.yaml` configuration.

For an offline SAR smoke run, configure testing and run the generated-fixture
CTest cases. The five shipped SAR YAML files are real-collection presets and
require local CPHD data before direct execution:

```bash
ctest --test-dir build -R '^Sar_.*_(smoke|viz)$' --output-on-failure
```

Start with the [SAR application index](applications/sar/README.md), then the
[CPHD preparation guide](tools/sar_data/README.md) for real data. The C++
fixture checks graph execution; it does not validate focus on public collects.

## Operator groups

| Group | Operators and role |
| --- | --- |
| [FMCW](operators/fmcw/README.md) | Range/Doppler processing, clutter and notch filtering, TDM/DDM separation and compensation, virtual-array assembly |
| [Detection](operators/detection/README.md) | CFAR, feature assembly, cluster reduction, selective multi-source expansion |
| [Angle estimation](operators/angle_estimation/README.md) | Snapshot gathering, array selection, Beamscan, MVDR, MUSIC, Root-MUSIC, ESPRIT, and dense beamforming |
| [Clustering](operators/clustering/README.md) | DBSCAN over caller-selected detection features |
| [SAR](operators/sar/README.md) | Range compression, aperture management, backprojection, and phase correction |
| [SAR I/O](operators/io/README.md) | CPHD decoding and diagnostic tensor dumps |

Every public operator README starts with the same compact reference: purpose,
responsibilities, selection guidance, ports, construction requirements, and
parameters. Precision is fixed per instance (`float` or `double`), and GPU
execution is required.

Shared descriptors and tensor conventions are documented in
[`operators/common/README.md`](operators/common/README.md). The FMCW operator API reference is [`docs/planning/OPERATOR_API.md`](docs/planning/OPERATOR_API.md); the current HSDK 5 SAR contract is [`docs/planning/SAR/OPERATOR_API.md`](docs/planning/SAR/OPERATOR_API.md).

## Example applications

| Application | Workflow |
| --- | --- |
| [SISO](applications/fmcw/siso/README.md) | Baseline range-Doppler detection and clustering |
| [SIMO](applications/fmcw/simo/README.md) | Detect first, then sparse angle estimation |
| [Dense beamforming](applications/fmcw/dense_beamforming/README.md) | Beamform every cell, then detect in range-Doppler-angle |
| [MIMO TDM](applications/fmcw/mimo_tdm/README.md) | Deinterleave TDM and compensate detected snapshots |
| [MIMO DDM](applications/fmcw/mimo_ddm/README.md) | Linear or nonlinear dense DDM demultiplexing |
| [Multi-waveform](applications/fmcw/multi_waveform/README.md) | Coherent switching between FMCW profiles |
| [Adaptive AoA](applications/fmcw/adaptive_aoa/README.md) | Route likely multi-source cells to a super-resolution estimator |
| [SAR spotlight](applications/sar/spotlight/README.md) | One reset image from one collection |
| [SAR video](applications/sar/video_sar/README.md) | Retained pulse window and repeated images |
| [SAR tiled strip](applications/sar/tiled_strip/README.md) | Tile formation and parent mosaic |
| [SAR continuous strip](applications/sar/continuous_strip/README.md) | Scrolling lattice and parent mosaic |
| [SAR independent images](applications/sar/independent_images/README.md) | One restaged image per collection |

Application READMEs include the exact graph and runnable commands. Shared graph
driving, simulation sources, reporting, and visualization support lives in
[`applications/common`](applications/common/README.md).

## Building

Requirements:

| Dependency | Requirement |
| --- | --- |
| Holoscan SDK | HSDK 5 staging image |
| CUDA Toolkit | 13.x |
| CMake | 3.31 or newer |
| C++ compiler | C++20-capable GCC 11 or newer |
| NVIDIA GPU | Required for operators, applications, and C++ tests |

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Python operator bindings are temporarily unavailable because the current SDK staging
image does not export the binding component they require. The current CMake build
therefore does not register the Python test suites, and a Python-enabled wheel
cannot be built from this image. See [`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md).

## Testing

```bash
ctest --test-dir build --output-on-failure -LE performance
pre-commit run --all-files
```

Monte Carlo statistical tests carry the `statistical` label. Performance tests
are characterization tools rather than latency gates:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -L performance
./build/performance/cpp/perf_cfar_detector --full
```

The 26 C++ performance binaries read colocated YAML recipes under
`performance/operators/` and report setup, first-frame, steady-state, and
applicable live-update latencies. Results are meaningful only on the machine
and GPU where they were captured. See [performance/README.md](performance/README.md).

## Project references

- [`DEVELOPER.md`](DEVELOPER.md): build, test, lint, and contribution workflow
- [Jira board 41309](https://jirasw.nvidia.com/secure/RapidBoard.jspa?rapidView=41309): deferred product, algorithm, performance, and tooling work
- [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md): supported radar chains and port wiring
- [`docs/planning/TESTING.md`](docs/planning/TESTING.md): test strategy and analytic ground truth

## License

Apache-2.0; see [`LICENSE`](LICENSE).
