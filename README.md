# Holoscan Radar

GPU-accelerated FMCW radar signal-processing operators for NVIDIA Holoscan.
Operators are implemented in C++/CUDA on MatX and compose through typed ports,
without pipeline-specific glue between processing stages.

Holoscan Radar currently targets the HSDK 5 `main-5x` development branch. Build
the local SDK image once before opening the dev container or using HoloHub; the
temporary source-build workflow and current packaging limitation are documented
in [`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md).

## Quick start

```bash
./.devcontainer/build-hsdk5-ea-image.sh
./.devcontainer/build-hsdk5-ea-image.sh --print-pin
./holohub run-container --base-img <reported-image>
```

Inside the container:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
./build/applications/fmcw/siso/cpp/siso
```

The SISO executable uses `applications/fmcw/siso/config/siso.yaml` by default.
Pass the smoke configuration explicitly for a two-frame headless run.

## Operator groups

| Group | Operators and role |
| --- | --- |
| [FMCW](operators/fmcw/README.md) | Range/Doppler processing, clutter and notch filtering, TDM/DDM separation and compensation, virtual-array assembly |
| [Detection](operators/detection/README.md) | CFAR, feature assembly, cluster reduction, selective multi-source expansion |
| [Angle estimation](operators/angle_estimation/README.md) | Snapshot gathering, array selection, Beamscan, MVDR, MUSIC, Root-MUSIC, ESPRIT, and dense beamforming |
| [Clustering](operators/clustering/README.md) | DBSCAN over caller-selected detection features |

Every public operator README starts with the same compact reference: purpose,
responsibilities, selection guidance, ports, construction requirements, and
parameters. Precision is fixed per instance (`float` or `double`), and GPU
execution is required.

Shared descriptors and tensor conventions are documented in
[`operators/common/README.md`](operators/common/README.md). The complete operator API reference is [`docs/planning/OPERATOR_API.md`](docs/planning/OPERATOR_API.md).

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

Application READMEs include the exact graph and runnable commands. Shared graph
driving, simulation sources, reporting, and visualization support lives in
[`applications/common`](applications/common/README.md).

## Building

Requirements:

| Dependency | Requirement |
| --- | --- |
| Holoscan SDK | HSDK 5 `main-5x` development image |
| CUDA Toolkit | 13.x |
| CMake | 3.31 or newer |
| C++ compiler | C++20-capable GCC 11 or newer |
| NVIDIA GPU | Required for operators, applications, and C++ tests |

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Python bindings are temporarily unavailable because the current SDK source
image does not export the binding component they require. See
[`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md).

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

The performance binaries report cold, steady-state, and applicable live-update
latencies. Results are meaningful only on the machine and GPU where they were
captured.

## Project references

- [`DEVELOPER.md`](DEVELOPER.md): build, test, lint, and contribution workflow
- [`OPEN_ITEMS.md`](OPEN_ITEMS.md): consolidated deferred work
- [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md): supported radar chains and port wiring
- [`docs/planning/TESTING.md`](docs/planning/TESTING.md): test strategy and analytic ground truth

## License

Apache-2.0; see [`LICENSE`](LICENSE).
