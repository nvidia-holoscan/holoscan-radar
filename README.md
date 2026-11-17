# Holoscan Radar

GPU-accelerated FMCW and SAR signal-processing operators for NVIDIA Holoscan.
Operators are implemented in C++/CUDA on MatX and compose through typed ports,
without pipeline-specific glue between processing stages.

Holoscan Radar targets HSDK 5. The Holoscan CLI pulls the public GA SDK image
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` when needed; the
devcontainer uses the same image. Setup and Python provider requirements are
documented in [`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md).

## Quick start

On the **host**, run this from the repository root. The CLI starts a container for the build and app:

```bash
uv sync
./holoscan-radar run siso --run-args=siso_smoke.yaml
```

The CLI builds the project image and SISO application as needed. When an app
has both C++ and Python implementations, the CLI selects Python if
`--language` is omitted. Pass `--language cpp` to run the C++ implementation.
To prepare the image separately, run `./holoscan-radar build-container`. For
the 250-frame visualization preset, run `./holoscan-radar run siso` on a host
with a display.
See [`DEVELOPER.md`](DEVELOPER.md#holoscan-cli) for CLI installation alternatives.

For offline SAR fixture checks, run `./holoscan-radar test` on the host; it
includes the registered SAR smokes. The shipped SAR YAML files are real-data
presets and need local CPHD data before direct execution. See the
[SAR application index](applications/sar/README.md) and
[CPHD preparation guide](tools/sar_data/README.md). The fixture checks graph
execution, not focus on public collections.

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

## Building and testing

Run these commands from the **host**; the CLI builds and runs in containers:

```bash
./holoscan-radar build holoscan-radar --language cpp
./holoscan-radar test
```

The development image supplies the HSDK 5 SDK, CUDA toolkit, and C++ build tools.
A CUDA-capable NVIDIA GPU is required for operators, apps, and tests. The test
command includes the Python provider and its pytest suite by default. To
run only the C++ tests, pass `--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`.
To run a Python application:

```bash
./holoscan-radar run siso --language python --run-args=siso_smoke.yaml
```

The Python provider is optional in ordinary C++ builds; see
[`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md) for SDK components and packaging.
It exposes all 33 public operators through 68 native registrations under
`holoscan.radar` and thirteen Python Graph workflows through
`holoscan_radar.applications`: seven FMCW and six SAR. See the
[SAR Python guide](applications/sar/README.md#python-applications) for precision
pairs, input data, visualization, and CLI examples.

For manual CMake and targeted CTest work **inside** a development container,
see [`DEVELOPER.md`](DEVELOPER.md#inside-a-development-container). The same
guide covers host-side lint. Performance measurements and their container-side
build commands are documented in [`DEVELOPER.md`](DEVELOPER.md#performance-characterization)
and [performance/README.md](performance/README.md).

## Project references

- [`DEVELOPER.md`](DEVELOPER.md): build, test, lint, and contribution workflow
- [Jira board 41309](https://jirasw.nvidia.com/secure/RapidBoard.jspa?rapidView=41309): deferred product, algorithm, performance, and tooling work
- [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md): supported radar chains and port wiring
- [`docs/planning/TESTING.md`](docs/planning/TESTING.md): test strategy and analytic ground truth

## License

Apache-2.0; see [`LICENSE`](LICENSE).
