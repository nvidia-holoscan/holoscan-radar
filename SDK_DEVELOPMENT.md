# Holoscan SDK development setup

Holoscan Radar builds against the public HSDK 5 GA image
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`. This replaces the
pre-release staging image; access to `nvstaging` is no longer required.

The image was verified on an NVIDIA GB10 host as Linux `arm64` with digest
`sha256:faa23a3a2479697fd64babcb27d28cf97f21d5da54d65c92225f8a9b0d99264a`.
Its CMake package is at `/usr/lib/cmake/holoscan`, including the visualization
and testing components.

Use the SDK `main-5x` source branch when an API is not yet documented, but
build against the GA image.

## Prepare the image

From the repository root on the host, install the CLI as described in
[`DEVELOPER.md`](DEVELOPER.md#holoscan-cli). You can pull the SDK image directly:

```bash
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
```

The CLI pulls the GA base image when needed. You can build the project
image separately with `./holoscan-radar build-container`; the run, build, and
test commands also build it when needed. The devcontainer uses the same base
image. Rebuild and reopen it when switching SDK images.

Run the repository's `./holoscan-radar` wrapper **on the host**. It selects
the GA base image and uses the CLI installed in `.venv` when available.
Each command below starts a container to do its work. If a
sibling `../holoscan-sdk` checkout exists, the CLI still mounts it; the
wrapper selects the image SDK for nested builds.

```bash
./holoscan-radar build holoscan-radar --language cpp
./holoscan-radar run siso --language cpp --run-args=siso_smoke.yaml
./holoscan-radar test
```

Run `./holoscan-radar run-container` on the host to open a shell for direct
CMake or CTest work; those commands then run inside the container. See the
[interactive workflow](DEVELOPER.md#inside-a-development-container). The CLI
selects Python when both app languages exist and `--language` is omitted. It
tags images from the project and branch automatically. Do not pass `--local` to
the host commands: that flag runs the build or application on the host instead
of in the container. Start with a clean `build/<app>` tree when switching SDK
images so CMake does not retain old paths.

## Python provider

The GA image installs HSDK 5 Python Core and Visualization under
`/usr/lib/python3.12/site-packages`. Radar's FMCW/SAR provider uses nanobind and
does not require the SDK's former pybind11 CMake target. The development image
installs nanobind 3.0.1. `./holoscan-radar test` enables the provider and its
`pytest_provider` CTest entry by default. For a C++-only test run, pass
`--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Direct CMake builds still
default to Python off; enable them with `-DHOLOSCAN_RADAR_BUILD_PYTHON=ON`.

```bash
./holoscan-radar test
./holoscan-radar run siso --language python --run-args=siso_smoke.yaml
```

The provider covers all 33 public operators: 26 FMCW operators with F32/F64
variants; `SARRangeCompressionOp`, `PhaseCorrectionOp`, and `AutofocusOp` with
F32/F64 variants; and `CPHDReaderOp`, `ApertureManagerOp`, and `BackprojectionOp`
with `F32GeomF32`, `F32GeomF64`, and `F64GeomF64` variants. `TensorDumpOp` is a
single dtype-independent sink without an `extents` argument. Existing FMCW
imports and call signatures are unchanged.

All six SAR applications also have Python graphs with the same YAML,
visualization, dumping, and playback behavior as C++. They default to single
samples with double geometry; `sample_precision: double` selects double for
both. Python owns its signal handlers and polls interrupts while native
execution releases the GIL; SIGINT/SIGTERM stop and join the session before
propagating `KeyboardInterrupt`. The C++ drivers own SIGINT/SIGTERM handlers and poll a lock-free stop flag
from their control loops, then request cooperative session shutdown. They use
caller-managed `run_async(plan)`, so profiler-created startup threads do not
conflict with HSDK managed-signal admission. Window closure and Escape request
graph shutdown directly. In-flight GPU work must finish before cleanup returns.

For an operator trace, run the built executable inside the development container:

```bash
nsys profile --trace=cuda,nvtx --sample=none --backtrace=none --cpuctxsw=none \
  -o /tmp/siso-profile build/holoscan-radar/applications/fmcw/siso/cpp/siso \
  build/holoscan-radar/applications/fmcw/siso/cpp/siso_smoke.yaml
```

Use a fresh output name and the actual generated preset path for your build.
No profiling-specific application flag or signal-mask changes are required.

Python `[live]` enum inputs remain unavailable because the SDK cannot register
those port payloads; the corresponding C++ contracts are unchanged.

## When the SDK image changes

Update the image reference in the root `Dockerfile`,
`.devcontainer/devcontainer.json`, `holoscan-radar` wrapper, and this guide together. Confirm the new
image's architecture, `holoscan_DIR`, visualization and testing components, and
Python Core and Visualization packages. Then rebuild the development image and
run the build, CTest, and application smoke checks described in `DEVELOPER.md`.
