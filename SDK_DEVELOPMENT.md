# Holoscan SDK development setup

Holoscan Radar builds against the HSDK 5 staging image
`nvcr.io/nvstaging/holoscan/holoscan:v5.0.0.7-cuda13`. Access to the
`nvstaging` repository is required. The image replaces the former local build
from the SDK `main-5x` source branch.

The image was pulled on an NVIDIA GB10 host as Linux `arm64` with digest
`sha256:16f37addcce05e4013cdefacae6e28ed73706d08d49682af4eabc055ac888591`.
Its CMake package is at `/usr/lib/cmake/holoscan`; it includes the separate
visualization and testing components. Use the SDK source branch when an API is
not yet documented, but build against this image.

## Prepare the image

This pre-release image requires `nvstaging` as your **user org** in NGC Catalog.
Select that user org before authenticating to `nvcr.io`; a successful Docker
login with a different user org does not grant access to the image.

From the repository root on a host with NGC staging access, install the CLI
as described in [`DEVELOPER.md`](DEVELOPER.md#holoscan-cli), then authenticate:

```bash
docker login nvcr.io
```

The CLI pulls the staging base image when needed. You can build the project
image separately with `./holoscan-radar build-container`; the run, build, and
test commands also build it when needed. The devcontainer uses the same base
image. Rebuild and reopen it when switching from the former locally built SDK
image.

Run the repository's `./holoscan-radar` wrapper **on the host**. It selects
the staging base image and uses the CLI installed in `.venv` when available.
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

The staging image installs HSDK 5 Python Core and Visualization under
`/usr/lib/python3.12/site-packages`. Radar's FMCW provider uses nanobind and
does not require the SDK's former pybind11 CMake target. The development image
installs nanobind 3.0.1. `./holoscan-radar test` enables the provider and its
`pytest_provider` CTest entry by default. For a C++-only test run, pass
`--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Direct CMake builds still
default to Python off; enable them with `-DHOLOSCAN_RADAR_BUILD_PYTHON=ON`.

```bash
./holoscan-radar test
./holoscan-radar run siso --language python --run-args=siso_smoke.yaml
```

Python `[live]` enum inputs remain unavailable because the SDK cannot register
those port payloads; the corresponding C++ contracts are unchanged.

## When the SDK image changes

Update the image reference in the root `Dockerfile`,
`.devcontainer/devcontainer.json`, `holoscan-radar` wrapper, and this guide together. Confirm the new
image's architecture, `holoscan_DIR`, visualization and testing components, and
Python Core and Visualization packages. Then rebuild the development image and
run the build, CTest, and application smoke checks described in `DEVELOPER.md`.
