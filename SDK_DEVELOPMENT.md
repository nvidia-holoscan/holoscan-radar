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

From the repository root on a host with NGC staging access:

```bash
docker login nvcr.io
docker pull nvcr.io/nvstaging/holoscan/holoscan:v5.0.0.7-cuda13
holoscan build-container \
  --base-img nvcr.io/nvstaging/holoscan/holoscan:v5.0.0.7-cuda13 \
  --img holoscan-radar-dev:hsdk5
```

The devcontainer builds from the same staging image. Rebuild and reopen it
when switching from the former locally built SDK image.

If a `../holoscan-sdk` checkout exists, the CLI automatically mounts it. Set
this option in the host shell before running the commands below to use the
image SDK's dependencies throughout:

```bash
export HOLOSCAN_CLI_DEFAULT_DOCKER_RUN_ARGS='-e HOLOSCAN_SDK_ROOT=/usr -e HOLOSCAN_LIB_PATH= -e CMAKE_PREFIX_PATH=/usr -e PYTHONPATH=/usr/lib/python3.12/site-packages'
```

Run the CLI on the host and let it execute builds, applications, and tests in
the named development image:

```bash
holoscan build holoscan-radar --language cpp --img holoscan-radar-dev:hsdk5 \
  --no-docker-build --configure-args=-Dholoscan_DIR=/usr/lib/cmake/holoscan
holoscan run siso --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --configure-args=-DBUILD_ALL=OFF \
  --configure-args=-DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
  --configure-args=-Dholoscan_DIR=/usr/lib/cmake/holoscan \
  --run-args=siso_smoke.yaml
holoscan test --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --cmake-options=-Dholoscan_DIR=/usr/lib/cmake/holoscan
```

`holoscan run-container --img holoscan-radar-dev:hsdk5 --no-docker-build`
opens an interactive shell if needed. Do not pass `--local` to the host commands:
that flag runs the build or application on the host instead of in the container.
The explicit `holoscan_DIR` selects the image CMake package; the environment
override above also makes nested CLI builds select the image SDK and prevents
transitive dependencies from using the mounted checkout. Start with a clean
`build/<app>` tree when switching SDK images so CMake does not retain old paths.

## Current packaging limitation

The staging image provides the Holoscan Python package, but does not export the
CMake binding target required by this module's pybind11 extensions. C++
operators, applications, tests, and performance tools remain supported;
`HOLOSCAN_RADAR_BUILD_PYTHON` is forced off until that SDK component is
available. CMake does not register pytest while bindings are off, and this
image cannot produce a Python-enabled radar wheel. The Python package source
remains in the repository.

## When the SDK image changes

Update the image reference in the root `Dockerfile`,
`.devcontainer/devcontainer.json`, and this guide together. Confirm the new
image's architecture, `holoscan_DIR`, visualization and testing components, and
Python binding target. Then rebuild the development image and run the build,
CTest, and application smoke checks described in `DEVELOPER.md`.
