# Holoscan SDK development setup

Holoscan Radar currently builds against the HSDK 5 `main-5x` development
branch. Until a suitable public SDK image is available, the repository keeps
one temporary integration tool: [`.devcontainer/build-hsdk5-ea-image.sh`](.devcontainer/build-hsdk5-ea-image.sh).

This is the single location for temporary SDK-source-build instructions and
known SDK packaging limitations. Operator and application READMEs describe the
public radar behavior only.

## Build the local SDK image

From the repository root:

```bash
./.devcontainer/build-hsdk5-ea-image.sh
```

The script resolves `main-5x`, clones it under the ignored `.cache/` directory,
builds HSDK, installs the separately generated visualization component, and
creates a commit-addressed local image. It requires Docker and SSH access to
the internal SDK repository.

The operator-testing migration is qualified against immutable SDK commit
`79dc6cb48e5b571b6f0a6daf2b5dbc783de01b76` on `main-5x`, after
`holoscan-sdk!5394` and `holoscan-sdk!5397` merged. Build that exact source and
select its image with:

```bash
./.devcontainer/build-hsdk5-ea-image.sh \
  --pin 79dc6cb48e5b571b6f0a6daf2b5dbc783de01b76
holoscan build-container --base-img holoscan-radar-hsdk5-ea:79dc6cb \
  --img holoscan-radar-dev:hsdk5
```

Use `--print-pin` to resolve the current branch tip and image name without
building:

```bash
./.devcontainer/build-hsdk5-ea-image.sh --print-pin
```

`--pin` also accepts a tag or commit. A branch checkout must match the current
remote tip; if it does not, remove that pin's stale `.cache/holoscan-sdk-*`
checkout and rebuild. Update `.devcontainer/devcontainer.json` when the
commit-addressed image name changes.

Run the CLI on the host and let it execute builds, applications, and tests in
the named development image:

```bash
holoscan build-container --base-img <reported-image> --img holoscan-radar-dev:hsdk5
holoscan build holoscan-radar --language cpp --img holoscan-radar-dev:hsdk5 \
  --no-docker-build --configure-args=-Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
holoscan run siso --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --configure-args=-DBUILD_ALL=OFF \
  --configure-args=-DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
  --configure-args=-Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan \
  --run-args=siso_smoke.yaml
holoscan test --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --cmake-options=-Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan
```

`holoscan run-container --img holoscan-radar-dev:hsdk5 --no-docker-build`
opens an interactive shell if needed. Do not pass `--local` to the host commands:
that flag runs the build or application on the host instead of in the container.

## Current packaging limitation

The `main-5x` SDK source currently does not export the Python binding target
required by this module's pybind11 extensions. C++ operators, applications,
tests, and C++ performance tools are supported; `HOLOSCAN_RADAR_BUILD_PYTHON`
is forced off until that SDK component is available. CMake does not register
pytest while bindings are off, and the current image cannot produce a
Python-enabled wheel. The Python package source remains in the repository.
This is a build-surface limitation, not an operator contract difference.

## Removal

When a public SDK image supplies the required HSDK, visualization, testing,
and Python components:

1. Delete `.devcontainer/build-hsdk5-ea-image.sh` and
   `.devcontainer/sdk-installed.Dockerfile`.
2. Remove the local `BASE_IMAGE` override from
   `.devcontainer/devcontainer.json`.
3. Enable and validate the Python build.
4. Remove this document after transferring ordinary build instructions to the
   root README and `DEVELOPER.md`.
