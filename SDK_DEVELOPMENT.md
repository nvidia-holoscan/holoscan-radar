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
`ee2af259821c0291dcc6a91f9e01a8312b0b4397` (`holoscan-sdk!5397`, stacked on
`holoscan-sdk!5394`). Build that exact source and select its image with:

```bash
./.devcontainer/build-hsdk5-ea-image.sh \
  --pin ee2af259821c0291dcc6a91f9e01a8312b0b4397
./holohub run-container --base-img holoscan-radar-hsdk5-ea:ee2af25
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

To use the image outside the dev container:

```bash
./holohub run-container --base-img <reported-image>
```

Inside the dev container, use `./holohub build --local`, `run --local`, and
`test --local` so HoloHub reuses the installed SDK.

## Current packaging limitation

The `main-5x` SDK source currently does not export the Python binding target
required by this module's pybind11 extensions. C++ operators, applications,
tests, and performance tools are supported; `HOLOSCAN_RADAR_BUILD_PYTHON` is
forced off until that SDK component is available. This is a build-surface
limitation, not an operator contract difference.

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
