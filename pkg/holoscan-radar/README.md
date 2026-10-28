# holoscan-radar package

This directory defines the Debian package for `holoscan-radar`.

## Usage

From the repository root **on the host**; the CLI builds the package in a container:

```bash
./holoscan-radar package holoscan-radar --pkg-generator DEB
```

For a custom operator subset in a direct CMake package build, configure with
`-DBUILD_ALL=OFF -DPKG_holoscan_radar=ON` and set
`-DHOLOSCAN_RADAR_PKG_OPERATORS='clutter_filter_op;cphd_reader_op'` to the desired
operator names. The default list in [`pkg/CMakeLists.txt`](../CMakeLists.txt)
includes all 32 public operators.

## metadata.json

`metadata.json` registers this package with the Holoscan CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it through the module root and the `pkg/` search path,
  which makes it appear under the `PACKAGES` section of `./holoscan-radar list`.
- **`package.dockerfile`** — path (relative to the module root) to the Dockerfile
  used for container-based package builds.

Looking for Python packaging? Review the project [pyproject.toml](../../pyproject.toml)
for configuration.
