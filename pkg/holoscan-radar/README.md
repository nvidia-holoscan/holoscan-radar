# holoscan-radar package

This directory defines the Debian package for `holoscan-radar`.

## Usage

From the repository root **on the host**; the CLI builds the package in a container:

```bash
./holoscan-radar package holoscan-radar --pkg-generator DEB
```

For a custom operator subset in a direct CMake package build, use a fresh build
directory, configure with
`-DBUILD_ALL=OFF -DPKG_holoscan_radar=ON` and set
`-DHOLOSCAN_RADAR_PKG_OPERATORS='clutter_filter_op;cphd_reader_op'` to the desired
operator names. The default list in [`pkg/CMakeLists.txt`](../CMakeLists.txt)
includes all 32 public operators. When the Radar package is enabled, unknown
operator names cause a configuration error; intentional subsets are allowed.

The package list enables operators; it does not disable operators enabled by
other build options or clear cached `OP_*` settings. Reducing the list in an
existing build directory can therefore leave previously enabled operators in
the package. For an exact subset, keep `BUILD_ALL=OFF` and avoid enabling
applications, the Python provider, or extra operators that expand the selection.

Package-only configurations default tests to off. Explicit or cached
`HOLOSCAN_RADAR_BUILD_TESTING` settings take precedence; pass
`-DHOLOSCAN_RADAR_BUILD_TESTING=OFF` to disable tests in a reused build tree.
Normal standalone builds keep tests enabled by default.

## CI coverage

The GPU-backed `./holoscan-radar test` driver builds and checks Debian and wheel
artifacts. Run only the packaging tests with
`./holoscan-radar test --ctest-options='-DINCLUDE_LABELS=packaging'`. The Debian
check verifies all registered operators are present and builds a separate C++
consumer using the extracted package. The wheel check installs into a separate
Python environment and verifies imports, the provider entry point, and Graph
authoring in both precisions. See [DEVELOPER.md](../../DEVELOPER.md) for options
and artifact locations. These checks do not publish artifacts.

## metadata.json

`metadata.json` registers this package with the Holoscan CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it through the module root and the `pkg/` search path,
  which makes it appear under the `PACKAGES` section of `./holoscan-radar list`.
- **`package.dockerfile`** — path (relative to the module root) to the Dockerfile
  used for container-based package builds.

Looking for Python packaging? Review the project [pyproject.toml](../../pyproject.toml)
for configuration.
