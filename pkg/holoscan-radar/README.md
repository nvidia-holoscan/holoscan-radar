# holoscan-radar package

This directory defines the Debian package for `holoscan-radar`.

## Usage

From the repository root **on the host**; the CLI builds the package in a container:

```bash
./holoscan-radar package holoscan-radar --pkg-generator DEB
```

To build on a CUDA devel image with the Holoscan SDK from the CUDA APT
repository instead, see
[CUDA devel image with the APT SDK](../../DEVELOPER.md#cuda-devel-image-with-the-apt-sdk).

For a custom operator subset in a direct CMake package build, use a fresh build
directory, configure with
`-DBUILD_ALL=OFF -DPKG_holoscan_radar=ON` and set
`-DHOLOSCAN_RADAR_PKG_OPERATORS='clutter_filter_op;cphd_reader_op'` to the desired
operator names. The default list in [`pkg/CMakeLists.txt`](../CMakeLists.txt)
includes all 33 public operators. This curated list defines the expected package
contents; intentional subsets are allowed. CI verifies these selected libraries
in the Debian artifact.

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

The GPU-backed `./holoscan-radar test` driver builds and checks DEB, TGZ, and
wheel artifacts through `holoscan package holoscan-radar --pkg-generator=...`.
The command runs with `--local` inside the existing SDK container. Run only the
packaging tests with
`./holoscan-radar test --ctest-options='-DINCLUDE_LABELS=packaging'`. The Debian
check verifies all operators selected by the curated package list are present
in both the DEB and TGZ artifacts and builds a separate C++ consumer using the
extracted package. The wheel check installs into a separate Python environment and verifies imports, the provider entry point, and Graph
authoring in both precisions. See [DEVELOPER.md](../../DEVELOPER.md) for options
and artifact locations. These checks do not publish artifacts.

## Reuse the wheel smoke check

Inside the SDK container, use the Python environment where the wheel is installed:

```bash
python tests/packaging/wheel_smoke.py --sdk-site /usr/lib/python3.12/site-packages
```

Omit `--sdk-site` when the SDK is already importable. The check requires Radar's
Python facade and native extension to come from the active Python environment;
`--expected-prefix` can explicitly select a different expected install prefix.
CI calls this same file with Python's `-I` isolation flag after installing the wheel.

## metadata.json

`metadata.json` registers this package with the Holoscan CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it through the module root and the `pkg/` search path,
  which makes it appear under the `PACKAGES` section of `./holoscan-radar list`.
- **`package.dockerfile`** — path (relative to the module root) to the Dockerfile
  used for container-based package builds.

Looking for Python packaging? Review the project [pyproject.toml](../../pyproject.toml)
for configuration.

## Consuming the installed C++ package

Point `CMAKE_PREFIX_PATH` at the Radar installation and the matching Holoscan SDK:

```cmake
find_package(holoscan-radar CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE holoscan::fmcw_range_processor_op)
```

All built public targets share `holoscan_radar_targets.cmake`; individual target
names are unchanged. Discovery also works with Debian packaging disabled and
with intentional operator subsets. The package has no feature components.
Consumers that manually included the old per-target export files must migrate
to `find_package`; those files are no longer installed. Use a fresh installation
prefix when upgrading to avoid retaining stale exports. CUDA, MatX, and the SDK
remain required; installations containing CPHD also resolve tinyxml2 and Threads.

For Debian packages containing CPHD, the package dependency on `libtinyxml2-dev`
installs tinyxml2's CMake configuration and runtime library. A CPHD-free build
omits this dependency. The full package resolves tinyxml2 during `find_package`,
even when the application uses only FMCW operators.

TGZ archives do not install system dependencies. Before using an archive that
contains CPHD, install tinyxml2's development package yourself, in addition to
the matching Holoscan SDK and CUDA toolkit. On Debian/Ubuntu:

```bash
sudo apt-get install libtinyxml2-dev
```
