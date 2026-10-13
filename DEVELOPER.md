# Developer Guide — Holoscan Radar

This guide covers the layout, build system, and day-to-day workflow for developing and
distributing this Holoscan Module.

---

## Module layout

```text
holoscan-radar/
├── pyproject.toml                  # CLI, CTest, and wheel configuration
├── Dockerfile                      # Development container image
├── CMakeLists.txt                  # Root CMake — orchestrates operators/applications/tests
├── metadata.json                   # Module-level metadata (schema: urn:holohub:module:v2)
├── operators/
│   ├── common/                     # Shared radar_common library
│   │   └── radar_common/           # Organized by dependency layer:
│   │       ├── descriptors/        #   shared config-plane types (STL only)
│   │       ├── dsp/                #   host-side signal math
│   │       ├── sim/                #   analytic point-target simulator (STL only)
│   │       ├── tensor/             #   tensor/DLPack/MatX plumbing
│   │       └── sdk/                #   Holoscan-integration glue
│   ├── sar/                        # SAR formation operators
│   ├── io/                         # CPHD reader and tensor dump
│   ├── detection/                  # CFAR, feature assembly, and reduction
│   ├── clustering/                 # DBSCAN
│   ├── angle_estimation/           # Sparse estimators and dense beamforming
│   └── fmcw/                       # FMCW signal-processing group
│       └── <operator_name>/
│           ├── <operator_name>.hpp / .cu   # Operator + stage kernel
│           ├── <operator_name>_stage.hpp   # (stage header name varies)
│           └── metadata.json       # Operator-level metadata
├── applications/
│   ├── common/radar_app/            # Shared Graph/config/report/dashboard support
│   ├── sar/                        # Five standalone C++ formation apps
│   │   ├── spotlight/
│   │   ├── video_sar/
│   │   ├── tiled_strip/
│   │   ├── continuous_strip/
│   │   └── independent_images/
│   └── fmcw/
│       ├── siso/
│       ├── simo/
│       ├── dense_beamforming/
│       ├── mimo_tdm/
│       ├── adaptive_aoa/
│       ├── multi_waveform/
│       └── mimo_ddm/
├── tools/sar_data/               # Optional CPHD download/subset CLIs
├── python/holoscan/radar/
│   └── __init__.py                 # Operator exports; compiled bindings currently disabled
├── performance/                   # C++ YAML recipes, adapters, and shared runner
└── tests/
    ├── cpp/                        # GTest suite (+ utils/: fixtures and Graph helpers)
    └── python/                     # pytest suite (requires compiled bindings)
```

---

## Holoscan CLI

The development-only `cli` and `test` groups in [`pyproject.toml`](pyproject.toml)
pin the Holoscan CLI and Python CTest dependencies. With Python 3.11+ available,
install them from the repository root using uv:

```bash
uv sync
./holoscan-radar version
```

The project pins the CLI to NVIDIA's package index for uv. `uv sync` installs
only these development tools; it does not compile the radar wheel. The groups
are not part of the published wheel. With pip 25.1+, the equivalent setup is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip>=25.1'
.venv/bin/python -m pip install --extra-index-url https://pypi.nvidia.com --group cli --group test
./holoscan-radar version
```

The development Dockerfile installs the same groups. From the repository root,
use `./holoscan-radar` to run the pinned CLI; the wrapper prefers `.venv`
without requiring activation. Authenticate to NGC as described in
[`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md), then use the commands below. The CLI
builds the project image when needed and tags it from the project and branch.
`./holoscan-radar build-container` builds the image separately if desired.

| Command | What it does |
| --- | --- |
| `./holoscan-radar build holoscan-radar --language cpp` | Build the C++ module in a container |
| `./holoscan-radar build simo` | Build SIMO and its dependencies in a container |
| `./holoscan-radar run simo` | Build and run SIMO in a container |
| `./holoscan-radar test` | Build and run the test suite in a container |
| `./holoscan-radar package holoscan-radar` | Build the Debian package in a container |
| `./holoscan-radar run-container` | Open an interactive container shell |

Each host command starts a container; no interactive shell is required for
builds or tests. The CLI selects the required app dependencies, and the wrapper
selects the image SDK for nested builds when a sibling `../holoscan-sdk`
checkout exists. The CLI still mounts that sibling checkout. Start with a clean
`build/<app>` tree when switching SDK images because CMake caches dependency
paths. `[tool.holoscan]` selects [`ci/container.ctest`](ci/container.ctest) for
`./holoscan-radar test`. To try an unpublished CLI checkout, install it into
the selected environment and run its `holoscan` command there.

---

## Building directly with CMake inside the container

For manual debugging in an interactive development container:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Run C++ tests:

```bash
ctest --test-dir build --output-on-failure -LE performance
```

Run all five SAR fixture and headless visualization paths without public data:

```bash
ctest --test-dir build -R '^Sar_.*_(smoke|viz)$' --output-on-failure
```

For individual SAR build gates use `-DBUILD_ALL=OFF -DAPP_spotlight=ON`
(or `APP_video_sar`, `APP_tiled_strip`, `APP_continuous_strip`,
`APP_independent_images`). See the [SAR app index](applications/sar/README.md)
for mode-specific graphs, [data preparation](tools/sar_data/README.md) for
optional public collections, and the [HSDK 5 SAR contract](docs/planning/SAR/OPERATOR_API.md).
The fixture is generated in C++; Python is required only for optional data
preparation. All five SAR GUI apps have been visually checked with their
current presets. That check does not establish full-resolution image quality
or validate every public collection and backprojection backend.

Python bindings are disabled in the current SDK image because it does not
export the required binding target. Consequently, CMake does not register
`pytest` or `pytest_statistical`; the direct pytest suite also requires
compiled bindings and is not a current build gate. Once the SDK supplies the
target and bindings are enabled, run the Python suite with:

```bash
# conftest.py extends the installed SDK package with build-tree modules.
# Do not put build/python/lib alone on PYTHONPATH: it shadows the SDK package.
HOLOSCAN_RADAR_BUILD_DIR=build pytest tests/python/ -v
```

---

## Running the full suite with `./holoscan-radar test`

From the host, `./holoscan-radar test` builds the project image if needed,
configures and builds the module, and runs the suite in a container:

```bash
./holoscan-radar test
./holoscan-radar test --dryrun --verbose
```

The wrapper selects the staging base image and handles the sibling SDK
checkout when one exists. The CLI finds the image SDK's CMake package without
an explicit `holoscan_DIR` override.

What it does, and where to change it:

- Runs [`ci/container.ctest`](ci/container.ctest), this module's CTest driver, selected
  by `pyproject.toml`. The driver bundled with `holoscan-cli` resolves its source directory
  inside the pip install tree and cannot see this repo.
- Builds into **`build-test/`**, a dedicated tree. The driver wipes it on every run, which
  is why it is deliberately not your `build/` directory.
- Runs every registered C++ tier except `performance`, which has its own invocation.
  That includes the Monte Carlo `statistical` tests. Python suites are registered
  only when bindings are enabled; they are absent with the current SDK image.
  Narrow to one tier with
  `--ctest-options='-DINCLUDE_LABELS=statistical'`, or widen the exclusion with
  `-DEXCLUDE_LABELS='performance|statistical'`.
- The selection is **exclude-based on purpose**: an include list silently drops any label
  nobody remembered to add, which is how the `integration` tier went unrun in CI for the
  life of the suite. A new tier now runs the day it is added.
- Writes JUnit results to `build-test/test-results.xml`.
- Uses `ccache` (`CCACHE_DIR` defaults to the gitignored `.cache/ccache`), so repeat runs
  reuse compiled objects despite the wipe. The first run is a full cold build.

Useful escapes:

| Flag | Effect |
| --- | --- |
| `--no-xvfb` | Skip the default `xvfb-run -a` wrapper |
| `--cmake-options OPT` | Append a CMake configure flag (repeatable) |
| `--ctest-options OPT` | Append a raw ctest argument (repeatable) |
| `--local` | Run natively in the current environment; omit it for host-invoked container execution |

A note on `xvfb`: it satisfies the CLI's default wrapper, but Holoviz also
needs a working Vulkan/display stack. The ordinary FMCW smoke presets keep
visualization disabled; the separate visualization-enabled SISO check has
completed under Xvfb in the pinned environment.

---

## Formatting and lint

C++ and CUDA formatting is defined by `.clang-format` and pinned to clang-format 18.1.8
through pre-commit. The policy uses a 100-column limit. Run the authoritative formatter and
then the complete CI-equivalent lint suite with:

```bash
pre-commit run clang-format --all-files
pre-commit run --all-files
```

The formatting hook rewrites tracked `.cpp`, `.hpp`, `.cu`, and `.cuh` files. Commit any
resulting changes before pushing; CI runs the complete command and fails when formatting would
change a file. Use the pinned hook instead of a system-installed formatter because releases can
produce different layouts.

---

## `pyproject.toml`

`pyproject.toml` configures CLI tooling and CTest as well as
[scikit-build-core](https://scikit-build-core.readthedocs.io/) wheel packaging.
Key fields to update before publishing:

| Field | Purpose |
| --- | --- |
| `[project].name` | PyPI package name — should match `metadata.json:module.binary_packages.pypi` |
| `[project].version` | Sync with `metadata.json:module.version` |
| `[project].description` | Short description shown on PyPI |
| `[project].authors` | Your name / organisation |
| `[tool.scikit-build].cmake.args` | Extra CMake flags passed during `pip install` |

The current HSDK 5 image cannot build the Python extensions, so a
Python-enabled wheel is not currently available. When the SDK exports the
required binding target and the Python build is enabled, build one with:

```bash
pip install build
python -m build --wheel
```

---

## Naming conventions

| Context | Convention | Example |
| --- | --- | --- |
| Python import / C++ namespace | `snake_case` | `holoscan.radar` |
| Repository folder | `holoscan-<slug>` (kebab) | `holoscan-radar` |
| Debian package | `holoscan-<slug>` (kebab) | `holoscan-radar` |
| PyPI package | `holoscan-<slug>` (kebab) | `holoscan-radar` |
| CMake option prefix | `UPPER_SNAKE` | `HOLOSCAN_RADAR_BUILD_TESTING` |

---

## Further reading

- [HoloHub documentation](https://github.com/nvidia-holoscan/holohub)
- [Holoscan SDK documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/getting-started)
- [Holoscan Module ecosystem](https://nvidia-holoscan.github.io/)
- [Holoscan CLI documentation](https://github.com/nvidia-holoscan/holoscan-cli)
