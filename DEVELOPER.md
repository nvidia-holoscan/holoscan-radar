# Developer Guide — Holoscan Radar

This guide covers the layout, build system, and day-to-day workflow for developing and
distributing this Holoscan Module.

---

## Module layout

```text
holoscan-radar/
├── holohub                         # CLI wrapper (delegates to holoscan-cli)
├── Dockerfile                      # Development container image
├── CMakeLists.txt                  # Root CMake — orchestrates operators/applications/tests
├── pyproject.toml                  # Python packaging metadata (scikit-build-core)
├── metadata.json                   # Module-level metadata (schema: urn:holohub:module:v2)
├── operators/
│   ├── common/                     # Shared radar_common library
│   │   └── radar_common/           # Organized by dependency layer:
│   │       ├── descriptors/        #   shared config-plane types (STL only)
│   │       ├── dsp/                #   host-side signal math
│   │       ├── sim/                #   analytic point-target simulator (STL only)
│   │       ├── tensor/             #   tensor/DLPack/MatX plumbing
│   │       └── sdk/                #   Holoscan-integration glue
│   └── fmcw/                       # FMCW signal-processing group
│       └── <operator_name>/
│           ├── <operator_name>.hpp / .cu   # Operator + stage kernel
│           ├── <operator_name>_stage.hpp   # (stage header name varies)
│           └── metadata.json       # Operator-level metadata
├── applications/
│   ├── common/radar_app/            # Shared Graph/config/report/dashboard support
│   └── fmcw/
│       ├── siso/
│       ├── simo/
│       ├── dense_beamforming/
│       ├── mimo_tdm/
│       ├── multi_waveform/
│       └── mimo_ddm/
├── python/holoscan/radar/
│   └── __init__.py                 # Placeholder (bindings deferred)
└── tests/
    ├── cpp/                        # GTest suite (+ utils/: harness, fixtures, bin model)
    └── python/                     # pytest suite
```

---

## `holohub` wrapper commands

The `holohub` script at the module root wraps the `holoscan-cli` package with module-specific
defaults. On first run it installs `holoscan-cli` via pip if it isn't already importable
(internet required); subsequent runs reuse the installed copy.

| Command | What it does |
| --- | --- |
| `./holohub run-container` | Build and start the development container |
| `./holohub build simo` | Configure and build the example pipeline inside the container |
| `./holohub run simo` | Configure, build, and run the example pipeline inside the container |
| `./holohub test` | Configure, build, and run the test suite inside the container |
| `./holohub install --dev` | Install a `.pth` hook so `import holoscan.radar` works in any shell |
| `./holohub package holoscan-radar` | Build a native Debian package with Holoscan Radar operators |
| `./holohub package holoscan-radar --pkg-generator=WHEEL` | Build a native Python 3.12 package with Holoscan Radar operators |

Set `HOLOSCAN_CLI_INSTALL_ARGS` to override the pip install arguments. The
wrapper forwards this value into Docker builds as the
`HOLOSCAN_CLI_INSTALL_ARGS` build arg, so the development image installs the
same CLI package spec as the host wrapper. Set `HOLOSCAN_CLI_SOURCE` to a local
checkout for host-side CLI development.

---

## Building without the wrapper

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Run C++ tests:

```bash
ctest --test-dir build --output-on-failure -LE performance
```

Run Python tests:

```bash
# Holoscan SDK must import first; conftest.py then prepends
# build/python/lib/holoscan onto holoscan.__path__. Do not put
# build/python/lib alone on PYTHONPATH — that shadows the SDK package.
HOLOSCAN_RADAR_BUILD_DIR=build \
pytest tests/python/ -v
```

---

## Running the full suite with `./holohub test`

`./holohub test` configures, builds, and tests inside the development container in one
step — the containerized equivalent of the manual sequence above.

```bash
./holohub test                    # build the image if needed, then configure/build/test
./holohub test --no-docker-build  # reuse the existing image (faster)
./holohub test --dryrun --verbose # print the commands without running them
```

What it does, and where to change it:

- Runs [`ci/container.ctest`](ci/container.ctest), this module's CTest driver. The wrapper
  points `HOLOSCAN_CLI_CTEST_SCRIPT` at it; the driver bundled with `holoscan-cli` resolves
  its source directory inside the pip install tree and cannot see this repo.
- Builds into **`build-test/`**, a dedicated tree. The driver wipes it on every run, which
  is why it is deliberately not your `build/` directory.
- Runs every tier except `performance`, which has its own invocation. That includes the
  Monte Carlo `statistical` tests — they cost ~13 s against a ~207 s suite, so there is
  nothing to gain by deferring them. Narrow to one tier with
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
| `--local` | Run on the host instead of in the container |

A note on `xvfb`: it satisfies the CLI's default wrapper but does **not** make Holoviz
render headlessly — that needs Vulkan support. Application smoke runs keep visualization
disabled.

---

## `pyproject.toml`

`pyproject.toml` configures [scikit-build-core](https://scikit-build-core.readthedocs.io/) for
wheel packaging. Key fields to update before publishing:

| Field | Purpose |
| --- | --- |
| `[project].name` | PyPI package name — should match `metadata.json:module.binary_packages.pypi` |
| `[project].version` | Sync with `metadata.json:module.version` |
| `[project].description` | Short description shown on PyPI |
| `[project].authors` | Your name / organisation |
| `[tool.scikit-build].cmake.args` | Extra CMake flags passed during `pip install` |

Build a wheel:

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
