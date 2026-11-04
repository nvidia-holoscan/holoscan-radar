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
├── python/holoscan_radar/
│   ├── _radar.cpp                  # HSDK 5 FMCW operator provider
│   └── applications/              # Shared Python Graph helpers
├── performance/                   # C++ YAML recipes, adapters, and shared runner
└── tests/
    ├── cpp/                        # GTest suite (+ utils/: fixtures and Graph helpers)
    └── provider/                   # FMCW provider and Graph pytest suite
```

---

## Holoscan CLI

The development-only `cli` and `test` groups in [`pyproject.toml`](pyproject.toml)
pin the Holoscan CLI and Python CTest dependencies. On the **host**, with
Python 3.11+ available, install them from the repository root using uv:

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

The development Dockerfile installs the same groups. Run the following
commands from the **host** at the repository root. The wrapper prefers the
pinned CLI in `.venv` without requiring activation; the CLI performs the build,
run, and test work in containers. Authenticate to NGC as described in
[`SDK_DEVELOPMENT.md`](SDK_DEVELOPMENT.md), then use the commands below. The CLI
builds the project image when needed and tags it from the project and branch.
`./holoscan-radar build-container` prepares the image separately when you
want to inspect or reuse it before building an app; it is not a prerequisite.

| Command | What it does |
| --- | --- |
| `./holoscan-radar build holoscan-radar --language cpp` | Build the C++ module in a container |
| `./holoscan-radar build simo --language cpp` | Build the C++ SIMO app and its dependencies in a container |
| `./holoscan-radar run simo --language cpp` | Build and run the C++ SIMO app in a container |
| `./holoscan-radar test` | Build and run the C++ and Python test suites in a container |
| `./holoscan-radar test --cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF` | Build and run only the C++ test suite |
| `./holoscan-radar run siso --language python --run-args=siso_smoke.yaml` | Run the Python SISO Graph smoke preset |
| `./holoscan-radar package holoscan-radar` | Build the Debian package in a container |
| `./holoscan-radar run-container` | Open an interactive container shell |

Each host command starts a container; no interactive shell is required for
builds or tests. When an app has both C++ and Python implementations, the CLI
selects Python if `--language` is omitted; use `--language cpp` or
`--language python` to choose explicitly. The CLI selects the required app
dependencies. The wrapper selects the image SDK for nested builds when a sibling
`../holoscan-sdk`
checkout exists. The CLI still mounts that sibling checkout. Start with a clean
`build/<app>` tree when switching SDK images because CMake caches dependency
paths. `[tool.holoscan]` selects [`ci/container.ctest`](ci/container.ctest) for
`./holoscan-radar test`. To try an unpublished CLI checkout, install it into
the selected environment and run its `holoscan` command there.

---

### Application logging

FMCW C++ and Python applications prefix Holoscan app/operator logs with local
wall-clock date/time, milliseconds and the UTC offset. For example,
`[2026-10-26 12:34:56.789 +00:00] [info] [report_ops.hpp:461] ...`.
Set `HOLOSCAN_LOG_FORMAT` to use an SDK preset or custom spdlog format instead.
HoloViz's separate renderer logger retains its SDK format.

### GPU placement

FMCW applications offer a presenter choice when X11/XRender support is built,
`DISPLAY` is nonempty, visualization is enabled, and
`app.visualization.x11_presentation` is omitted. At startup, an interactive terminal
asks whether the setup needs X11 image transport (for example SSH/X11 forwarding)
or a native HoloViz window on a local desktop. Enter selects X11.
An explicit boolean bypasses both the question and any remembered answer:

```yaml
app:
  visualization:
    x11_presentation: false
```

Set `true` to select X11 without asking, or `false` to select native HoloViz.
Both C++ and Python then ask how long to remember the choice:

- **This run only** (default): no preference is saved.
- **Remember for 7 days**: expires exactly seven days after selection.
- **Remember until changed**: no expiration.

Preferences are keyed by `DISPLAY`, `WAYLAND_DISPLAY`, and local/SSH session type,
under `$XDG_CACHE_HOME/holoscan-radar/presentation-v2`, or
`$HOME/.cache/holoscan-radar/presentation-v2` when `XDG_CACHE_HOME` is unset.
Version-2 JSON records store the presenter, setup key, save time, and expiration
as UTC Unix seconds (`expires_at: null` means until changed). Each load checks
version, field types, presenter, setup identity, timestamps, and expiration.
Expired, malformed, wrong-setup, or future-dated records are ignored. Old
`presentation-v1` files are not reused. Display variables identify a setup;
they do not prove that SSH forwarding is in use.

The host CLI mounts the project as the container home, so its default cache is
`.cache/holoscan-radar/presentation-v2` in the host checkout and survives container
restarts. Direct runs use their own environment's cache location. Every reused
answer prints its expiration (or "until changed") and how to choose again.
Use `--choose-presenter` to clear the current setup's saved answer and ask again;
selecting "this run only" then leaves no saved answer. Explicit YAML still wins:
omit `x11_presentation` to use the interactive choice. For example, on the host:

```bash
./holoscan-radar run mimo_tdm --language cpp --run-args=--choose-presenter
./holoscan-radar run mimo_tdm --language python --run-args=--choose-presenter
```

Direct executables accept `[config.yaml] --choose-presenter`; Python's
`build_graph(path, choose_presenter=True)` exposes the same behavior.
The reset option requires an interactive terminal when a choice is needed.
Without a terminal, a valid saved answer is used; otherwise X11 remains the
default and no answer is cached. EOF at the presenter question defaults to X11;
EOF at the remember question keeps the selected presenter for this run only.
Nonvisual runs and configurations without X11 never ask. Cache write failures
are reported but do not prevent the selected presenter from running.

This setting controls window presentation; HoloViz still composes the image and
native text offscreen in either case. The native window path requires a
compatible Vulkan presentation surface. Startup logs
`Radar presentation: X11 finished-image transport` for X11,
or `Radar presentation: CUDA device <N>` for the native window path.

All seven FMCW workflows (C++ and Python) and all five SAR applications retain
the caller's current CUDA device for computation. CUDA ordinals are relative to
the devices visible inside the container.

On X11, FMCW renders the complete dashboard with the SDK's offscreen CUDA sink,
downloads it through a bounded host pool, and sends it to the X server with
`X11ImagePresenterOp`. XRender scales the finished image when the window resizes.
This supports forwarded SSH displays and unrestricted multi-GPU hosts without
requiring a Vulkan window swapchain on the display GPU. Startup logs the compute
and composition devices and `X11 finished-image transport`. All DSP, plot
composition and native SDK text rendering remain on the GPU; the host only packs
and transports finished pixels. No GPU visibility setting is required.

HSDK 5.0 has no public renderer-device override: the application isolates its
startup discovery for SAR and other window systems in
`radar_app/app/gpu_placement.cpp`, using the installed renderer API and the SDK's
adapter-selection order. A small temporary window
queries presentation support before the graph is compiled. When an image
producer and renderer select different adapters, the graph explicitly copies
the RGBA image through a bounded host pool to the receiving GPU. This covers
FMCW's offscreen composer and window, plus SAR's formation and timeline displays.
It works without CUDA peer access, at the cost of two host/device copies and
synchronization per device boundary. Same-device connections remain direct.

This fixes visualization placement; it does not add arbitrary computation-device
selection. Some operators allocate CUDA scratch in construction or `start()`,
where the installed SDK does not establish the output pool's device scope.
Moving their output bindings alone to a different device can therefore fail.

## Inside a development container

For manual CMake work, first open the devcontainer or run
`./holoscan-radar run-container` **on the host**. Then run the following
commands **inside that container**, from the repository root:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

Run the C++ tests or a built SISO executable inside the same container:

```bash
ctest --test-dir build --output-on-failure -LE performance
./build/applications/fmcw/siso/cpp/siso \
  build/applications/fmcw/siso/cpp/siso_smoke.yaml
```

Run all five SAR fixture and headless visualization paths inside the same
container without public data:

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

The HSDK 5 GA image supplies Python Core and Visualization. Configure the
FMCW provider with nanobind and run its Graph suite inside the container:

```bash
cmake -S . -B build-python -DHOLOSCAN_RADAR_BUILD_PYTHON=ON
cmake --build build-python -j"$(nproc)"
PYTHONPATH=/usr/lib/python3.12/site-packages:build-python/python/lib \
  python3.12 -m pytest tests/provider -q
```

The seven Python Graph sources are colocated at
`applications/fmcw/<app>/python/<app>.py`; CMake stages them under
`holoscan_radar.applications` for source builds and wheels. The provider
registers all 26 FMCW operators in both precisions under `holoscan.radar`.
Python `[live]` enum ports remain unavailable until the SDK can register those
port payloads.

### Performance characterization

Run these commands **inside** the development container after opening it as
described above. These are diagnostic measurements, not pass/fail latency
thresholds:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -L performance
./build/performance/cpp/perf_cfar_detector --full
```

The 26 C++ performance tools read colocated YAML recipes and report setup,
first-frame, steady-state, and live-update latencies. Results depend on the
GPU and host. See [performance/README.md](performance/README.md) for scopes,
recipes, and output formats.

---

## From the host: full test suite

From the host, `./holoscan-radar test` builds the project image if needed,
configures and builds the module, and runs the suite in a container:

```bash
./holoscan-radar test
./holoscan-radar test --dryrun --verbose
```

The wrapper selects the GA base image and handles the sibling SDK
checkout when one exists. The CLI finds the image SDK's CMake package without
an explicit `holoscan_DIR` override.

What it does, and where to change it:

- Runs [`ci/container.ctest`](ci/container.ctest), this module's CTest driver, selected
  by `pyproject.toml`. The driver bundled with `holoscan-cli` resolves its source directory
  inside the pip install tree and cannot see this repo.
- Builds into **`build-test/`**, a dedicated tree. The driver wipes it on every run, which
  is why it is deliberately not your `build/` directory.
- Runs every registered C++ tier except `performance`, which has its own invocation,
  plus the Python `pytest_provider` binding and Graph suite by default. The C++
  tier includes the Monte Carlo `statistical` tests; the provider has no Python
  statistical suite. Pass `--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`
  for a C++-only run. Narrow to one tier with
  `--ctest-options='-DINCLUDE_LABELS=statistical'`, or widen the exclusion with
  `-DEXCLUDE_LABELS='performance|statistical'`.
- Enables `HOLOSCAN_RADAR_TEST_PACKAGING` and runs `Packaging_debian` plus
  `Packaging_wheel` (the latter when Python is enabled). These serial integration
  checks build fresh artifacts through `holoscan package holoscan-radar`
  (`DEB,TGZ` and `WHEEL`, using `--local` inside the SDK container), verify selected
  operator libraries in both native formats, compile/run a C++ consumer against
  both extracted native packages, and install/import the wheel in a separate
  Python environment.
  Configure checks cover the curated package selection, subsets, and
  explicit/cached test preferences; missing expected libraries fail artifact
  verification. `HOLOSCAN_RADAR_PKG_OPERATORS` entries are operator subdirectory
  selection keys, not CMake target names. By the existing library naming convention,
  each selected key produces an installed `lib<key>.*`; the artifact check relies
  on this convention and allows additional libraries. If an operator changes its
  output name or produces multiple libraries, update the artifact check to verify
  the intended outputs. File presence alone does not prove an operator is usable;
  the separate C++ consumer checks representative headers, exports, and linking.
  No packages are published or installed system-wide.
  Artifacts remain under `build-test/tests/packaging/<mode>/run-*/artifacts/`
  until the next dashboard run. Build parallelism defaults to two; set
  `CMAKE_BUILD_PARALLEL_LEVEL` to adjust it. To run just these checks, pass
  `--ctest-options='-DINCLUDE_LABELS=packaging'`. To omit artifact builds, pass
  `--cmake-options=-DHOLOSCAN_RADAR_TEST_PACKAGING=OFF`. Ordinary direct CMake
  builds default this additional suite to off. CI-driver and package-metadata
  fixtures run alongside the artifact tests as `Packaging_ci_helpers`; the lint
  workflow keeps its existing checks.
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

## From the host: formatting and lint

Run lint from the repository root **on the host**. Activate `.venv` for these
commands because the metadata hook imports the Holoscan CLI from the active
Python environment:

```bash
source .venv/bin/activate
python -m pip install -r utilities/requirements.lint.txt
```

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

The FMCW provider uses the SDK's Python Core and Visualization packages and
nanobind 3.0.1. Build its CPython 3.12 stable ABI wheel inside the GA
development container with:

```bash
python3.12 -m pip wheel . --no-deps -w dist
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
