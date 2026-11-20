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
│   ├── sar/                        # Six standalone C++ SAR apps
│   │   ├── spotlight/
│   │   ├── video_sar/
│   │   ├── tiled_strip/
│   │   ├── continuous_strip/
│   │   ├── independent_images/
│   │   └── autofocus/
│   └── fmcw/
│       ├── siso/
│       ├── simo/
│       ├── dense_beamforming/
│       ├── mimo_tdm/
│       ├── adaptive_aoa/
│       ├── multi_waveform/
│       └── mimo_ddm/
├── applications/sar/tools/               # Optional CPHD download/subset CLIs
├── python/holoscan_radar/
│   ├── _radar.cpp                  # HSDK 5 FMCW operator provider
│   └── applications/              # Shared Python Graph helpers
└── tests/
    ├── cpp/                        # GTest suite (+ utils/: fixtures and Graph helpers)
    └── provider/                   # Operator provider and Graph pytest suite
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
run, and test work in containers. See the [SDK setup section](#sdk-development-image)
for the configured GA image, then use the commands below. The CLI
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

### Application build scope

Application `build` and `run` modes default to `BUILD_ALL=OFF`,
`HOLOSCAN_RADAR_BUILD_TESTING=OFF`, and `PKG_holoscan_radar=OFF`. Their
`HOLOSCAN_RADAR_APPLICATION=<name>` selection makes the default build depend
only on the C++ executable, or on `holoscan_radar_python` for Python. The shared
Python provider still needs every public operator. A configure diagnostic prints
the selected default build target.

Existing `build/<app>` caches need no reset: cached `APP_*` and `OP_*` switches
remain available, but unrelated targets are excluded from the application default
build. This also excludes unused common CUDA helpers. Explicit `build --target`
requests can still build any enabled target. C++/Python switches select the right
default target when reusing the same application directory.

CLI `--configure-args` follow the mode defaults. An explicit `BUILD_ALL=ON`,
`HOLOSCAN_RADAR_BUILD_TESTING=ON`, or
`PKG_holoscan_radar=ON` retains the broad default build. Module-wide tests enable
their required operator libraries even with `BUILD_ALL=OFF`. To disable only the
application default selection, pass `--configure-args=-DHOLOSCAN_RADAR_APPLICATION=`.

Full module builds (`./holoscan-radar build holoscan-radar --language cpp`),
package/wheel builds, nested module consumers, and `./holoscan-radar test` keep
their existing scope. Use a full module tree for installation and packaging;
application trees intentionally omit unneeded installable libraries from their
default build.

An aggregate module build lives in `build/holoscan-radar`; ordinary runs continue
to use `build/<app>`. They do not automatically reuse aggregate binaries. To use
an aggregate binary, invoke it from that tree inside the container, supplying the
configuration path explicitly. Python likewise needs that tree's `python/lib` on
`PYTHONPATH`. No build-directory aliases or automatic cache migrations are used.

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
ctest --test-dir build --output-on-failure
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
for mode-specific graphs, [data preparation](applications/sar/tools/README.md) for
optional public collections, and the [SAR operator index](operators/sar/README.md).
The fixture is generated in C++; C++ applications do not require the Python
provider. The five formation GUI apps have been visually checked with their
current presets. That check does not establish full-resolution image quality
or validate every public collection and backprojection backend.

The HSDK 5 GA image supplies Python Core and Visualization. Configure the
FMCW/SAR provider with nanobind and run its Graph suite inside the container:

```bash
cmake -S . -B build-python -DHOLOSCAN_RADAR_BUILD_PYTHON=ON
cmake --build build-python -j"$(nproc)"
ctest --test-dir build-python -R pytest_provider --output-on-failure
```

The thirteen Python Graph sources are colocated at
`applications/{fmcw,sar}/<app>/python/<app>.py`; CMake stages them under
`holoscan_radar.applications` for source builds and wheels. The provider
registers all 33 public operators, including SAR precision pairs and the
dtype-independent dump sink, under `holoscan.radar`. Python-enabled test builds
also build the six C++ SAR references and generate their CPHD fixtures. CTest
sets `HOLOSCAN_RADAR_TEST_BUILD_DIR` for provider tests; set it to the absolute
build directory when invoking pytest directly.
Python `[live]` enum ports remain unavailable until the SDK can register those
port payloads.

---

## Blossom CI for pull requests

Blossom accepts `/build` comments only from its maintainer allowlist. Its
credential-bearing vulnerability scan and dependent CI trigger run only when
the authorized PR source repository matches this repository. Fork PRs are
skipped; a maintainer must review the changes and copy the reviewed commit to
a branch here before requesting a Blossom build. Ordinary PR lint workflows
keep their own triggers. The scanner token permits content reads and
commit-status writes. The pinned action runs a bundled scanner binary; this
restriction avoids giving its build-enabled scanner outside-repository code.

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
- Runs every registered C++ tier, plus the Python `pytest_provider` binding and Graph suite by default. The C++
  tier includes the Monte Carlo `statistical` tests; the provider has no Python
  statistical suite. Pass `--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`
  for a C++-only run. Narrow to one tier with
  `--ctest-options='-DINCLUDE_LABELS=statistical'`, or exclude tiers with
  `-DEXCLUDE_LABELS='statistical|visualization'`.
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
  each selected key produces an installed `libholoscan_radar_<key>.*`; the artifact check relies
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

## Consuming the installed C++ package

Native installs default to `/usr` with the multiarch library directory
(`lib/<triplet>`), where CMake finds them without extra hints; see the
[installed layout](pkg/holoscan-radar/README.md#installed-layout). For a custom
prefix, point `CMAKE_PREFIX_PATH` at the Radar installation and the matching
Holoscan SDK:

```cmake
find_package(holoscan-radar CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE holoscan::fmcw_range_processor_op)
```

All built public targets share `holoscan_radar_targets.cmake`; individual target
names are unchanged. Discovery also works with Debian packaging disabled and
with intentional operator subsets. The package has no feature components.
Consumers that manually included the old per-target export files must migrate
to `find_package`; those files are no longer installed. Use a fresh installation
prefix when upgrading to avoid retaining stale exports.

Libraries are shared by default and installed as `libholoscan_radar_<target>.so`;
configure with `-DBUILD_SHARED_LIBS=OFF` to build and install static archives
instead. CUDA and the SDK are required. Static installations also resolve MatX,
and those containing CPHD also resolve tinyxml2 and Threads. Shared libraries
keep these dependencies private.
Debian packages containing CPHD declare a tinyxml2 dependency.
For TGZ archives, install it yourself; see the
[archive prerequisites](pkg/holoscan-radar/README.md#consuming-the-installed-c-package).

## SDK development image

Holoscan Radar builds against the public HSDK 5 GA image
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`. This replaces the
pre-release staging image; access to `nvstaging` is no longer required.

The image was verified on an NVIDIA GB10 host as Linux `arm64` with digest
`sha256:faa23a3a2479697fd64babcb27d28cf97f21d5da54d65c92225f8a9b0d99264a`.
Its CMake package is at `/usr/lib/cmake/holoscan`, including the visualization
and testing components.

Use the SDK `main-5x` source branch when an API is not yet documented, but
build against the GA image.

## Prepare the image

From the repository root on the host, install the CLI as described in
[`DEVELOPER.md`](DEVELOPER.md#holoscan-cli). You can pull the SDK image directly:

```bash
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
```

The CLI pulls the GA base image when needed. You can build the project
image separately with `./holoscan-radar build-container`; the run, build, and
test commands also build it when needed. The devcontainer uses the same base
image. Rebuild and reopen it when switching SDK images.

Run the repository's `./holoscan-radar` wrapper **on the host**. It selects
the GA base image and uses the CLI installed in `.venv` when available.
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

The GA image installs HSDK 5 Python Core and Visualization under
`/usr/lib/python3.12/site-packages`. Radar's FMCW/SAR provider uses nanobind and
does not require the SDK's former pybind11 CMake target. The development image
installs nanobind 3.0.1. `./holoscan-radar test` enables the provider and its
`pytest_provider` CTest entry by default. For a C++-only test run, pass
`--cmake-options=-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Direct CMake builds still
default to Python off; enable them with `-DHOLOSCAN_RADAR_BUILD_PYTHON=ON`.

```bash
./holoscan-radar test
./holoscan-radar run siso --language python --run-args=siso_smoke.yaml
```

The provider covers all 33 public operators: 26 FMCW operators with F32/F64
variants; `SARRangeCompressionOp`, `PhaseCorrectionOp`, and `AutofocusOp` with
F32/F64 variants; and `CPHDReaderOp`, `ApertureManagerOp`, and `BackprojectionOp`
with `F32GeomF32`, `F32GeomF64`, and `F64GeomF64` variants. `TensorDumpOp` is a
single dtype-independent sink without an `extents` argument. Existing FMCW
imports and call signatures are unchanged.

All six SAR applications also have Python graphs with the same YAML,
visualization, dumping, and playback behavior as C++. They default to single
samples with double geometry; `sample_precision: double` selects double for
both. Python owns its signal handlers and polls interrupts while native
execution releases the GIL; SIGINT/SIGTERM stop and join the session before
propagating `KeyboardInterrupt`. The C++ drivers own SIGINT/SIGTERM handlers and poll a lock-free stop flag
from their control loops, then request cooperative session shutdown. They use
caller-managed `run_async(plan)`, so profiler-created startup threads do not
conflict with HSDK managed-signal admission. Window closure and Escape request
graph shutdown directly. In-flight GPU work must finish before cleanup returns.

For an operator trace, run the built executable inside the development container:

```bash
nsys profile --trace=cuda,nvtx --sample=none --backtrace=none --cpuctxsw=none \
  -o /tmp/siso-profile build/holoscan-radar/applications/fmcw/siso/cpp/siso \
  build/holoscan-radar/applications/fmcw/siso/cpp/siso_smoke.yaml
```

Use a fresh output name and the actual generated preset path for your build.
No profiling-specific application flag or signal-mask changes are required.

Python `[live]` enum inputs remain unavailable because the SDK cannot register
those port payloads; the corresponding C++ contracts are unchanged.

## When the SDK image changes

Update the image reference in the root `Dockerfile`,
`.devcontainer/devcontainer.json`, `holoscan-radar` wrapper, and this guide together. Confirm the new
image's architecture, `holoscan_DIR`, visualization and testing components, and
Python Core and Visualization packages. Then rebuild the development image and
run the build, CTest, and application smoke checks described in `DEVELOPER.md`.

## Implementing and reviewing operators

Keep the Holoscan shell thin and math in a reusable stage. Infer named axes each tick, validate dtype/layout/extents before touching memory, and honor the compute stream. Hold input/descriptor ownership until asynchronous use completes. Separate cold plan/window/manifold/calibration work from steady compute. Reuse bounded scratch; do not allocate dense buffers every tick or do host walks over device data. Gate optional work, cache expensive configurations within the instance byte budget, and preserve warn-on-rebuild behavior.

Before landing, synchronize public headers, parameter schemas/defaults, wire schemas and conversion, owner README tables/behavior, build/install/export metadata, and Python bindings. Cover the supported precision combinations, live misses/revisits, invalid inputs, optional outputs, output lifetimes, and independent analytic predictions described in [testing requirements](docs/TESTING.md). Review temporal publication order before changing `contract()` or any sibling output ordering. No benchmark gate is required.

Support needed by distributed operators belongs in library packaging; checkout diagnostics stay beside their owners and outside packages. CFAR calibration remains available through its existing C++ and Python APIs. Use the existing module CMake helpers and package metadata; do not introduce a parallel packaging framework.

## Pinned lint workflow

On the host, after `uv sync`, install the lint dependencies into `.venv` and activate it so metadata validation imports the pinned CLI:

```bash
uv pip install --python .venv/bin/python -r utilities/requirements.lint.txt
source .venv/bin/activate
pre-commit run --files <changed-files>
pre-commit run --all-files
```

Use the pinned pre-commit hooks, not system formatter versions. Run applicable hooks before committing and the complete suite before handoff/push. Re-stage formatter changes. New code/CMake/YAML/Docker files need the two-line NVIDIA SPDX header. clang-format 18.1.8 owns C++ layout (100 columns); include each directly named standard-library type. Markdown lines including table rows are limited to 800 characters, fences need languages, and table delimiter rows use padded pipes.
