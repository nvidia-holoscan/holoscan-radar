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
│   │       ├── tensor/             #   tensor/DLPack/MatX plumbing
│   │       └── sdk/                #   Holoscan-integration glue
│   └── fmcw/                       # FMCW signal-processing group
│       └── <operator_name>/
│           ├── <operator_name>.hpp / .cu   # Operator + stage kernel
│           ├── <operator_name>_stage.hpp   # (stage header name varies)
│           └── metadata.json       # Operator-level metadata
├── applications/
│   └── fmcw_siso/
│       └── cpp/
│           ├── fmcw_siso.cpp        # Example SISO FMCW pipeline
│           └── metadata.json        # Application-level metadata
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
| `./holohub build fmcw_siso` | Configure and build the example pipeline inside the container |
| `./holohub run fmcw_siso` | Configure, build, and run the example pipeline inside the container |
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
ctest --test-dir build --output-on-failure -L unit
```

Run Python tests:

```bash
HOLOSCAN_RADAR_BUILD_DIR=build \
# Prepend to PYTHONPATH so that the build directory Python module is importable
PYTHONPATH=build/python${PYTHONPATH:+:$PYTHONPATH} \
pytest tests/python/ -v
```

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
