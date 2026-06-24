# Developer Guide — radar

This guide covers the layout, build system, and day-to-day workflow for developing and
distributing this Holoscan Module.

---

## Module layout

```
holoscan-radar/
├── holohub                         # CLI wrapper (delegates to holoscan-cli)
├── Dockerfile                      # Development container image
├── CMakeLists.txt                  # Root CMake — orchestrates operators/applications/tests
├── pyproject.toml                  # Python packaging metadata (scikit-build-core)
├── metadata.json                   # Module-level metadata (schema: urn:holohub:module:v2)
├── operators/
│   └── radar_op/
│       ├── radar_op.cpp / .hpp  # Operator implementation
│       └── metadata.json           # Operator-level metadata
├── applications/
│   └── radar_pipeline/
│       └── radar_pipeline.cpp   # Example pipeline
├── python/holoscan/radar/
│   └── __init__.py                 # Re-exports operators for `from holoscan.radar import ...`
└── tests/
    ├── cpp/                        # GTest suite (C++ modules only)
    └── python/                     # pytest suite
```

---

## `holohub` wrapper commands

The `holohub` script at the module root wraps the `holoscan-cli` package with module-specific
defaults. On first run it installs `holoscan-cli` via pip if it isn't already importable
(internet required); subsequent runs reuse the installed copy.

| Command | What it does |
|---|---|
| `./holohub run-container` | Build and start the development container |
| `./holohub build radar_pipeline` | CMake configure + build inside the container |
| `./holohub run radar_pipeline` | Run the example pipeline |
| `./holohub test` | Run CTest (C++ unit tests) and pytest |
| `./holohub install --dev` | Install a `.pth` hook so `import holoscan.radar` works in any shell |

Set `HOLOSCAN_CLI_INSTALL_ARGS` to override the pip install arguments. The
wrapper forwards this value into Docker builds as the
`HOLOSCAN_CLI_INSTALL_ARGS` build arg, so the development image installs the
same CLI package spec as the host wrapper. Set `HOLOSCAN_CLI_SOURCE` to a local
checkout for host-side CLI development.

---

## Building without the wrapper

```bash
cmake -S . -B build -DBUILD_ALL=ON -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```


Run C++ tests:

```bash
ctest --test-dir build --output-on-failure -L unit
```


Run Python tests:

```bash
HOLOSCAN_RADAR_BUILD_DIR=build \
PYTHONPATH=build/python${PYTHONPATH:+:$PYTHONPATH} \
pytest tests/python/ -v
```

`PYTHONPATH` is **prepended via `${PYTHONPATH:+:$PYTHONPATH}`** so that an existing entry on the variable is kept while an unset/empty variable doesn't yield a trailing colon. Two failure modes the shorter forms invite:

- **`PYTHONPATH=build/python`** (replace): drops any ambient holoscan SDK install on `PYTHONPATH`. The module-level `importorskip("holoscan")` then fires, pytest exits with code 5, and CTest marks the run as Skipped.
- **`PYTHONPATH=build/python:$PYTHONPATH`** (naive prepend): on a fresh shell or CI runner where `$PYTHONPATH` is unset, this expands to `PYTHONPATH=build/python:` — Python treats the trailing empty entry as the current directory, silently shadowing installed packages with whatever happens to live in the test CWD.

---

## `pyproject.toml`

`pyproject.toml` configures [scikit-build-core](https://scikit-build-core.readthedocs.io/) for
wheel packaging. Key fields to update before publishing:

| Field | Purpose |
|---|---|
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
|---|---|---|
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
