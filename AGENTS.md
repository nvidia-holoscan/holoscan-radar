# Agent Instructions — holoscan-radar

GPU radar signal-processing operator library for the NVIDIA Holoscan SDK: Holoscan operators implemented in C++/CUDA on [MatX](https://github.com/NVIDIA/MatX), composed into pipelines purely by port connections — no glue code between operators.

**Current phase: FMCW operator API lockdown.** The operator contracts (ports, shapes, dtypes, parameters) are fully specified in `docs/planning/`; implementation lands against them. The only operator code in the tree today is the `radar_op` template placeholder.

## Source of truth

Read before starting non-trivial work, in this precedence order:

1. [`docs/planning/OPERATOR_API.md`](docs/planning/OPERATOR_API.md) — **the locked contract**: every operator's inputs/outputs/parameters, shapes, dtype pairs, and defaults, plus the shared supporting data structures. If we discover a conflict, bug, or unexpected design corner, consult with the developer before changing.
2. [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md) — scope: the SISO / SIMO / MIMO-TDM / MIMO-DDM operator chains, with port-level wiring diagrams.
3. [`docs/planning/TESTING.md`](docs/planning/TESTING.md) — what tests must assert; the analytic point-target simulator requirements.
4. [`docs/planning/CREATE_A_MODULE.md`](docs/planning/CREATE_A_MODULE.md) — how the Holoscan Module scaffolding (build, packaging, bindings) works.

Rules that follow from "lockdown":

- **Docs first.** Code must match `OPERATOR_API.md` exactly — port names, shapes, dtype pairs, parameter names/defaults, `[live]` tags. To change an interface, change the doc (in the same PR at minimum), call it out as a contract change in the PR description, and get it reviewed as such. Never deviate silently in code.
- **Do not silently resolve documented open items.** The known ones: the covariance estimators' multi-snapshot / spatial-smoothing input form; where the source count comes from (fixed vs. MDL/AIC vs. CFAR-cluster); ULA/subarray selection for Root-MUSIC/ESPRIT on non-uniform virtual arrays. Surface these for a team decision instead of picking an answer in code.

## Design invariants (do not violate)

1. **Plain-tensor data ports.** Operators exchange plain tensors on the data plane — never bespoke record/message types (e.g. no "detection" struct: CFAR emits `detection_idx`, `detection_score`, `noise_estimate` as separate tensors). Where producers and consumers need bridging, that is an explicit small add-on operator (`SnapshotGatherOp`, `FeatureAssemblyOp`, `DetectionReductionOp`), not a shared message type.
2. **Shared descriptors are bounded.** Exactly six shared config-plane types exist: `ArrayGeometry`, `AngleConvention`, `WindowSpec`, `FMCWWaveformDescriptor`, `DDMCodeDescriptor`, `TXSchedule`. They are parameters, and ride input ports only as `[live]` overrides; the single sanctioned descriptor *output* port is `VirtualArrayAssemblyOp.array_geometry` (runtime-computed virtual geometry). Do not add new shared types or new descriptor-carrying ports without a contract change to `OPERATOR_API.md`.
3. **Precision by compile-time template instantiation** (single + double). Never a runtime dtype flag, never inferred from data; an instance's inputs and outputs match its templated precision.
4. **Rank-agnostic batching.** Operate on the meaningful trailing axes; treat leading `...` axes as opaque batch dims; infer shapes from the tensors each tick rather than taking them as parameters. Downstream never assumes fixed shapes — that is what makes shape-changing `[live]` overrides legal.
5. **`[live]` override convention.** Parameters are build-time config by default; a `[live]` parameter may additionally be overridden in-flight via a same-named non-blocking input port. Costly overrides (FFT plans, window coefficients, CFAR kernels) warn once per new configuration and cache prior ones. Structural parameters (e.g. `cfar_variant`, `angle_convention`, `multiplexing_scheme`) stay static — instantiate separate operators to vary them. Philosophy: don't over-constrain a caller who knows what they're doing; allow reasonable footguns and warn where relevant.
6. **Convention-free kernels, explicit angle boundary.** Kernels only dot direction vectors with element positions in the fixed canonical array-local frame. `AngleConvention` is a **required parameter with no default** on every operator that consumes or emits angles; it is applied only as a boundary relabeling. `direction_cosine` (u, v) is the convention-free escape hatch.
7. **Industry-standard operator names** over internal schemes.

**Scope guardrail:** this wave is FMCW SISO/SIMO/MIMO-TDM/MIMO-DDM only. Do not add Pulse-Doppler, SAR, or PVA operators; CDM/BPM is deferred to a future wave; FDM and PMCW are excluded outright.

## Repo map

| Path | What it is |
| --- | --- |
| `docs/planning/` | The specs above — the work queue and contract source of truth |
| `operators/` | Operator libraries, one dir per group, gated by `add_holohub_operator()` / `OP_<name>` CMake options |
| `operators/radar_op/` | Template placeholder operator — kept as the scaffolding exemplar until the first real operator lands |
| `applications/radar_pipeline/` | Example pipeline app (C++ and Python) |
| `python/holoscan/radar/` | Python package root; re-exports operators from per-operator pybind11 submodules |
| `tests/cpp/`, `tests/python/` | GTest and pytest suites |
| `pkg/` | Debian/wheel packaging (`add_holohub_package`) |
| `cmake/` | HoloHub CMake helpers (`HoloHubConfigHelpers.cmake`, `pybind11_add_holohub_module.cmake`) |
| `metadata.json` | Module metadata (schema `urn:holohub:module:v2`); per-operator `metadata.json` under each operator dir |

## Build, test, lint

See [`DEVELOPER.md`](DEVELOPER.md) for detail. Quick reference:

```bash
# Configure + build (requires Holoscan SDK >= 4.3.0 discoverable via find_package;
# `pip install holoscan` gives a CPU-capable config, or use `./holohub run-container`)
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"

# C++ tests
ctest --test-dir build --output-on-failure -L unit

# Python tests (conftest.py wires in the build tree; set HOLOSCAN_RADAR_BUILD_DIR
# if your build dir is not ./build). Skips cleanly (exit 5) if holoscan isn't importable.
pytest tests/python/ -v

# Lint — CI runs exactly this on every PR; run it before pushing
pre-commit run --all-files
```

The lint stack (`.pre-commit-config.yaml`): ruff + isort + black (Python), cpplint (C++), cmakelint, markdownlint, codespell, `metadata.json` schema validation, and NVIDIA SPDX copyright headers on all code/CMake/YAML/Docker files (`check_copyright.py`; markdown is exempt). Every new source file needs the two-line SPDX header — copy it from any existing file of the same type.

## Conventions

- C++ namespace `holoscan::radar`; Python package `holoscan.radar`; operator classes CamelCase ending in `Op`; directories snake_case.
- Commits: imperative subject, `Area: Summary` prefix style (e.g. `CI: Add linting workflow`, `Operators: ...`).
- Branches: `<user>/<topic>` (e.g. `deustice/planning`). PRs target `main`; lint CI must pass; contract changes (see Docs first) get called out explicitly.

## Team workflow

There is no external issue tracker: **`docs/planning/` is the work queue** and [`docs/planning/STATUS.md`](docs/planning/STATUS.md) is the coordination point. Before starting non-trivial work: read the relevant planning doc sections, then check `STATUS.md` for what is claimed/in flight and record what you pick up. When work lands (or a team decision is made), update `STATUS.md` in the same PR.
