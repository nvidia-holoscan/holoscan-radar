# Agent Instructions — holoscan-radar

GPU radar signal-processing operator library for the NVIDIA Holoscan SDK: Holoscan operators implemented in C++/CUDA on [MatX](https://github.com/NVIDIA/MatX), composed into pipelines purely by port connections — no glue code between operators.

**Current phase: FMCW implementation.** The operator contracts (ports, shapes, dtypes, parameters) are fully specified in `docs/planning/`; implementation lands against them. The signal-processing group (`operators/fmcw/`: range, clutter, Doppler, fused range-Doppler) and the shared `operators/common/` library are implemented; detection, clustering, angle estimation, and MIMO marshaling are next.

## Source of truth

Read before starting non-trivial work, in this precedence order:

1. [`docs/planning/OPERATOR_API.md`](docs/planning/OPERATOR_API.md) — **the locked contract**: every operator's inputs/outputs/parameters, shapes, dtype pairs, and defaults, plus the shared supporting data structures. If we discover a conflict, bug, or unexpected design corner, consult with the developer before changing.
2. [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md) — scope: the SISO / SIMO / MIMO-TDM / MIMO-DDM operator chains, with port-level wiring diagrams.
3. [`docs/planning/TESTING.md`](docs/planning/TESTING.md) — what tests must assert; the analytic point-target simulator requirements.
4. [`docs/planning/CREATE_A_MODULE.md`](docs/planning/CREATE_A_MODULE.md) — how the Holoscan Module scaffolding (build, packaging, bindings) works.
5. [`docs/planning/NEW_OPERATOR_CHECKLIST.md`](docs/planning/NEW_OPERATOR_CHECKLIST.md) — landing gate; verify before claiming an operator (or variant slice) done.
6. [`docs/planning/OPERATOR_DEVELOPMENT.md`](docs/planning/OPERATOR_DEVELOPMENT.md) — how to implement well: hot-path discipline, diagnosis, `[live]`/calibration pitfalls, tests, and performance characterization (lessons from landed operators).
7. [`docs/planning/TEAM_BLOCKED.md`](docs/planning/TEAM_BLOCKED.md) — index of items blocked on a team decision (detail in each operator’s `OPEN_ITEMS.md`).

Rules that follow from "lockdown":

- **Docs first.** Code must match `OPERATOR_API.md` exactly — port names, shapes, dtype pairs, parameter names/defaults, `[live]` tags. To change an interface, change the doc (in the same PR at minimum), call it out as a contract change in the PR description, and get it reviewed as such. Never deviate silently in code.
- **Do not silently resolve documented open items.** Team-blocked questions are
  indexed in [`docs/planning/TEAM_BLOCKED.md`](docs/planning/TEAM_BLOCKED.md)
  (detail in each operator’s `OPEN_ITEMS.md`). Known highlights: the covariance
  estimators' multi-snapshot / spatial-smoothing input form; where the source
  count comes from (fixed vs. MDL/AIC vs. CFAR-cluster); ULA/subarray selection
  for Root-MUSIC/ESPRIT on non-uniform virtual arrays. Surface these for a team
  decision instead of picking an answer in code.
- **Checklist + development guidance.** Follow `NEW_OPERATOR_CHECKLIST.md` as the done gate; use `OPERATOR_DEVELOPMENT.md` when designing `compute()`, live overrides, tests, or chasing performance.

## Design invariants (do not violate)

1. **Plain-tensor data ports.** Operators exchange plain tensors on the data plane — never bespoke record/message types (e.g. no "detection" struct: CFAR emits `detection_idx`, `detection_score`, `noise_estimate` as separate tensors). Where producers and consumers need bridging, that is an explicit small add-on operator (`SnapshotGatherOp`, `FeatureAssemblyOp`, `DetectionReductionOp`), not a shared message type.
2. **Shared descriptors are bounded.** Exactly six shared config-plane types exist: `ArrayGeometry`, `AngleConvention`, `WindowSpec`, `FMCWWaveformDescriptor`, `DDMCodeDescriptor`, `TXSchedule`. They are parameters, and ride input ports only as `[live]` overrides; the single sanctioned descriptor *output* port is `VirtualArrayAssemblyOp.array_geometry` (runtime-computed virtual geometry). Do not add new shared types or new descriptor-carrying ports without a contract change to `OPERATOR_API.md`.
3. **Precision by compile-time template instantiation** (single + double). Never a runtime dtype flag, never inferred from data; an instance's inputs and outputs match its templated precision.
4. **Rank-agnostic batching.** Operate on the meaningful trailing axes; treat leading `...` axes as opaque batch dims; infer shapes from the tensors each tick rather than taking them as parameters. Downstream never assumes fixed shapes — that is what makes shape-changing `[live]` overrides legal.
5. **`[live]` override convention.** Parameters are build-time config by default; a `[live]` parameter may additionally be overridden in-flight via a same-named non-blocking input port. Costly overrides (FFT plans, window coefficients, CFAR kernels) warn once per new configuration and cache prior ones. Structural parameters (e.g. `cfar_variant`, `angle_convention`, `multiplexing_scheme`) stay static — instantiate separate operators to vary them. Philosophy: don't over-constrain a caller who knows what they're doing; allow reasonable footguns and warn where relevant.
6. **Convention-free kernels, explicit angle boundary.** Kernels only dot direction vectors with element positions in the fixed canonical array-local frame. `AngleConvention` is a **required parameter with no default** on every operator that consumes or emits angles; it is applied only as a boundary relabeling. `direction_cosine` (u, v) is the convention-free escape hatch.
7. **Industry-standard operator names** over internal schemes.
8. **A CUDA-capable NVIDIA GPU is a hard dependency** — for the operators, the apps, and the C++ test suite (decided 2026-07-10). Do not add CPU fallback compute paths, skip-if-no-GPU test guards, or GPU-optional build modes; the only host-only code is deliberately host-only by design (the analytic simulator, window generation).

**Scope guardrail:** this wave is FMCW SISO/SIMO/MIMO-TDM/MIMO-DDM only. Do not add Pulse-Doppler, SAR, or PVA operators; CDM/BPM is deferred to a future wave; FDM and PMCW are excluded outright.

## Repo map

| Path | What it is |
| --- | --- |
| `docs/planning/` | The specs above — the work queue and contract source of truth |
| `docs/planning/OPERATOR_DEVELOPMENT.md` | Agent/human guidance for implementing operators (hot path, diagnosis, perf) |
| `operators/` | Operator libraries, one dir per group, gated by `add_holohub_operator()` / `OP_<name>` CMake options |
| `operators/common/` | Shared `radar_common` library: descriptors, window generation, point-target simulator, config cache, tensor/MatX utilities |
| `operators/fmcw/` | FMCW signal-processing operators (+ future MIMO marshaling) |
| `applications/fmcw_siso/` | Example SISO FMCW pipeline (C++ + Python Holoviz) |
| `applications/fmcw_simo/` | Sparse SIMO angle-then-cluster demo (C++ + Python) |
| `applications/fmcw_mimo/` | MIMO TDM/DDM demo (C++ + Python) |
| `applications/fmcw_dense_simo/` | Dense SIMO beamformer demo (C++ + Python) |
| `applications/common/radar_app/` | Shared report + Holoviz viz operators (C++ header + Python bindings) |
| `performance/` | Measurement harness (C++ + Python) — cold vs steady, not pass/fail tests |
| `python/holoscan/radar/` | Python package root; re-exports operators from per-operator pybind11 submodules |
| `tests/cpp/`, `tests/python/` | GTest and pytest suites (Python: set `HOLOSCAN_RADAR_BUILD_DIR`) |
| `pkg/` | Debian/wheel packaging (`add_holohub_package`) |
| `cmake/` | HoloHub CMake helpers (`HoloHubConfigHelpers.cmake`, `pybind11_add_holohub_module.cmake`) |
| `metadata.json` | Module metadata (schema `urn:holohub:module:v2`); per-operator `metadata.json` under each operator dir |

## Building example applications

The `applications/fmcw_*` examples follow a strict layering so each app's `.cpp`
reads as the graph, not scaffolding. When adding or changing an app, keep code
in the right layer:

- **App `.cpp` = the graph.** `compose()` (operators + `add_flow`) plus a
  one-line `main` that calls `radar_app`'s `run_fmcw_app<App>()`. Nothing else
  belongs here.
- **App-owned code is colocated** in the app's `cpp/` dir: `config.hpp`
  (`AppConfig : app::CommonRunConfig` + `load_config`) and `custom_ops.hpp`
  (operators specific to one app — e.g. the MIMO ADC sources, the dense-path
  bridges). Same-dir headers, included bare with `// NOLINT(build/include_subdir)`.
- **Reusable, Holoscan-aware app helpers → `applications/common/radar_app/`**
  (see its README): the report/viz operators and the `app/` scaffolding
  (`CommonRunConfig` + `parse_*`, `run_fmcw_app`, the rank-agnostic
  `SimAdcSourceOp`, the `HolovizLayout` builder). Header-only (`inline` /
  in-class); it is not a wrapper around `HolovizOp` — the app keeps its own
  `add_flow` wiring, `radar_app` supplies data + operators + a factory.
- **Pure model helpers → `operators/common/radar_common/`**: the analytic
  simulators and array builders (`sim/`), descriptors, DSP — no Holoscan / CUDA /
  Holoviz. `point_target_sim.hpp` stays minimal-dependency (STL only); the MIMO
  TDM/DDM simulators live beside it in `sim/mimo_sim.hpp`.

**The config boundary.** Operator parameters flow natively through Holoscan's
`from_config()`; apps never parse operator YAML blocks. Apps parse only the
app-owned sections (`waveform`, `run`, `cube`, `targets`, `plots`, the viz
toggles, `rx_array`/`tx_array`). Reading an operator block *only* to size a viz
axis or geometry grid is allowed, but must be commented as a read-only mirror.

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

The lint stack (`.pre-commit-config.yaml`): ruff + isort + black (Python), cpplint (C++), cmakelint, markdownlint, codespell, `metadata.json` schema validation, and NVIDIA SPDX copyright headers on all code/CMake/YAML/Docker files (`check_copyright.py`; markdown is exempt). Every new source file needs the two-line SPDX header — copy it from any existing file of the same type. cpplint enforces 80-column lines and include-what-you-use (add the direct `#include` for every type you name, e.g. `<string>` for `std::string`); run it before pushing, not after CI fails. **markdownlint caps line length at 800** (`MD013`, incl. table rows + headings; no table exemption and table cells can't wrap, so keep dense one-line `STATUS.md` rows / contract bullets under 800 by tightening prose).

**Verifying a change.** The analytic simulators (`radar_common/sim/`) are the ground-truth "tautology guard" — assert operator/pipeline output against them, and add sim self-checks (`tests/cpp/common/test_sim.cpp`) when you add or change a signal model. For a behavior-preserving refactor, diff an app's deterministic log output before/after — but exclude the per-cluster representative (`rep_*`) fields: they are non-deterministic run-to-run (a GPU medoid tie-break; see `operators/detection/detection_reduction_op/OPEN_ITEMS.md` item 6), so a naive diff shows false churn. Build and run in the Holoscan container (`./holohub run-container`).

## Conventions

- C++ namespace `holoscan::radar`; Python package `holoscan.radar`; operator classes CamelCase ending in `Op`; directories snake_case.
- Include headers by their path from the include root (`#include "group/op/op.hpp"`), never bare — the one exception is a same-dir colocated app-private header (`config.hpp` / `custom_ops.hpp`), tagged `// NOLINT(build/include_subdir)`.
- Commits: imperative subject, `Area: Summary` prefix style (e.g. `CI: Add linting workflow`, `Operators: ...`).
- Branches: `<user>/<topic>` (e.g. `deustice/planning`). PRs target `main`; lint CI must pass; contract changes (see Docs first) get called out explicitly.

## Team workflow

There is no external issue tracker: **`docs/planning/` is the work queue** and [`docs/planning/STATUS.md`](docs/planning/STATUS.md) is the coordination point. Before starting non-trivial work: read the relevant planning doc sections, then check `STATUS.md` for what is claimed/in flight and record what you pick up. When work lands (or a team decision is made), update `STATUS.md` in the same PR.
