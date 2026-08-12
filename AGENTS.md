# Agent Instructions — holoscan-radar

GPU radar signal-processing operator library for the NVIDIA Holoscan SDK: Holoscan operators implemented in C++/CUDA on [MatX](https://github.com/NVIDIA/MatX), composed into pipelines purely by port connections — no glue code between operators.

**Current phase: FMCW implementation, plus SAR contract design.** The operator contracts (ports, shapes, dtypes, parameters) are fully specified in `docs/planning/`; implementation lands against them. The signal-processing group (`operators/fmcw/`: range, clutter, Doppler, fused range-Doppler) and the shared `operators/common/` library are implemented; detection, clustering, angle estimation, and MIMO marshaling are next.

**SAR** is a second modality whose contract is specified in [`docs/planning/SAR/`](docs/planning/SAR/) and is **in review, not yet locked**. No SAR operator code exists. See the Scope guardrail below for what that permits.

## Source of truth

Read before starting non-trivial work, in this precedence order:

1. [`docs/planning/OPERATOR_API.md`](docs/planning/OPERATOR_API.md) — **the locked contract**: every operator's inputs/outputs/parameters, shapes, dtype pairs, and defaults, plus the shared supporting data structures. If we discover a conflict, bug, or unexpected design corner, consult with the developer before changing.
2. [`docs/planning/WORKFLOWS.md`](docs/planning/WORKFLOWS.md) — scope: the SISO / SIMO / MIMO-TDM / MIMO-DDM operator chains, with port-level wiring diagrams.
3. [`docs/planning/TESTING.md`](docs/planning/TESTING.md) — what tests must assert; the analytic point-target simulator requirements.
4. [`docs/planning/CREATE_A_MODULE.md`](docs/planning/CREATE_A_MODULE.md) — how the Holoscan Module scaffolding (build, packaging, bindings) works.
5. [`docs/planning/NEW_OPERATOR_CHECKLIST.md`](docs/planning/NEW_OPERATOR_CHECKLIST.md) — landing gate; verify before claiming an operator (or variant slice) done.
6. [`docs/planning/OPERATOR_DEVELOPMENT.md`](docs/planning/OPERATOR_DEVELOPMENT.md) — how to implement well: hot-path discipline, diagnosis, `[live]`/calibration pitfalls, tests, and performance characterization (lessons from landed operators).
7. [`docs/planning/TEAM_BLOCKED.md`](docs/planning/TEAM_BLOCKED.md) — index of items blocked on a team decision (detail in each operator’s `OPEN_ITEMS.md`).
8. [`docs/planning/SAR/`](docs/planning/SAR/) — the SAR modality's own `OPERATOR_API.md` / `WORKFLOWS.md` / `TESTING.md` / `SIGPROC_PLAN.md` / `TEAM_BLOCKED.md`, mirroring 1–3 and 7 one level down. **Accepted 2026-08-07** and now governing for SAR work, on the same footing as 1–7 rather than subordinate to them.

Rules that follow from "lockdown":

- **Docs first.** Code must match `OPERATOR_API.md` exactly — port names, shapes, dtype pairs, parameter names/defaults, `[live]` tags. To change an interface, change the doc (in the same PR at minimum), call it out as a contract change in the PR description, and get it reviewed as such. Never deviate silently in code.
- **Never change the SAR contract without the developer's approval in the chat
  session — ask first, edit after.** SAR is in active implementation against an
  accepted contract, so *proposing* revisions is expected rather than exceptional.
  Landing one unasked is not. This applies to every contract change, SAR-scoped
  ones included: **stop, present it, and wait for an explicit answer.** Do not
  edit the contract and report it afterwards, and do not write code that assumes
  a change you have not had approved — if implementation reveals the need for one,
  that is the moment to stop and ask, not to proceed and reconcile later.
  - **What to present**, in the chat, before touching anything: what the change
    is, exactly which ports / parameters / shapes / conventions move, why the
    contract as written does not work, what it breaks or invalidates, and the
    alternative you considered and rejected. Offer it as a decision to accept or
    decline, not as a notification.
  - **After approval:** tag it at the point of change, and append it to the log in
    [`docs/planning/SAR/OPERATOR_API.md`](docs/planning/SAR/OPERATOR_API.md)
    → *Contract changes since approval*.
  - **Beyond SAR is the same gate plus two obligations** — the invariants below,
    the shared descriptor set, `operators/common/`, the `holoscan.radar` public
    Python API, FMCW code, the metadata key namespace, or anything a second
    modality would inherit. Say plainly that it reaches beyond SAR when asking,
    mark it in the document so reviewers see it, and add a row in
    [`docs/planning/TEAM_BLOCKED.md`](docs/planning/TEAM_BLOCKED.md) so FMCW owners
    see it without reading SAR docs. Prefer the SAR-scoped option where both exist.
  - **Not a contract change**, so no approval needed: typos, broken links, clearer
    prose, or a doc fix that makes the text match what the contract already says.
    The rule covers changes to the *interface or behaviour* — port names, shapes,
    dtypes, parameters, defaults, `[live]` tags, algorithms, conventions.
- **Do not silently resolve documented open items.** Team-blocked questions are
  indexed per modality — [`docs/planning/TEAM_BLOCKED.md`](docs/planning/TEAM_BLOCKED.md)
  for FMCW, [`docs/planning/SAR/TEAM_BLOCKED.md`](docs/planning/SAR/TEAM_BLOCKED.md)
  for SAR — with detail in each operator’s `OPEN_ITEMS.md`. **FMCW has none open**
  (L1–L15 and S1–S4 are all settled; see `STATUS.md` → Settled decisions). **SAR
  has one: R10**, the metadata key namespace, which survived contract acceptance
  because it is a decision about FMCW taken in a SAR document. It may not be
  resolved in code. Take the current set from that index rather than from any list
  here, since rows close as decisions land, and surface a new open item for a team
  decision instead of picking an answer. Note the distinction from the bullet
  above: revising the accepted SAR contract is allowed and merely has to be
  declared, whereas an open `R` item is a question nobody has answered yet.
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

The invariants above govern all current code, and are **amended for SAR as of 2026-08-07** by the accepted contract in [`docs/planning/SAR/OPERATOR_API.md`](docs/planning/SAR/OPERATOR_API.md) → *Contract changes against the locked invariants*. Four invariants are amended and two conventions added. Landed FMCW code is unaffected: every change is additive and the contract renames nothing (D24).

Invariant **2**, twice over: six descriptors become **nine** (`CoordinateFrame`, `SampleGeometry`, `LFMWaveformDescriptor`), and sanctioned descriptor-carrying *output* ports go from one to four. **3**: an orthogonal internal-arithmetic axis on one operator. **4**: output shape from declared geometry rather than an input tensor. **5**: a `[live]` geometry override on an accumulating operator forces a reset, and a wired override port beats its parameter. The two conventions are cross-tick accumulation and a one-key metadata plane.

Amending an invariant **further** is a beyond-SAR contract change: see the *Docs first* rules above, and get explicit agreement before landing it.

**Scope guardrail:** the *implementation* wave is FMCW SISO/SIMO/MIMO-TDM/MIMO-DDM only. Do not add Pulse-Doppler or PVA operators; CDM/BPM is deferred to a future wave; FDM and PMCW are excluded outright.

**SAR is in implementation.** The contract in [`docs/planning/SAR/`](docs/planning/SAR/) was accepted on 2026-08-07, so **SAR operator code is in scope** — the earlier gate holding it back until R1 closed is lifted. Build against [`docs/planning/SAR/SIGPROC_PLAN.md`](docs/planning/SAR/SIGPROC_PLAN.md)'s phases. Its own scope guardrail applies within it: stripmap and spotlight, monostatic, single-pol, time-domain backprojection — no frequency-domain formation, interferometry, polarimetry, ScanSAR, or SAR detection. First ship is modes 1–5 plus opt-in autofocus; modes 6–9 are deferred packaging.

## Repo map

| Path | What it is |
| --- | --- |
| `docs/planning/` | The specs above — the work queue and contract source of truth |
| `docs/planning/OPERATOR_DEVELOPMENT.md` | Agent/human guidance for implementing operators (hot path, diagnosis, perf) |
| `docs/planning/SAR/` | SAR modality contract (accepted 2026-08-07): API, workflows, testing, build plan; audience derivatives `PRACTITIONER_OVERVIEW.md` / `DEVELOPER_OVERVIEW.md` |
| `operators/` | Operator libraries, one dir per group, gated by `add_holohub_operator()` / `OP_<name>` CMake options |
| `operators/common/` | Shared `radar_common` library: descriptors, window generation, point-target simulator, config cache, tensor/MatX utilities |
| `operators/fmcw/` | FMCW signal-processing operators (+ future MIMO marshaling) |
| `operators/sar/` | SAR image-formation operators (backprojection, autofocus) — scaffolded, filling in demo order |
| `operators/io/` | Standard-format readers/writers (CPHD, SICD) + `TensorDumpOp`; separate group so the XML/NITF deps sit behind their own `OP_<name>` gates |
| `tools/sar_sim/` | In-repo public SAR simulator: recipes → CPHD + ground-truth sidecar. Python-only; `sarpy`/`sarkit` are confined here and nothing in the build links against them |
| `tools/sar_image/` | Reference cupy backprojector + offline PNG renderer. Shares no code with `sar_sim` on purpose — it is the independent implementation the sidecar's predicted pixels are checked against |
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
# Not -j"$(nproc)": a MatX translation unit costs a few GB in cicc, so the job
# count is bounded by RAM, not cores. Over-subscribing gets compilers OOM-killed
# ("cicc died due to signal 9") and can time out gtest discovery. See DEVELOPER.md.
cmake --build build -j8

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
