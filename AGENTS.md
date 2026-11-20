# Agent instructions — holoscan-radar

GPU radar operators use C++/CUDA, MatX, and HSDK 5 Graph/Tensor APIs. A CUDA-capable NVIDIA GPU is required for operators, applications and C++ tests. Do not add CPU fallbacks, no-GPU skips, or GPU-optional modes. Use SDK `main-5x` source for undocumented HSDK 5 APIs; never infer them from HSDK 4.

## Authority and scope

Public headers and parameter/wire schemas own declarations, types and defaults. Operator READMEs own behavior and compact port/parameter tables. Read [architecture](docs/ARCHITECTURE.md), [workflows](docs/WORKFLOWS.md), [testing requirements](docs/TESTING.md), and the relevant owner README before nontrivial changes. Application READMEs own runnable graphs and presets. [DEVELOPER.md](DEVELOPER.md) owns setup, implementation, build, lint and packaging guidance.

Preserve plain-tensor data ports, the bounded shared descriptor set, compile-time precision, named axes, constructor-captured tensor-pool bounds, per-tick Exact/SafeDefault live overrides, causal publication order, and explicit angle conventions. Public interface/behavior changes require synchronized declarations, schemas, README contracts and tests; identify them explicitly for review. Surface unresolved product questions instead of silently selecting behavior. Deferred work remains on [Jira board 41309](https://jirasw.nvidia.com/secure/RapidBoard.jspa?rapidView=41309); owner READMEs preserve relevant limitations. No status-file claim workflow is required.

Scope is FMCW SISO/SIMO/MIMO-TDM/MIMO-DDM and implemented SAR formation. Do not add Pulse-Doppler/PVA, FDM or PMCW; CDM/BPM and later SAR capabilities remain deferred. Fern integration is deferred.

## Implementation and verification

- Keep application entry points as graphs; app-private config/helpers stay colocated, reusable SDK-aware helpers in `applications/common/radar_app`, pure models in `operators/common/radar_common`. Keep executable README Mermaid graphs current. Operator YAML uses native `from_config()`; app parsing is limited to app-owned config except commented read-only sizing mirrors.
- Use the host workflow: `uv sync`, then `./holoscan-radar build`, `run`, or `test`. The wrapper runs containers. Specify `--language cpp` when selecting C++ for a dual-language app. Direct CMake debugging belongs inside the configured development container.
- Tests use independent analytic simulators; update simulator self-checks for signal-model changes. Run appropriate tests, then the full GPU suite for broad changes. Do not weaken assertions to accommodate missing prerequisites.
- Use pinned `pre-commit run --files <changed-files>` hooks before commits and `pre-commit run --all-files` before handoff/push. Every new code/CMake/YAML/Docker file needs NVIDIA SPDX headers. clang-format owns C++ layout at 100 columns; include direct dependencies. Markdown line limit is 800, including table rows.
- Runtime support belongs in library packaging; owner-local checkout tools stay outside packages. Preserve CFAR calibration APIs and installed package contents.

## Conventions

C++ namespace `holoscan::radar`, Python public package `holoscan.radar`, classes `CamelCaseOp`, directories `snake_case`. Include headers from their include root; same-dir app-private headers are the exception and use `NOLINT(build/include_subdir)`. Commits use imperative `Area: Summary`. Preserve the user's branch and unrelated working-tree changes. Never publish or merge without authorization.
