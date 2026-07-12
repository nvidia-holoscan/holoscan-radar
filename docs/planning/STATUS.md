# Project Status & Work Coordination

The coordination point for in-flight work (see [`AGENTS.md`](../../AGENTS.md) → Team workflow). Before starting non-trivial work, check the claims table and add a row; update it in the PR that lands the work.

## Where the project stands

- **2026-07-01** — Planning docs (`OPERATOR_API.md`, `WORKFLOWS.md`, `TESTING.md`) written; FMCW API lockdown specified. No operator code yet beyond the `radar_op` template placeholder.

## Settled decisions

Decisions made outside the planning docs, recorded so they aren't re-litigated:

| Decision | Choice | Date |
| --- | --- | --- |
| Operator directory layout | Per functional group under `operators/`: `fmcw/` (signal processing + MIMO marshaling), `detection/`, `clustering/`, `angle_estimation/`, plus a shared `common/` library for the six supporting types | 2026-07-06 |
| Python bindings for new operators | Deferred to a later phase — operators land C++-first with tests; bindings for the full operator set follow as a dedicated effort | 2026-07-06 |
| `radar_op` template placeholder | Removed on `deustice/sigproc-ops` once the first real operators landed with tests; app, tests, and package rewired to the FMCW operators | 2026-07-10 |
| MatX + CUDA build wiring | Wire into the build from the start (not deferred to first kernel) | 2026-07-01 |
| GPU as a hard dependency | A CUDA-capable NVIDIA GPU is required for operators, apps, and the C++ test suite; no CPU fallbacks, GPU-optional builds, or skip-if-no-GPU test guards (the `RADAR_TEST_SKIP_IF_NO_GPU` machinery was removed). Host-only code exists only where host-only by design (simulator, window generation) | 2026-07-10 |
| Uniform CUDA-accessible input requirement | All operators reject pageable-host (`kDLCPU`) input with a clear error; the range and fused operators' H2D staging was removed (originally kept because the range processor is potentially first in the chain — ingest operators upload instead). Amends SIGPROC_PLAN decision 7 and the OPERATOR_API memory-space convention | 2026-07-10 |

## In-flight work

| Work item | Owner | Branch | State |
| --- | --- | --- | --- |
| Fill out `AGENTS.md` + this status file | deustice | `deustice/planning` | in review |
| Signal-processing operators (`common/` + range, clutter, Doppler, fused) with MatX/CUDA wiring, simulator + test harness | deustice | `deustice/sigproc-ops` | in progress |
| Detection operator (`operators/detection/` + `CFARDetectorOp`) | dpc | `dpc/detection` | **All four CFAR variants done** (see below) |
| Detection bridge `DetectionReductionOp` | dpc | `dpc/detection-reduction` | **checklist-complete** — first/peak/medoid + built-in scorers + `ScoreFn`/`custom` + live + tests + perf |
| Detection bridge `FeatureAssemblyOp` | dpc | `dpc/feature-assembly` | **checklist-complete** (unit tests green; staged follow-ups in OPEN_ITEMS) |

### Landed on `dpc/detection`

- **`CFARDetectorOp` CA slice** — complex/real power paths, border-correct CA
  stencil, `[live]` overrides with sticky semantics, `nci_axes` non-coherent
  integration (Gamma(`K`) threshold via host incomplete-beta LUT), batch-widened
  `detection_idx`, unit + Monte Carlo statistical tests. Team open items in
  [`CFARDetectorOp open items`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md).
- **`CFARDetectorOp` GO slice** — 2-D GOGO greatest-of stencil (range GO ×
  Doppler GO), host Monte Carlo `alpha` LUT (CA-seeded bracket), full GO
  contract test matrix + `GoMonteCarloPfaK1` / `GoMonteCarloPfaNciK8`
  statistical tests.
- **`CFARDetectorOp` SO slice** — 2-D SOSO smallest-of stencil (shared GOGO
  partition), host Monte Carlo `alpha` LUT, full SO contract test matrix +
  `SoMonteCarloPfaK1` / `SoMonteCarloPfaNciK8` statistical tests.
- **`CFARDetectorOp` OS slice** — per-cell order-statistic stencil with
  `[live] os_rank` (0-based ascending), host Monte Carlo `alpha` LUT keyed by
  rank, OS-specific validation/live-override tests, full contract test matrix
  + `OsMonteCarloPfaK1` / `OsMonteCarloPfaNciK8` statistical tests.

## Up next (unclaimed)

Roughly in dependency order — claim by adding a row above:

1. **C++ API skeletons** — supporting data structures + operator headers with `setup()` port/parameter declarations per `OPERATOR_API.md`, stub `compute()`, compilable; MatX/CUDA wired into CMake.
2. **Simulator + test harness** — the analytic point-target simulator and test scaffolding per `TESTING.md`.
3. **Operator implementations** — kernels against the locked contracts, per functional group.
4. **Operator implementations, remaining groups** — detection & clustering, angle estimation, and MIMO marshaling kernels against the locked contracts (`operators/common/` supporting types `ArrayGeometry`, `AngleConvention`, `DDMCodeDescriptor`, `TXSchedule` land with their first consumer). The signal-processing group, the simulator + test harness, and the MatX/CUDA build wiring landed on `deustice/sigproc-ops`.
5. **Python bindings** — pybind11 bindings for the implemented operators (restores the Python example app). The YAML converters for the shared descriptor types landed on `deustice/sigproc-ops` (`radar_common/sdk/yaml_converters.hpp`; `from_config()` works for C++ apps).

## Open items for team review

Contract, SDK, and signal-processing questions that need a team decision are
tracked with the operator that surfaced them (for example,
[`CFARDetectorOp open items`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md)).
New operators should follow [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md)
and [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md).
