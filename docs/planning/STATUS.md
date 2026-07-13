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
| Signal-processing operators (`common/` + range, clutter, Doppler, fused) with MatX/CUDA wiring, simulator + test harness; `radar_op` placeholder removed and app/tests/package rewired | deustice | `deustice/sigproc-ops` | in review |

## Up next (unclaimed)

Roughly in dependency order — claim by adding a row above:

1. **Operator implementations, remaining groups** — detection & clustering, angle estimation, and MIMO marshaling kernels against the locked contracts (`operators/common/` supporting types `ArrayGeometry`, `AngleConvention`, `DDMCodeDescriptor`, `TXSchedule` land with their first consumer). The signal-processing group, the simulator + test harness, and the MatX/CUDA build wiring landed on `deustice/sigproc-ops`.
2. **Python bindings** — pybind11 bindings for the implemented operators (restores the Python example app). The YAML converters for the shared descriptor types landed on `deustice/sigproc-ops` (`radar_common/sdk/yaml_converters.hpp`; `from_config()` works for C++ apps).
