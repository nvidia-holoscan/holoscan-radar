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
| `radar_op` template placeholder | Keep until the first real operator lands with tests (bindings deferred per the row above), then remove and rewire the app/tests/package | 2026-07-06 |
| MatX + CUDA build wiring | Wire into the build from the start (not deferred to first kernel) | 2026-07-01 |

## In-flight work

| Work item | Owner | Branch | State |
| --- | --- | --- | --- |
| Fill out `AGENTS.md` + this status file | deustice | `deustice/planning` | in review |
| Signal-processing operators (`common/` + range, clutter, Doppler, fused) with MatX/CUDA wiring, simulator + test harness | deustice | `deustice/sigproc-ops` | in progress |

## Up next (unclaimed)

Roughly in dependency order — claim by adding a row above:

1. **C++ API skeletons** — supporting data structures + operator headers with `setup()` port/parameter declarations per `OPERATOR_API.md`, stub `compute()`, compilable; MatX/CUDA wired into CMake.
2. **Simulator + test harness** — the analytic point-target simulator and test scaffolding per `TESTING.md`.
3. **Operator implementations** — kernels against the locked contracts, per functional group.
