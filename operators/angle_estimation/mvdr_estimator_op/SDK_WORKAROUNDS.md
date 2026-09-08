# `MVDREstimatorOp` — SDK workarounds

**Uncommitted scratch.** Working notes on implementation shaped by the Holoscan SDK rather than by the radar algorithm. Not part of the operator's documented contract; see [`README.md`](README.md) for settled behavior.

## Optional Tensor input `is_valid` omitted from `contract()`

- **Where** — `mvdr_estimator_op.cu` `contract()`: `OnEach{snapshot}` plus one `Exact{}` entry per `[live]` override. `is_valid` is registered in `setup()` and read with `receive_optional_value` — never named in the contract.
- **Natural form** — Temporal policy would include the optional mask when the application connects it, and ignore it otherwise.
- **Constraint** — No optional input exists: a contract entry cannot tolerate an unwired port ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Omitted keeps unwired graphs compiling (`TEMPORAL_INPUT_UNCONNECTED` otherwise); wiring it while omitted kills the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Simplification trigger** — `SelectionReason::kNotConnected` through `on_missing`, or `InputSpecBuilder::optional()`.
- **Blast radius** — MUSIC / Root-MUSIC / ESPRIT / Beamscan `is_valid`; SubarraySelection `is_valid`.

## Optional outputs always allocated and emitted

- **Where** — Always loan/emit `angle_spectrum`, `doa_multi`, `source_count`, `eigenvalues` (and always run `matx::eig` / peak extract).
- **Natural form** — Emit a side output only when connected; skip `eig` when `eigenvalues` is unwired.
- **Constraint** — Unbound device outputs → `IMPLICIT_DEVICE_ZERO_REJECTED` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3).
- **Simplification trigger** — Optional Tensor outputs that may stay unbound, or an “is output connected” signal.
- **Blast radius** — MUSIC (same five); Root-MUSIC / ESPRIT (four); CFAR.

## Ctor-captured `MaxTensorExtents` for pool spans

- **Where** — Ctor sizes five output spans from `max_items`, `max_az`, `max_el`, `max_sources`, `max_channels`.
- **Natural form** — Infer bounds from the first tick’s `snapshot` and Params grid.
- **Constraint** — Graph cannot widen `tensor_allocation` after `setup()` ([`MAX_TENSOR_EXTENTS.md`](../../../docs/planning/MAX_TENSOR_EXTENTS.md)).
- **Simplification trigger** — Growable pool bounds from the first published shape.
- **Blast radius** — Every fat-output operator.

## Per-tick MatX eig/inv plan clear vs `stop()` MemTracker drain

- **Where** — `mvdr_stage.hpp` `clear_matx_solver_plan_caches()` (end of `run_mvdr_stage`) vs `retire_matx_solver_caches()` from `stop()`. MVDR is the only one of this quartet that also caches `matx::inv` plans.
- **Natural form** — MatX would key plans on shape/dtype, not this tick’s scratch pointers.
- **Constraint** — Inv/eig plans keyed on device pointers that Holoscan frees with per-tick scratch. A full `GetAllocMap()` drain every tick steals sibling cuFFT workspaces. HSDK 5 `OperatorBase::stop()` is the safe full drain.
- **Simplification trigger** — MatX plan keys that outlive the input pointer, or an SDK-owned MatX cache scope per operator.
- **Blast radius** — MUSIC / Root-MUSIC / ESPRIT (eig-only split of the same pattern).
