# `BeamscanEstimatorOp` — SDK workarounds

**Uncommitted scratch.** Working notes on implementation shaped by the Holoscan SDK rather than by the radar algorithm. Not part of the operator's documented contract; see [`README.md`](README.md) for settled behavior.

## Optional Tensor input `is_valid` omitted from `contract()`

- **Where** — `beamscan_estimator_op.cu` `contract()`: `OnEach{snapshot}` plus the five `[live]` override entries; `is_valid` is registered in `setup()` and read with `receive_optional_value(is_valid)` in `compute()`. Graph tests `AcceptsValidIsValidMaskOn3DSnapshot` / `RejectsInvalidIsValidMaskOn3DSnapshot` are skipped.
- **Natural form** — Optional Tensor input: unwired graphs compile; wired graphs align `snapshot` + `is_valid` on one tick.
- **Constraint** — No optional input exists: a contract entry cannot tolerate an unwired port ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Omitting it is required for unwired graphs (`TEMPORAL_INPUT_UNCONNECTED`); wiring it while omitted yields `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Simplification trigger** — `SelectionReason::kNotConnected` routed through `on_missing SafeDefault{}`, or `InputSpecBuilder::optional()`.
- **Blast radius** — MVDR / MUSIC / Root-MUSIC / ESPRIT `is_valid`; SubarraySelection `is_valid`; FeatureAssembly optional gathers.

## Always-emitted `angle_spectrum`

- **Where** — `Output<Tensor> angle_spectrum`; `compute()` always loans and emits it (including the empty-item path).
- **Natural form** — Contract-optional: a consumer that wants only `aoa_estimates` leaves the spectrum unwired.
- **Constraint** — Declared device outputs must be bound (`IMPLICIT_DEVICE_ZERO_REJECTED` / `RUNTIME_PORT_BINDING_UNAVAILABLE`). Apps drain to `NullSinkOp`.
- **Simplification trigger** — Optional device outputs that may stay unbound.
- **Blast radius** — SnapshotGather `is_valid`; MVDR / MUSIC / Root-MUSIC / ESPRIT optional estimator outs.

## Dual-loan allocate-then-fill

- **Where** — `beamscan_estimator_op.cu` lines 261–324: allocate both loans, one `run_beamscan_stage` into both pointers, then commit/emit. Empty path prepares both before either emit.
- **Natural form** — A multi-output fill helper, or two independent `emit_device_tensor` calls if the stage were split.
- **Constraint** — One kernel writes both buffers; publishing the first loan before the second is ready would leave a sibling downstream on a late failure. Same as SnapshotGather.
- **Simplification trigger** — A `radar_common/sdk` helper that prepares N loans, fills, then emits.
- **Blast radius** — SnapshotGather; any dual-write stage.
