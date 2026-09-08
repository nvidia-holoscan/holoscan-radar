# `RootMUSICEstimatorOp` — SDK workarounds

**Uncommitted scratch.** Working notes on implementation shaped by the Holoscan SDK rather than by the radar algorithm. Not part of the operator's documented contract; see [`README.md`](README.md) for settled behavior.

## Optional Tensor input `is_valid` omitted from `contract()`

- **Where** — `root_music_estimator_op.cu` `contract()`: `OnEach{snapshot}` plus one `Exact{}` entry per `[live]` override. `is_valid` is registered in `setup()` and read with `receive_optional_value` — never named in the contract.
- **Natural form** — Temporal policy would include the optional mask when the application connects it, and ignore it otherwise.
- **Constraint** — No optional input exists: a contract entry cannot tolerate an unwired port ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Omitted keeps unwired graphs compiling (`TEMPORAL_INPUT_UNCONNECTED` otherwise); wiring it while omitted kills the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Simplification trigger** — `SelectionReason::kNotConnected` through `on_missing`, or `InputSpecBuilder::optional()`.
- **Blast radius** — MUSIC / MVDR / ESPRIT / Beamscan `is_valid`; SubarraySelection `is_valid`.

## Optional outputs always allocated and emitted

- **Where** — Always loan/emit `doa_multi`, `source_count`, `eigenvalues` (no `angle_spectrum`).
- **Natural form** — Emit a side output only when connected.
- **Constraint** — Unbound device outputs → `IMPLICIT_DEVICE_ZERO_REJECTED` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3).
- **Simplification trigger** — Optional Tensor outputs that may stay unbound.
- **Blast radius** — MUSIC/MVDR (five outs); ESPRIT (same four); CFAR.

## Ctor-captured `MaxTensorExtents` for pool spans

- **Where** — Ctor sizes four output spans from `max_items` / `max_sources` / `max_channels` (`max_az`/`max_el` unused).
- **Natural form** — Infer bounds from the first tick.
- **Constraint** — Graph cannot widen `tensor_allocation` after `setup()` ([`MAX_TENSOR_EXTENTS.md`](../../../docs/planning/MAX_TENSOR_EXTENTS.md)).
- **Simplification trigger** — Growable pool bounds from the first published shape.
- **Blast radius** — Every fat-output operator.

## Per-tick MatX eig-plan clear vs `stop()` drain; cuSOLVER on the operator

- **Where** — `retire_music_matx_caches()` from `stop()`; `RootMUSICSolverState` holds the `cusolverDn` handle and workspaces for the life of the operator. `solve_companion_roots` synchronizes the stream to read `geev` info.
- **Natural form** — A batched device root solver (OPEN_ITEMS §6) would drop the serial host-side `geev` loop and the info D2H sync. MatX plans would key on shape, not scratch pointers.
- **Constraint** — Same MatX pointer-keyed eig cache as MUSIC. HSDK 5 `OperatorBase::stop()` is the safe full `MemTracker` drain. Serial `cusolverDnXgeev` is an algorithm/perf choice, not an SDK gap — listed here only because teardown is tied to the operator lifecycle hook.
- **Simplification trigger** — MatX plan keys that outlive scratch pointers; a batched root kernel would delete the cuSOLVER state.
- **Blast radius** — MUSIC/MVDR MatX split; ESPRIT does **not** use cuSOLVER (in-block eigen).
