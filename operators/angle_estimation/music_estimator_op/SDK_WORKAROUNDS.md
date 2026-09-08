# `MUSICEstimatorOp` — SDK workarounds

**Uncommitted scratch.** Working notes on implementation shaped by the Holoscan SDK rather than by the radar algorithm. Not part of the operator's documented contract; see [`README.md`](README.md) for settled behavior.

## Optional Tensor input `is_valid` omitted from `contract()`

- **Where** — `music_estimator_op.cu` `contract()`: `OnEach{snapshot}` plus one `Exact{}` entry per `[live]` override. `is_valid` is registered in `setup()` and read with `receive_optional_value` — never named in the contract.
- **Natural form** — `OnEach` / `OnAligned` would include the optional mask when a Graph source is connected, and ignore the port otherwise.
- **Constraint** — No optional input exists: a contract entry cannot tolerate an unwired port ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Omitted keeps unwired graphs compiling (`TEMPORAL_INPUT_UNCONNECTED` otherwise); wiring it while omitted kills the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Simplification trigger** — `SelectionReason::kNotConnected` routed through the existing `on_missing` policies, or `InputSpecBuilder::optional()`.
- **Blast radius** — Beamscan / MVDR / Root-MUSIC / ESPRIT `is_valid`; SubarraySelection `is_valid`.

## Optional outputs always allocated and emitted

- **Where** — `setup()` / `compute()` always `allocate_tensor` + emit `angle_spectrum`, `doa_multi`, `source_count`, `eigenvalues`. Empty-item path prepares all five loans before any publish.
- **Natural form** — Emit a side output only when that port is connected (contract “optional”).
- **Constraint** — HSDK 5 device producers that skip `bind_tensor_output_device` fail with `IMPLICIT_DEVICE_ZERO_REJECTED` / `RUNTIME_PORT_BINDING_UNAVAILABLE`. Optional Connectivity gating is not usable the way HSDK 4.x `kNone` was ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3; OPEN_ITEMS §4).
- **Simplification trigger** — Optional Tensor outputs that may stay unbound without rejecting the device producer, or a first-class “no-op sink” in the SDK.
- **Blast radius** — MVDR (same five outs); Root-MUSIC / ESPRIT (four outs, no spectrum); CFAR’s five outs.

## Ctor-captured `MaxTensorExtents` for pool spans

- **Where** — `MUSICEstimatorOp(const MaxTensorExtents&)` computes `*_max_byte_span_` from `max_items`, `max_az`, `max_el`, `max_sources`, `max_channels`; `setup()` passes those into `tensor_allocation`.
- **Natural form** — Infer output bounds from the first tick’s `snapshot` and Params grid, or let the pool grow.
- **Constraint** — Graph may only tighten `tensor_allocation` after `setup()`, never widen (`GRAPH_PARAMETER_PORT_SHAPE_CHANGED`). Fat MUSIC spectrum / `doa_multi` cubes cannot use a library-wide 256 MiB ceiling without OOM ([`MAX_TENSOR_EXTENTS.md`](../../../docs/planning/MAX_TENSOR_EXTENTS.md)).
- **Simplification trigger** — SDK pool bounds that can be set or grown from the first published shape.
- **Blast radius** — Every fat-output operator in the module (`graph.op<Op>(name, extents)`).

## Per-tick MatX eig-plan clear vs `stop()` MemTracker drain

- **Where** — `music_stage.hpp` `clear_music_matx_caches()` (end of `run_music_stage`) vs `retire_music_matx_caches()` from `stop()`.
- **Natural form** — MatX would key plans on shape/dtype, not on the device pointer of this tick’s scratch `cov`, so the cache could live across ticks without a manual clear.
- **Constraint** — MatX `eig` plans are keyed on device pointers that Holoscan frees when the per-tick scratch Tensor dies. Leaving plans in the cache UAFs. Draining `GetAllocMap()` every tick also frees sibling operators’ cuFFT workspaces. HSDK 5 still runs `OperatorBase::stop()` at `kStop`, which is the only safe full drain.
- **Simplification trigger** — MatX plan keys that outlive the input pointer, or an SDK-owned MatX cache scope per operator.
- **Blast radius** — `mvdr_stage.hpp` (identical split); Root-MUSIC adds cuSOLVER teardown on the same `stop()` hook.
