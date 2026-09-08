# `PhaseShiftBeamformerOp` — SDK workarounds

**Uncommitted scratch.** Working notes on implementation shaped by the Holoscan SDK rather than by the radar algorithm. Not part of the operator's documented contract; see [`README.md`](README.md) for settled behavior.

## Pageable-host rejection not exercisable on Graph

- **Where** — `compute()` still calls `require_cuda_accessible(rd, "rd_maps", …)`. `RejectsPageableHostInput` is `GTEST_SKIP`'d (`test_phase_shift_beamformer.cpp` lines 555–562).
- **Natural form** — A host-producer path in the Graph harness that delivers a pageable Tensor so the operator's reject message can be asserted.
- **Constraint** — `TensorSourceOp` always allocates a device loan and memcpy's into it (`bind_tensor_output_device`). The operator never sees a pageable-host Tensor. [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3 / §4.
- **Simplification trigger** — A host-producer / foreign-handle publish path in `sdk_harness.hpp`.
- **Blast radius** — Every Graph-ported op whose TESTING.md memory-space clause is skipped the same way.
