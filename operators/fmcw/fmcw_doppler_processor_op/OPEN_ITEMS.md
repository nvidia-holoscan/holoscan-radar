# FMCWDopplerProcessorOp — open items

## 1. Schema-backed `[live]` payload ratification

**Class:** Needs team look; interim implementation does not block landing.

`doppler_window` and `waveform` are restored as Exact `[live]` Inputs through
generated `holoscan::radar::wire` physical payloads while the logical Params
remain `WindowSpec` and `FMCWWaveformDescriptor`. Descriptor bulk arrays use
a present rank-1 zero-length Tensor for absence; null Tensor members are
invalid.

The team must ratify, revise, or reject this physical payload shape and its
schema-evolution obligations. See
[`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md),
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9, and
[`TEAM_BLOCKED.md`](../../../docs/planning/TEAM_BLOCKED.md).
