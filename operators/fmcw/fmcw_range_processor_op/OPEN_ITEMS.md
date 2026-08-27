# FMCWRangeProcessorOp — open items

## 1. GPU-native coefficient construction

**Class:** Performance follow-up; no contract change.

**Current behavior:** On a premultiplier or window cache miss,
`range_stage.hpp` generates synthesized dechirp references, window taps, and
the combined premultiplier on the host, then uploads the coefficient tensor to
the GPU. The upload path synchronizes the compute stream so its host staging
buffer can be released safely.

**Question:** Should synthesized references, window taps, and the combined
premultiplier be generated directly on the GPU? Caller-provided
`reference_samples` would still require an upload, but the GPU path could
avoid host-side generation, staging, and synchronization for synthesized
configurations.

**Acceptance:** Benchmark cold ticks and unseen configuration changes across
the range performance size ladder before choosing a CUDA kernel, a MatX
expression, or the current host-build path. Preserve cache reuse, stream
ordering, and numerical parity with `radar_common/sim/`.

## 2. Coherence of the `waveform` override with the cube it describes

**Class:** Settled (L16 / L17, 2026-08-21); residual stamp-ownership question.

**Decision:** `[live]` ports join the data trigger with `Exact{}` and
`on_missing SafeDefault{}`. The application stamps the frame's transmit time
on both the override and the cube. `sticky_*` (default true) latches the
last Exact match; false reverts to Params on a miss. No epoch field in the
descriptor. The override entries are unconditional and an unwired port is
admitted by the SDK's optional-input rule, so wiring is the application's only
obligation and `sticky_waveform` is the sole per-port static
([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). `waveform` is
restored through a generated
`holoscan::radar::wire` schema payload; the logical parameter remains
`FMCWWaveformDescriptor`. This physical encoding is interim pending team
ratification. See
[`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md),
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9, and the
payload-shape/schema item in
[`TEAM_BLOCKED.md`](../../../docs/planning/TEAM_BLOCKED.md).

**Still open:** who owns the transmit stamp (host sequencer vs device
report). The consumer contract is the same either way — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).
