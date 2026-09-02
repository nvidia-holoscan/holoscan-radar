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
