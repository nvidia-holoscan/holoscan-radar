# FMCWDopplerProcessorOp — open items

## 1. GPU-native window construction

**Class:** Performance follow-up; no contract change.

**Current behavior:** On a window cache miss, `doppler_stage.hpp` generates
taps on the host (`generate_window`) and uploads them through
`upload_coefficients`, which synchronizes the compute stream so its host
staging buffer can be released safely.

**Question:** Should window taps be generated directly on the GPU? The GPU
path could avoid host-side generation, staging, and synchronization for
unseen window configurations.

**Acceptance:** Benchmark cold ticks and unseen configuration changes across
the Doppler performance size ladder before choosing a CUDA kernel, a MatX
expression, or the current host-build path. Preserve cache reuse, stream
ordering, and numerical parity with `generate_window`.
