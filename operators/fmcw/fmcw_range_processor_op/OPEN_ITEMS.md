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

**Class:** Needs team look; potential contract change to the `[live]`
convention.

**Current behavior:** `waveform` is a sticky `[live]` parameter like any other
— the newest value received wins, and the operator has no way to tell which
ADC cube a given descriptor belongs to. Staleness in most `[live]` knobs
(`nfft`, `fft_norm`) degrades an output; staleness in `waveform` *invalidates*
it, because the range axis is scaled by the wrong bandwidth.

**Question:** should coherence-critical descriptors keep sticky semantics, or
does the convention need a way to bind a descriptor to the frames it applies
to? Measured on a scalar probe (8 frames, one waveform switch at frame 4, HSDK 5
EA `v5.0.0.1`):

| Mechanism | Coherent frames | Failure mode |
| --- | --- | --- |
| Sticky override, command 2 frames early | 6/8 | applied too early, undetectable |
| Sticky override, command 2 frames late | 6/8 | applied too late, undetectable |
| Descriptor carries an effective-from frame id, command early | 8/8 | none |
| Descriptor carries an effective-from frame id, command late | 6/8 | applied late, but the operator knows and can flag the output |
| ADC source publishes the waveform alongside the cube | 8/8 | none (requires the source to own both) |
| Per-frame companion + `OnAligned{0 ns}` | 8/8 | wrong content if the generator is out of step with the TX schedule |

Two observations worth carrying into the decision. Sticky is wrong in *both*
directions, and a bare descriptor carries no statement about which frames it
belongs to, so the operator cannot detect either case. And an aligned join
proves only that two samples describe the same instant — it cannot prove the
descriptor is right, so alignment does not by itself substitute for the source
owning both facts.

**Acceptance:** a team decision on whether `OPERATOR_API.md` grows an
effective-from (epoch) concept for coherence-critical `[live]` descriptors, or
whether coherence is instead an application obligation discharged by the ADC
source. The choice is cross-cutting — it applies equally to
`FMCWDopplerProcessorOp`, `FMCWRangeDopplerProcessorOp`, and
`MIMOCoherentCompensationOp`, which take the same `waveform`. Note this is
independent of the EA port blocker in
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1, which currently
prevents the `waveform` override from being wired at all.
