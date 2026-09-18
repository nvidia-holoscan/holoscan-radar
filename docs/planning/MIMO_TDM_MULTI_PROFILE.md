# MIMO TDM Multi-Profile Application

## Goal

Add `applications/fmcw/mimo_tdm_multi_profile`, demonstrating alternating
high-resolution and long-range FMCW CPIs with exact waveform/data coherence
between independently scheduled TX and RX source operators.

The graph contains one `FMCWTransmitSourceOp` and one
`CorrelatedMimoTdmAdcSourceOp`. Profile selection and device-specific driver
backends are internal responsibilities of those operators. A reusable bounded
frame coordinator correlates the two sources without adding frame tokens or
bespoke records to graph ports.

## Temporal architecture

The TX source selects a profile, invokes its driver, and receives an opaque
frame token, the profile actually applied, and the TX-applied
`holoscan::TimePoint`. It emits `wire::FMCWWaveformT` at that time and only then
marks the coordinator record RX-ready.

The independently triggered RX driver produces the actual ADC tensor and frame
token; simulation is one driver implementation, not source-operator behavior.
The RX source receives that pair from its backend.
It holds early frames in bounded storage, claims the matching coordinator
record, and emits the ADC cube at the TX-applied time. Therefore:

```text
waveform.capture_time == adc_cube.capture_time == applied transmit time
```

Existing `Exact{}` waveform inputs on range, Doppler, and TDM compensation then
match without approximate joins or application-specific glue.

Add reusable Holoscan-aware helpers under `applications/common/radar_app/`:

- Opaque `FrameToken` and internal `TdmTransmitProfile`, `TransmitReceipt`, and
  `ReceivedAdcFrame` records.
- Internal `FMCWTransmitDriver` and `MimoTdmReceiveDriver` backend interfaces.
- A bounded, thread-safe `TransmitFrameCoordinator`.
- `FMCWTransmitSourceOp` and `CorrelatedMimoTdmAdcSourceOp`.

The simulator-backed reference driver uses the existing simulator API and owns
signal generation. Hardware drivers may replace it without rewriting the RX
source, changing ports, or changing coordinator semantics.

Duplicate tokens, duplicate claims, capacity overflow, expired unmatched
frames, invalid applied profiles, failed waveform publication, and shutdown
with unmatched frames fail the run. A failed waveform emit never releases its
ADC frame. Records are erased after one successful claim.

## Profiles and graph

Author physical profiles under `app.profiles` and a name sequence under
`app.profile_sequence`. This is the sole waveform, cube-shape, and physical
PRI authority; do not repeat a baseline `app.waveform`, `app.cube`, or
operator waveform. Each profile owns waveform fields, physical PRI,
chirps per TX, and samples per chirp. Derive the DSP descriptor with
`chirp_repetition_interval = num_tx * physical_pri`; never author both values.

Ship these defaults at 77 GHz and 12.8 MHz:

| Profile | Bandwidth | Duration | Samples | Physical PRI | 2-TX DSP PRI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `near` | 300 MHz | 20 us | 256 | 25 us | 50 us |
| `far` | 150 MHz | 40 us | 512 | 45 us | 90 us |

Both use 64 chirps per TX and the default sequence `[near, far]`. Keep carrier,
TX count, TX schedule, array geometry, and structural operator parameters
static. Validate unique names, sequence references, integral sample count,
physical PRI at least chirp duration, and frame period longer than the longest
complete TDM CPI. Compute `MaxTensorExtents` as maxima across profiles and use
`nfft = 0` for range and Doppler.

Run TX and RX in separate partitions. The reference TX source uses the frame
period; RX polls independently at 1 ms, with a simulated default receive
latency of 2 ms. Use `max_frames_in_flight = 4`, coordinator capacity four, and
non-dropping waveform fan-out queues of depth five.

Reuse the existing TDM processing chain after the sources: deinterleave, range,
clutter, Doppler, CFAR, sparse compensation, beamscan, feature assembly,
clustering, reduction, report, and optional dashboard. Set static waveform
Params invalid so the live descriptor is authoritative. Keep default targets
inside the principal CRI interval and both profiles' operational range.

The dashboard is profile-stable rather than sequence-rotating. Optional
`app.visualization.profile` names the displayed profile and defaults to the
first scheduled profile. The dashboard receives the TX-applied waveform as an
`Exact{}` side input, renders only matching CPIs, and advances its scenario
clock across every consumed CPI. Its histories therefore contain one profile
only, while Holoviz holds the last selected image for the maximum cyclic gap
derived from `profile_sequence`. Unknown or unscheduled selections are
configuration errors even when visualization is disabled.

Interactive profile selection is deferred to the HSDK 5 open item in
`HSDK5_PORT.md` section 6; the fixed selection does not introduce an X11
event workaround or a radar-owned presentation loop.

## Contracts and verification

Update `OPERATOR_API.md` with the narrow rule that an authoritative source may
output an existing shared descriptor representing configuration it actually
applied. This does not permit processing-operator descriptor echoes or add a
shared descriptor type.

Update `TEMPORAL_CONTRACTS.md` to assign timestamp ownership to TX-applied time
and document coordinator causality and queue sizing. Add the workflow and
grouped-operator diagram to `WORKFLOWS.md`, the application README, and
`control-plane-diagram.mmd`.

Tests cover coordinator ordering and failures, deliberately different source
activation times, exact waveform/ADC capture times, alternating descriptor and
tensor shapes, per-profile range/velocity axes, physical-versus-effective TDM
PRI, configuration validation, complete-chain smoke, and Exact dashboard
profile filtering while the processing path changes range-axis lengths.
