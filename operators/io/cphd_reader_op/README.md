# `CPHDReaderOp`

## Purpose

Read one selected CPHD channel and emit device phase history with its pulse geometry and an explicit boundary on each batch.

## Responsibilities

- Decode one selected CPHD channel, PVP geometry, and optional diagnostics into device tensors.
- Stamp every pulse batch with a typed, capture-time-aligned collection boundary.

## When to use

Use this source for recorded monostatic CPHD collections. It decodes selected pulse ranges and channels and keeps the collection boundary on the final batch so a downstream aperture manager can flush before the next collection begins.

## Ports

There are no input ports: this is a clock-triggered source. `T=float` means
complex64/float32; `T=double` means complex128/float64.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| Out | `phase_history` | complex `T` | `(num_pulses,num_samples)` | Every batch | required data |
| Out | `positions` | real `T` | `(num_pulses,3)` | Every batch | antenna phase centre |
| Out | `reference_position` | real `T` | `(num_pulses,3)` | Every batch | per-pulse SRP |
| Out | `sample_axis` | real `T` | `(num_samples)` | Optional, latched | frequency samples |
| Out | `pulse_times` | float64 | `(num_pulses)` | Optional, latched | transmit times |
| Out | `velocities` | real `T` | `(num_pulses,3)` | Optional, latched | only when both Tx/Rx velocities exist |
| Out | `coordinate_frame` | `CoordinateFrame` descriptor | descriptor | Optional, latched | configured ECEF frame |
| Out | `stream_boundary` | `uint8` | scalar | Every batch | `0` none, `1` collection, `2` EOS |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `CPHDReaderOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

`max_pulses` and `max_samples_per_pulse` must cover the largest selected reader batch and channel. The Graph captures these bounds before arm; it does not resize the output pool from later file headers.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `path` | `CollectionPaths` | required | Static | CPHD file, directory, or ordered list |
| `coordinate_frame` | `CoordinateFrame` | required | Static | Validate position frame |
| `pulses_per_tick` | `int64` | required | Static | Pulse batch limit |
| `channel` | `int64` | 0 | Static | Selected CPHD channel |
| `pulse_range` | `vector<int64>` | empty | Static | Half-open per-file selection |
| `loop` | `bool` | false | Static | Repeat the file list |
| `playback_rate` | `double` | 0 | Static | `0` unmetered (wait on full); `>0` pace by TxTime |
| `duty_cycle` | `double` | 1 | Static | Gap between collections |
| `wait_for_downstream` | `bool` | false | Static | Paced ingest waits on full queues |

## Decode and timing

`path` accepts a file, directory, or ordered list; directory entries are sorted
by filename. The reader validates all file headers before emitting any data.
`pulse_range` is a half-open `[first, last)` selection applied per file.
`channel` selects one CPHD channel. `pulses_per_tick` bounds each emitted batch;
the last batch may be shorter. With `loop: true`, the ordered file list repeats.
A positive `playback_rate` paces collection time relative to wall time;
`duty_cycle` adds an inter-collection gap. Collect time is the finite `TxTime`
span (non-finite PVP samples are skipped; some Umbra vectors store NaN there).
Playback slack and duty-cycle quiet sleeps return early when the run requests
cooperative stop. Unmetered `playback_rate` 0 always retries required Tensor
emits on `kBackpressured` / `kResourceExhausted` through the shared helper.
HSDK 5 `QueuePolicy::kBlocking` does not park `OnClock` until the consumer
has room, and a failed `emit(TensorOutputLoan)` consumes the loan, so the
reader waits and re-prepares. A positive `playback_rate` fail-fasts with the
shared explanation unless `wait_for_downstream: true`. Only this operator
takes the Param; formation operators fail-fast. The wait returns on
cooperative stop. The lag HUD
is live target versus wall (`TxTime` span / `playback_rate` versus now since
the collection start), matching the HSDK 4 `playback_status()` poll. Pace
sleep uses that same slack (no sleep while behind). A paced reader
(`playback_rate > 0`) also WARNs at most once per second while behind by
0.2 s or more (the Gantt lag-band entry), and stops once behind is back to
0.05 s. The WARN prints `target` (paced transmit time) and wall so behind is
`wall − target`. Unpaced `playback_rate: 0` does not warn. Per-batch
collect/wall/`signal`/`decode`/`emit`/`slack` timing is DEBUG
(`HOLOSCAN_LOG_LEVEL=DEBUG`). Apps put the reader and backprojection in
separate Graph partitions so formation cannot steal the playback clock.

The CPHD PVP fields `TxTime`, `TxPos`, `RcvPos`, `SRPPos`, `SC0`, and `SCSS`
are required. The antenna phase centre is `(TxPos + RcvPos) / 2`; the
reference position is `SRPPos`. `sample_axis[i] = SC0 + i * SCSS` when that
output is connected. `pulse_times` remains float64 even for a `float` reader.
`velocities` emits only if both transmit and receive velocity fields exist.
The optional coordinate-frame output carries the configured ECEF declaration.
All tensors are device tensors; each template instance fixes complex and real
precision together. CF8 collections read stored bytes into reused pinned
scratch, H2D 8 bytes/sample, and byte-swap / widen / apply AmpSF on the
device; integer formats still decode on the host through `read_signal_into`.

`stream_boundary` is emitted on **every** batch at the batch's capture time:
`0` means more pulses in the collection, `1` closes a collection followed by
another file or loop iteration, and `2` closes the final collection. Connect
it to `ApertureManagerOp.stream_boundary` with the same batch's profiles and
geometry. If an app splits a batch, put the original boundary on its last
slice and `0` on earlier slices. After the last batch the reader stops
producing and parks this OnClock worker; it does not emit a separate empty EOS
tick. HSDK `request_stop` does not drain queued samples, so requesting stop on
the next 1 µs tick aborts the in-flight D22 `EmitAndReset`. Finite (non-loop)
SAR apps request session stop after the formed image and `visualize_hold_s`.

## Validation and failure cases

The configured frame must be ECEF. The reader rejects missing paths, an empty
file selection, invalid channel or pulse bounds, missing required PVP fields,
a frequency step exceeding `1e-5` relative to the batch's first `SC0` or
`SCSS` value, malformed CPHD metadata, and sample or pulse extents beyond its
`MaxTensorExtents`. Smaller recorded-grid ramps are accepted; `sample_axis`
uses the first pulse's values for that batch. It warns once per file
when non-finite phase-centre coordinates pass through. Optional outputs are latched: an unwired first emit is absorbed, then
subsequent work for that output is skipped.

## Use and verification

A SAR app config supplies the operator block directly; for example:

```yaml
cphd_reader:
  coordinate_frame: ecef
  pulses_per_tick: 1024
  pulse_range: [0, 4096]
```

The application owns `cphd.path` and passes it to the reader. For a small
offline file, run `ctest --test-dir build -R '^Sar_.*_smoke$' --output-on-failure`
**inside a development container** after configuring the test build. This
uses the C++ fixture generator. The CPHD reader C++ tests cover decoding,
selection, boundaries, and malformed inputs. Real-collection validation is
tracked in the [SAR open-items roll-up](../../../docs/planning/SAR/OPEN_ITEMS.md); see
[SAR data preparation](../../../tools/sar_data/README.md).
