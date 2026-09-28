# `CPHDReaderOp`

## Purpose

Read one selected CPHD channel and emit device phase history with its pulse geometry and an explicit boundary on each batch.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use this source for recorded monostatic CPHD collections. It decodes selected pulse ranges and channels and keeps the collection boundary on the final batch so a downstream aperture manager can flush before the next collection begins.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| Out | phase_history | complex T | (num_pulses,num_samples) | every batch | Decoded samples |
| Out | positions, reference_position | real T | (num_pulses,3) | every batch | APC and SRP |
| Out | sample_axis, pulse_times, velocities | real T / float64 | named axes | optional | Diagnostics |
| Out | coordinate_frame | descriptor | scalar | optional | Declared frame |
| Out | stream_boundary | uint8 | scalar | every batch | none / collection / EOS |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `CPHDReaderOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `path` | `CollectionPaths` | required | Static | CPHD file, directory, or ordered list |
| `coordinate_frame` | `CoordinateFrame` | required | Static | Validate position frame |
| `pulses_per_tick` | `int64` | required | Static | Pulse batch limit |
| `channel` | `int64` | 0 | Static | Selected CPHD channel |
| `pulse_range` | `vector<int64>` | empty | Static | Half-open per-file selection |
| `loop` | `bool` | false | Static | Repeat the file list |
| `playback_rate` | `double` | 0 | Static | Pace by TxTime |
| `duty_cycle` | `double` | 1 | Static | Gap between collections |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
