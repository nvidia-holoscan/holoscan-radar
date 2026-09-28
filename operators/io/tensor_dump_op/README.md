# `TensorDumpOp`

## Purpose

Write an arbitrary device tensor to a NumPy or raw file for diagnosis.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use this sink to inspect an intermediate tensor or formed image during development. It copies device data to host and writes synchronously; use every_n or max_files to limit file cost. It is deliberately untemplated because it accepts any dtype.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | tensor | any Tensor dtype | any rank | required | Data to dump |
| In | every_n | int64 | scalar | optional [live] | Per-tick throttle |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `TensorDumpOp` | Add to an HSDK 5 Graph | Own the named ports and Params |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `path` | `string` | required | Static | Output path pattern |
| `format` | `TensorDumpFormat` | npy | Static | NumPy or raw bytes |
| `every_n` | `int64` | 1 | [live] | Tick throttle |
| `max_files` | `optional<int64>` | unlimited | Static | File count cap |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
