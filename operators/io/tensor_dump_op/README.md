# `TensorDumpOp`

## Purpose

Write an arbitrary device tensor to a NumPy or raw file for diagnosis.

## Responsibilities

- Throttle diagnostic writes by input tick and a same-tick optional override.
- Preserve tensor dtype and shape in `.npy`, or write raw bytes when selected.

## When to use

Use this sink to inspect an intermediate tensor or formed image during development. It copies device data to host and writes synchronously; use every_n or max_files to limit file cost. It is deliberately untemplated because it accepts any dtype.

Declarations and parameter defaults: [tensor_dump_op.hpp](tensor_dump_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `tensor` | Contiguous CUDA-accessible Tensor and supported dtype | any rank | Required | triggers each tick |
| In | `every_n` | `int64` | scalar | Optional `[live]`, same-tick `Exact{}`/`SafeDefault{}` | Tick-local file throttle |

There are no outputs. The sink writes a file only for selected ticks.

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `TensorDumpOp` | Add to an HSDK 5 Graph | Own the named ports and Params |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `path` | `string` | required | Static | Output path pattern |
| `format` | `TensorDumpFormat` | npy | Static | NumPy or raw bytes |
| `every_n` | `int64` | 1 | [live] | Tick throttle |
| `max_files` | `optional<int64>` | unlimited | Static | File count cap |

## File behavior and limits

The sink counts incoming tensor ticks from zero. It writes a tick when
`index % every_n == 0`, until `max_files` is reached. A connected `every_n`
port overrides the parameter for **that tick only**; a missing value falls
back to Params. `max_files: 0` writes nothing. `{index}` in `path` expands to
the input tick number. Without that token, later writes overwrite the same
file, and the operator warns unless at most one file can be written.

`npy` records dtype and shape in a NumPy header; `raw` writes only tensor
bytes, so a caller must preserve dtype and shape separately. The tensor may
have any rank and supported dtype, but it must be contiguous and CUDA
accessible. The operator copies it to host memory, synchronizes the CUDA
stream, creates parent directories, and writes synchronously. Expect disk
latency in the Graph's critical path; throttle or cap diagnostic dumps.

For 2-D floating-point or complex NumPy images saved within a development
checkout, saved paths are relative to the repository root and the save log
also prints a shell-quoted host viewer command. Files outside the checkout
retain absolute paths.
`uv run applications/sar/tools/view_image.py <image.npy>` saves a log-magnitude PNG
beside the array; see the [viewer guide](../../../applications/sar/tools/README.md#view-saved-images).
The viewer is an optional host tool and is not a runtime dependency of the sink.

## Failure cases and use

An empty `path`, `every_n < 1`, or negative `max_files` is invalid. Unsupported
NumPy dtypes, a failed device copy, and file-system errors fail the tick.
Example operator configuration:

```yaml
tensor_dump:
  path: /tmp/sar/image_{index}.npy
  format: npy
  every_n: 1
  max_files: 6
```

The standalone SAR applications expose an app-level `dump: true` toggle and
`output` settings; see their guides for the actual wiring. The C++ tests
check NumPy headers, bytes, tick throttling, output limits, and invalid
inputs. The public contract is [architecture](../../../docs/ARCHITECTURE.md).

## Behavior and qualification notes

This sink is intentionally untemplated: it preserves the incoming tensor's dtype rather than computing a new precision. Whether a future API should template sinks remains a product question. File writes and device-to-host staging are synchronous costs; do not assume dumping is free. `{index}` counts input ticks, not saved files, so `every_n` leaves gaps. A path without `{index}` warns and overwrites according to the existing policy. Unsupported NumPy dtypes fail for `npy`; raw output preserves bytes. Inputs still require CUDA-accessible storage; pageable-host input is not a fallback path.
