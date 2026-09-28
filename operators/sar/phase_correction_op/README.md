# `PhaseCorrectionOp`

## Purpose

Apply a per-pulse phase correction to every range bin of that pulse.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use when a phase estimator supplies a pulse-aligned correction before image formation. The sense parameter must explicitly choose add or subtract so the sign is visible at the graph boundary. Input and output names differ because HSDK 5 requires unique port names.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | range_profiles | complex T | (num_pulses,num_range_bins) | required | Input profiles |
| In | phase_correction | real T | (num_pulses) | required Exact | Per-pulse phase radians |
| Out | range_profiles_out | complex T | (num_pulses,num_range_bins) | every tick | Corrected profiles |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `PhaseCorrectionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `sense` | `PhaseCorrectionSense` | required | Static | Add or subtract per-pulse phase |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
