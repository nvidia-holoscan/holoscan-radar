# `PhaseCorrectionOp`

## Purpose

Apply a per-pulse phase correction to every range bin of that pulse.

## Responsibilities

- Apply a declared add or subtract per-pulse phase to every range bin.
- Require exact capture-time alignment and preserve profile shape and precision.

## When to use

Use when a phase estimator supplies a pulse-aligned correction before image formation. The sense parameter must explicitly choose add or subtract so the sign is visible at the graph boundary. Input and output names differ because HSDK 5 requires unique port names.

## Ports

The two required inputs join with `Exact{}` at the same capture time. `T=float`
means complex64/float32; `T=double` means complex128/float64.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `range_profiles` | complex `T` | `(num_pulses,num_range_bins)` | Required, aligned | Pulse data |
| In | `phase_correction` | real `T`, radians | `(num_pulses)` | Required, aligned | Per-pulse phase in radians |
| Out | `range_profiles_out` | complex `T` | `(num_pulses,num_range_bins)` | Every aligned tick | Phase-rotated profiles |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `PhaseCorrectionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

Set `max_pulses` and `max_range_bins` to cover the largest aligned tick. The output Tensor pool is sized from these construction bounds before arm.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `sense` | `PhaseCorrectionSense` | required | Static | Add or subtract per-pulse phase |

## Operation and synchronization

For pulse `p` and range bin `r`, the output is
`range_profiles[p,r] * exp(j * s * phase_correction[p])`, where `s = +1`
for `sense: add` and `s = -1` for `sense: subtract`. Correction values are
radians. This is an explicitly signed phase rotation, with no cross-tick
state and no image formation. Both required input tensors join with `Exact{}`
at the same capture time; there is no missing-value fallback.

Profiles and correction must be contiguous CUDA-accessible tensors. Their
shapes are `(num_pulses,num_range_bins)` and `(num_pulses)` with matching
positive pulse counts within `MaxTensorExtents`. Profile dtype is complex
`T` and correction dtype real `T`; output dtype and shape match profiles.
The operator rejects an invalid `sense`, mismatched pulse axes, invalid
dtypes, or missing CUDA stream. `sense` is static for the instance.

## Use and verification

```yaml
phase_correction:
  sense: subtract
```

Connect a pulse-aligned phase estimator upstream and the corrected profiles
to a subsequent formation graph. None of the five shipped SAR apps currently
uses this optional autofocus bridge. The C++ tests assert both sign choices
against an analytic phase rotation and check shape errors. See the
[SAR operator contract](../../../docs/planning/SAR/OPERATOR_API.md).
