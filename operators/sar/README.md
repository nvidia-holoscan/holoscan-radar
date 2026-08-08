# SAR formation operators

GPU operators for synthetic-aperture image formation by time-domain
backprojection. The chain runs phase history → range profiles → image, with
optional autofocus, and is specified in
[`docs/planning/SAR/OPERATOR_API.md`](../../docs/planning/SAR/OPERATOR_API.md) —
the locked contract for every port, shape, dtype pair and parameter. Each
operator is templated on precision (`<float>` / `<double>`); module-wide port,
batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's header
carries its full port/parameter reference.

The standard-format readers and writers that bracket this chain are **not** in
this group: they live in [`operators/io/`](../io/README.md), so a build that
wants only formation pays nothing for the XML and NITF dependencies.

Typical placement (wave one, autofocus off — see
[`docs/planning/SAR/WORKFLOWS.md`](../../docs/planning/SAR/WORKFLOWS.md) for
the full wiring, including where `ApertureManagerOp` sits):

```text
CPHDReaderOp → phase_history → SARRangeCompressionOp → range_profiles → BackprojectionOp → image → …
```

## Operators

Nothing has landed yet. The group is scaffolded so that the first
implementation phase adds operators rather than build system. Planned
contents, in the build order set by
[`docs/planning/SAR/SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md):

| Operator | In → Out | Function |
| --- | --- | --- |
| `SARRangeCompressionOp<T>` | `phase_history` → `range_profiles` (+ `range_bins`) | Inverse transform of deramped phase history to the range axis |
| `BackprojectionOp<T>` | `range_profiles`, `positions`, `control` → `image` | Time-domain backprojection onto a `SampleGeometry` lattice or an explicit `sample_points` one, accumulating across ticks |
| `PhaseCorrectionOp<T>` | `phase_history`, `phase_correction` → `phase_history` | Broadcast multiply applying an autofocus estimate |
| `AutofocusOp<T>` | `image` → `phase_correction` | Phase-gradient autofocus estimate over a conjugate grid |
| `RangeReferenceMultiplyOp<T>` | `pulses` → `phase_history` | Direct-sampled branch entry: forward FFT and multiply by the range reference function |

## Implementation status

Empty. The contract was accepted on 2026-08-07 and implementation is
sequenced by the minimum-per-demo rule in
[`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md), so this group
fills in demo order rather than in the table order above:
`SARRangeCompressionOp` and `BackprojectionOp` first, for the mode 1 spotlight
demo, with autofocus and the direct-sampled branch later.

Changes to the SAR contract require the developer's approval in session before
the edit — see [`AGENTS.md`](../../AGENTS.md) → *Docs first*.
