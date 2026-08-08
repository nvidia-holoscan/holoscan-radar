# Standard-format I/O operators

Library-provided sources and sinks for the standard radar interchange formats,
plus the tensor dump sink. These are chain **boundaries**, never inline
stages: a reader is a source, a writer is a sink, and no signal processing
happens in either.

## Why this is its own group

The standard-format operators carry XML and NITF dependencies that the
signal-processing operators do not. A separate group puts each behind its own
`OP_<name>` gate, so a build that wants only the formation chain pays nothing
for the format libraries. `TensorDumpOp` sits here because it is a sink, not
because it shares the dependency.

The grouping is by dependency, not by function: `CPHDReaderOp` is ingress and
`CPHDWriterOp` is egress, and they are neighbors here because they parse and
emit the same XML.

## Operators

Nothing has landed yet. The group is scaffolded so that the first
implementation phase adds operators rather than build system. Planned
contents, contracted in
[`docs/planning/SAR/OPERATOR_API.md`](../../docs/planning/SAR/OPERATOR_API.md):

| Operator | In → Out | Function |
| --- | --- | --- |
| `CPHDReaderOp<T>` | — → `phase_history`, `positions`, `reference_position` (+ optional descriptors) | Streams pulses from a CPHD file, announcing the stream boundary |
| `CPHDWriterOp<T>` | `phase_history`, `positions` → — | Records phase history; paired with the reader it gives a round-trip test |
| `SICDWriterOp<T>` | `image` → — | Writes the formed image as a SICD product |
| `TensorDumpOp<T>` | any tensor → — | Dumps a tensor to disk for offline inspection |

## Implementation status

Empty. `CPHDReaderOp` and `TensorDumpOp` land first — the reader gates
everything downstream, since without it there is no data to test with, and the
dump operator is how intermediates are inspected while the rest is built (and
is the mode 1 demo's first sink). `CPHDWriterOp` and the round-trip test it
enables are deliberately deferred; see
[`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md) →
*Work no demo forces* for what that costs and what triggers pulling it
forward.
