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

**Build dependency: `tinyxml2`** (`apt-get install libtinyxml2-dev`), required
only when `OP_cphd_reader_op` is on. It is linked `PRIVATE` and named by no
public header, so it does not propagate to consumers.

**When one of these operators joins the Debian package, add its library to
`DEPENDS`.** `pkg/CMakeLists.txt` currently lists four FMCW operators and
`fmcw_siso`, so nothing here ships today and
`pkg/holoscan-radar/CMakeLists.txt` correctly declares only Holoscan. Adding
`cphd_reader_op` to the package without adding `libtinyxml2-10` beside it
would produce a `.deb` that installs and then fails to link — noted here
because the two files are far apart and nothing connects them automatically.

Extend it as one comma-separated string, `DEPENDS "holoscan (>= 4.3.0),
libtinyxml2-10"`. `holohub_configure_deb` takes `DEPENDS` as a single-value
argument and passes it straight to `CPACK_DEBIAN_PACKAGE_DEPENDS`, so a second
quoted argument would be silently dropped rather than appended.

## Operators

`CPHDReaderOp` has landed; the rest are planned. Contracted in
[`docs/planning/SAR/OPERATOR_API.md`](../../docs/planning/SAR/OPERATOR_API.md):

| Operator | In → Out | Function |
| --- | --- | --- |
| `CPHDReaderOp<T>` | — → `phase_history`, `positions`, `reference_position` (+ optional descriptors) | Streams pulses from a CPHD file, announcing the stream boundary |
| `CPHDWriterOp<T>` | `phase_history`, `positions` → — | Records phase history; paired with the reader it gives a round-trip test |
| `SICDWriterOp<T>` | `image` → — | Writes the formed image as a SICD product |
| `TensorDumpOp<T>` | any tensor → — | Dumps a tensor to disk for offline inspection |

## Implementation status

**`CPHDReaderOp` — landed, for the mode 1 subset.** All seven output ports,
`pulse_range` selection, short final batches, and the `end_of_stream`
announcement work; the host-side parser (`cphd_reader_op/cphd_file.hpp`) reads
the ASCII header, the XML metadata, the declared PVP layout, and the signal
block. Per `SIGPROC_PLAN.md`'s minimum-per-demo rule it implements only what
the mode 1 collection needs — a single file, one channel, `CF8` samples, no
`AmpSF`, FX domain. **A reader that reads one collection is not a reader that
reads CPHD**: multi-file `path`, `CI4`/`CI2`, `AmpSF` and TOA domain are mode 3
obligations, and each throws a clear error rather than guessing until then. See
[`cphd_reader_op/OPEN_ITEMS.md`](cphd_reader_op/OPEN_ITEMS.md).

`TensorDumpOp` is next — it is how intermediates are inspected while the rest
is built, and it is the mode 1 demo's first sink. `CPHDWriterOp` and the
round-trip test it enables are deliberately deferred; see
[`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md) →
*Work no demo forces* for what that costs and what triggers pulling it
forward.

Python bindings are not written yet. The mode 1 demo is a C++ application, so
nothing forces them; they land with the first Python consumer.
