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

`CPHDReaderOp` and `TensorDumpOp` have landed; the rest are planned.
Contracted in
[`docs/planning/SAR/OPERATOR_API.md`](../../docs/planning/SAR/OPERATOR_API.md):

| Operator | In → Out | Function |
| --- | --- | --- |
| `CPHDReaderOp<T>` | — → `phase_history`, `positions`, `reference_position` (+ optional descriptors) | Streams pulses from a CPHD file, announcing the stream boundary |
| `CPHDWriterOp<T>` | `phase_history`, `positions` → — | Records phase history; paired with the reader it gives a round-trip test |
| `SICDWriterOp<T>` | `image` → — | Writes the formed image as a SICD product |
| `TensorDumpOp` | any tensor → — | Dumps a tensor to disk for offline inspection |

**`TensorDumpOp` carries no `<T>`, and it is the only operator here that
does not.** The contract calls it *deliberately untyped* — "the one operator in
the module with no dtype pairing" — because it exists to inspect a port whose
contract is still being written, and it reads the dtype off the tensor to
record in the `npy` header. A precision template would have nothing to pair
(there is no output) and would let `TensorDumpOp<float>` write a `complex128`
dump correctly, which is a worse lie than no template. This table said
`TensorDumpOp<T>` before the operator was implemented; that was a planning
placeholder, not a second reading of the contract.

## Implementation status

**`CPHDReaderOp` — landed, contract-complete.** All seven output ports, all
five parameters, `pulse_range` selection, short final batches, and both halves
of the boundary announcement (`collection_boundary` at a file seam,
`end_of_stream` at the last). The host-side parser
(`cphd_reader_op/cphd_file.hpp`) reads the ASCII header, the XML metadata, the
declared PVP layout, and the signal block in `CF8`, `CI4` and `CI2` with
per-vector `AmpSF` applied, in either the `FX` or `TOA` domain.

It reached that state by going past `SIGPROC_PLAN.md`'s minimum-per-demo rule
rather than by following it. Mode 1 needs one file, one channel, `CF8`, no
`AmpSF`, FX — and the rule scheduled the rest into mode 3, two demos later,
which is where the maxim *a reader that reads one collection is not a reader
that reads CPHD* came from. The checklist gate is the stronger claim: a
vertical slice never clears it. So the deferred paths were pulled forward, and
the concrete payoff is that all 229 surveyed public collections now read
instead of 115 — every Capella file needed `CI4` *and* `AmpSF` together, so
neither alone unlocked any of them. See
[`cphd_reader_op/OPEN_ITEMS.md`](cphd_reader_op/OPEN_ITEMS.md) for the
conventions and measured costs that remain.

`performance/cpp/perf_cphd_reader.cpp` measures it across batch size and sample
format. The reader was written up as out of harness scope for this wave — its
cost being disk rather than GPU — but a harness that reports bytes decoded and
uploaded per second says something useful about a source, and it is what found
that per-tick device allocation, not the read, sets the floor at small batches.

**`TensorDumpOp` — landed, contract-complete.** Both formats, both throttles,
and the `[live]` `every_n` override port. `npy` is written by
`tensor_dump_op/npy_writer.hpp`, a v1.0 writer with no Holoscan or CUDA in it,
so the format is testable without a graph; `raw` is the element bytes and
nothing else. A tick copies the tensor device-to-host on the received stream,
synchronizes, and writes — the sink has to see the producer's finished data,
so the wait is unavoidable and is part of what a dump costs.

Its own test cannot check the format, only that our writer and our parser
agree, so `tests/cpp/io/npy_readback.py` loads the dumps in NumPy as a second
CTest and asserts the values against a formula both sides compute
independently.

`performance/cpp/perf_tensor_dump.cpp` answers the question a caller actually
has — what does leaving a dump wired in cost? On an RTX PRO 6000, an 8 MiB
tensor takes ~8.7 ms per tick to a file and ~1.1 ms to `/dev/null`, so the
**filesystem is roughly seven eighths of it** and the device-to-host copy runs
at 7–15 GB/s. `npy` and `raw` are indistinguishable, the header being 128
bytes. A tick that `every_n` skips costs ~10 µs, which is the number that says
a throttled dump can stay wired in.

`CPHDWriterOp` and the
round-trip test it enables are deliberately deferred; see
[`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md) →
*Work no demo forces* for what that costs and what triggers pulling it
forward.

Python bindings are not written yet. The mode 1 demo is a C++ application, so
nothing forces them; they land with the first Python consumer.
