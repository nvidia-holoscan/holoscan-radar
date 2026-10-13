# TensorDumpOp — open items

Items this operator carries. **The contract surface is complete** as of
2026-08-10: both formats, both throttles, and the `[live]` `every_n` override
port are implemented, and nothing in
[`OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md) → *TensorDumpOp*
is staged or throws. What follows is conventions the contract leaves open, a
measured cost consciously not paid, and one deviation from a module-wide
convention that the contract itself calls for.

Team-blocked questions belong in
[`docs/planning/SAR/TEAM_BLOCKED.md`](../../../docs/planning/SAR/TEAM_BLOCKED.md);
nothing here is one.

## 1. Not a class template, unlike every other operator

The module convention — and the new-operator checklist — is
`template <typename T>` with `T ∈ {float, double}`. This operator is a plain
class. The contract asks for that in as many words ("deliberately untyped …
the one operator in the module with no dtype pairing"), and invariant 3's
pairing has nothing to pair: there is no output, and the dtype recorded in the
`npy` header is read off the incoming tensor rather than assumed.

The alternative — template it and ignore the parameter — was rejected because
`TensorDumpOp<float>` would write a `complex128` dump perfectly well, which
makes the type argument a lie a reader would have to discover.

`operators/io/README.md` listed `TensorDumpOp<T>` in its planning table before
the operator existed; that row was corrected in the same change rather than
treated as a second reading of the contract. **This is a doc correction, not a
contract change** — the authoritative text always said untyped.

**Accepted by the developer in session on 2026-08-10, and indexed for the team.**
The convention it departs from is module-wide rather than SAR-scoped, so it
carries a row in [`docs/planning/TEAM_BLOCKED.md`](../../../docs/planning/TEAM_BLOCKED.md)
→ *Needs team look* — where FMCW owners see it without reading SAR docs — and a
pointer from [`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md)
→ *`setup()`*, so the next operator author does not read the rule as optional.
What the team ratifies is the general reading — that invariant 3 binds operators
that *emit* tensors, and a pure sink is outside it — not this operator. Nothing
is blocked either way: if the team reads the invariant literally, the fix is a
vestigial template parameter here, not a contract change.

## 2. The write is synchronous, and the filesystem is most of the cost

`performance/cpp/perf_tensor_dump.cpp`, on an RTX PRO 6000 Blackwell:

| Tensor | To a file | To `/dev/null` | Skipped tick |
| --- | --- | --- | --- |
| 512 KiB | 0.56 ms | 0.11 ms | ~17 µs |
| 8 MiB | 8.7 ms | 1.1 ms | ~10 µs |
| 32 MiB | 39.0 ms | 7.0 ms | ~10 µs |

So the device-to-host copy runs at 7–15 GB/s and everything above it is the
`write()`. A dump left wired into a 30 Hz graph at 8 MiB per frame will not
keep up; throttled with `every_n`, it costs ~10 µs and can stay.

Two costs are deliberately unpaid. The write is on the compute thread rather
than a writer thread with a bounded queue, and the copy is a fresh
`std::vector::resize` reuse into pageable host memory rather than a pinned
staging buffer. Both would help; neither is worth the complexity in a
debugging tap whose contract already provides the throttle, and the numbers
above are what would justify revisiting that. **The measurement, not a guess,
is the argument** — before this harness existed the plausible story was that
the D2H dominated, and it does not.

Not fsync'd: the numbers are the cost to the graph, not time to durable
storage.

## 3. `{index}` is the tick, and the contract could be read either way

"`{index}` is substituted with the tick counter" is implemented literally: under
`every_n: 3` the files are `..._0`, `..._3`, `..._6`, with gaps. The other
reading — a dense counter over files written — would name them `_0`, `_1`,
`_2`. The literal reading was taken because a dump's value is knowing which
message produced it, and because a dense counter is recoverable from the sorted
listing while the tick is not recoverable from a dense counter.

## 4. A path without `{index}` warns rather than failing

The contract calls `path` a pattern but does not require the token. Writing one
tensor to a fixed path is a real use, so it is legal; when more than one file
could be written the operator warns once at `start()` that each dump overwrites
the previous. Setting `max_files: 1` states the intent and silences it.

## 5. Dtypes NumPy cannot spell error under `npy` and dump under `raw`

`bfloat16` and multi-lane (vector) dtypes have no `.npy` type string. The
operator throws, naming `format: raw` as the way to get the bytes out. This is
the one input the "any dtype" contract cannot honour in one of its two formats,
and it fails loudly in the format that cannot rather than writing a header that
misdescribes the file.

## 6. Pageable host input is rejected, same as everywhere else

The contract says "any dtype, any shape" and is silent on memory space, so the
module-wide rule applies:
[`TESTING.md`](../../../docs/planning/SAR/TESTING.md) → *Memory spaces* makes
`CPHDReaderOp` the sanctioned exception and a debugging sink is not a second
one. Worth recording because the rationale for the rule — kernels need
CUDA-accessible memory — does not apply to an operator that only copies to
host, and `cudaMemcpyDefault` would have accepted pageable input silently.
Consistency won; if a caller ever needs to dump a host tensor, this is the item
to reopen.
