# CPHDReaderOp — open items

Items this operator carries. None of them block the mode 1 demo; each is a
place where the landed slice is narrower than the contract, or where a
convention was chosen that real collections have not yet tested.

Team-blocked questions belong in
[`docs/planning/SAR/TEAM_BLOCKED.md`](../../../docs/planning/SAR/TEAM_BLOCKED.md);
nothing here is one yet.

## 1. The phase-centre convention is a choice, not a reading of the contract

`positions` is contracted as "per-pulse Aperture Reference Point positions",
and CPHD stores `TxPos` and `RcvPos` separately. The operator emits their
**midpoint**, which is exact for a monostatic collection (where the two
coincide, as they do in the simulator's fixtures) and is the conventional
bistatic approximation otherwise. `velocities` follows the same rule.

The contract does not say which of the three readings it means, and
[`SIGPROC_PLAN.md`](../../../docs/planning/SAR/SIGPROC_PLAN.md) names "the
Tx/Rcv phase-centre convention" as one of the reader paths the mode 3
collections force.

*Updated 2026-08-09 by the public-collection survey below.* The two **always**
differ in real data: across 228 Umbra and Capella collections the separation
is 19.5–52.9 m, median ~35 m, and never zero. It is the platform moving
through the pulse round-trip — real collections do not assume stop-and-go, and
our simulator's coincident Tx/Rcv is the special case, not theirs. So the
midpoint is not a formality that happens to be exact; it is a real choice
being exercised on every pulse of every real file, and at X-band tens of
metres is a large number of wavelengths.

This does not make the midpoint wrong — it is the conventional bistatic
approximation, and it stays the choice. What changed is that "untested because
no collection distinguishes them" is no longer true, so the question is now
answerable: form the same Umbra scene under midpoint, `TxPos`, and `RcvPos`
and see which focuses. Worth doing when `BackprojectionOp` can form an image,
since focus quality is the only test that discriminates. Still a contract
clarification rather than a bug, since the contract permits any of the three.

## 2. The paths deferred to mode 3, and what each does instead

Every one throws with a message naming what it found, rather than reading the
file under an assumption that would produce a plausible wrong image:

- **Multi-file `path`.** `path` is a single string. The directory and ordered
  list forms, and with them the `collection_boundary` half of the boundary
  announcement, land when a demo reads more than one file. Only
  `end_of_stream` is emitted today.
- **`CI4` and `CI2` sample decoding.** The parser recognises the formats and
  reports them by name; only `CF8` decodes.
- **`AmpSF`.** A file carrying per-vector amplitude scale factors is refused
  outright. Reading it while ignoring them would understate amplitudes
  silently, which is worse than not reading it.
- **TOA-domain collections.** Refused. `sample_axis` is built as a frequency
  axis from `SC0`/`SCSS`, which is the FX-domain reading.

## 3. Frequency-axis constancy is enforced per batch, not per collection

`SC0` and `SCSS` are per-vector fields, but `sample_axis` is one shared row, so
the operator checks that they are constant across the pulses of each batch and
throws otherwise. A collection that stepped its start frequency **between**
batches at exactly the batch boundary would slip through. Cheap to close by
checking the whole channel at `start()`; not done because it costs a full-file
PVP pass at open, which on a 28 GiB collection is not free, and no known
collection behaves that way. Revisit if one does.

## 4. Endianness — closed, and not the portability question it looked like

*Resolved 2026-08-08.* CPHD is big-endian and the parser swaps
unconditionally. The host being little-endian is a **precondition, not an
assumption**: invariant 8 makes a CUDA-capable NVIDIA GPU a hard dependency,
and every target NVIDIA ships CUDA for is little-endian (x86_64, aarch64
Linux, ppc64*le*). There is no machine this module can run on where the swap
is wrong.

It is now a `static_assert` on `__BYTE_ORDER__`, verified available under both
g++ and nvcc, and the three `if (host_is_little_endian())` branches are gone.
Runtime detection was rejected deliberately: it would add a big-endian path no
buildable machine could execute and therefore no test could cover, which is
the same objection invariant 8 raises to CPU fallback paths. A big-endian host
now fails to compile with a message saying what to do.

Recorded rather than deleted because the original entry read as an open
portability risk when the real defect was narrower — a function named
`host_is_little_endian()` that returned a hardcoded `true`, so the code looked
like it checked something it did not.

## 5. XML namespace handling — hand-rolled, and now exercised on real files

tinyxml2 has no namespace support: it reports element names exactly as
written, so a document using a prefix yields `cphd:Data` rather than `Data`.
The parser compensates by matching on the local name after any colon
(`local_name_is`), which accepts both that and the default-namespace form
CPHD normally uses.

*Closed 2026-08-09.* This was flagged as the likeliest place a real collection
would surprise the reader. It was not: all 231 public files parsed, and not
one used a prefixed form, so they all exercise the same default-namespace
branch our simulator does. The concern was misplaced — the risk was never
namespaces, it was that no real file had been read at all.

Since nothing in the wild writes a prefixed document, one was manufactured: a
real Umbra collection re-serialised with every element under an explicit
`cphd:` prefix, header offsets recomputed around the longer XML block, PVP and
signal bytes untouched. The reader's output is **identical** to the original
file's, field for field including signal energy. `local_name_is` does what it
claims, and the branch is no longer untested.

## 6. Remaining hot-path cost: synchronous default-stream uploads

The tick's four to five `cudaMemcpy` calls are synchronous on the default
stream rather than async on the operator's own. Defensible for a source whose
cost is the read, but it serializes against downstream work, and once
`BackprojectionOp` is doing real work per tick that will show. Left until the
reader reaches the performance harness, since a change here wants a
measurement of the whole chain rather than of the reader alone.

*Two costs closed, both measured rather than assumed:*

- **The parameter block was read once per field** — eight passes over the same
  interleaved bytes every tick. `read_pvp_block` reads the span once and the
  fields come out of host memory: exactly **8.00x** fewer bytes through
  `read()`, identical values.
- **The signal was copied three times** — raw bytes, byte-swapped
  `complex<float>`, then "narrowed" to `complex<T>`, which for the
  single-precision instance everyone uses was an identity copy that did
  nothing. It existed because `read_signal` hardcoded `complex<float>` while
  the operator is templated. `read_signal_into` now reads at the caller's
  precision directly into the caller's buffer: CF8 is bit-for-bit big-endian
  `complex<float>`, so a float instance byte-swaps in place and makes **no**
  intermediate copy. Measured **6.03x** faster per tick on the 2048x2048
  fixture and a third of the peak buffer, identical values. The double
  instance still needs one scratch buffer to widen through, which is why the
  two precisions take different paths and why a test pins them against each
  other.

## 7. Integer-typed PVP fields were decoded as doubles — fixed

*Found and fixed 2026-08-09 by the survey below; confirmed against an
independent reader.*

`PvpBlock::field()` calls `read_be_double()` on every field regardless of what
the XML declares. CPHD PVP fields are not all `F8`: `SIGNAL` is `I8`, a signed
64-bit integer. Its bits reinterpreted as a double give `5e-324`, the smallest
denormal, where the value is `1`. sarkit reads `1.0` from the same bytes on
all 218 surveyed files that carry the field.

`PvpField::format` is already parsed and stored — `field()` simply never
consults it, so the information needed to decode correctly is sitting in the
struct, unused.

**Latent, not active.** The operator reads only `F8` fields (`TxPos`,
`RcvPos`, `TxVel`, `RcvVel`, `SRPPos`, `TxTime`, `SC0`, `SCSS`), so nothing it
emits today is wrong. The exposure is that `field()` is a general accessor
that returns silent garbage for any integer field, and `SIGNAL` is precisely
the field a later consumer will want, since it marks which vectors hold valid
signal — a reader that ignores it processes flagged-bad pulses as if they were
good.

**Fixed by decoding to the declared format, with the precision limit
enforced rather than documented.** `field()` now reads the `Format` text it
was already storing: `F8` as before, `I8` and `U8` as integers converted to
double. Since a `double` cannot hold every `int64` exactly, a magnitude above
2^53 throws instead of returning a quietly rounded value, naming the field and
the value. Nothing in the CPHD defined set can reach that — `SIGNAL` is the
only integer field in it and is a 0/1 flag — but a user-added field carrying a
64-bit identifier could, and silent rounding there would be the same class of
bug as the one being fixed. A `U8` with its top bit set throws for the same
reason.

Compound formats (`X=F8;Y=F8;Z=F8;`) take their type from the first component,
which is correct for every collection surveyed; a genuinely mixed record would
need per-component types, and none exists.

`SIGNAL` now reads `1` on both the fixture and real Capella data, float fields
are unchanged, and `IntegerPvpFieldsDecodeAsIntegersNotBitPatterns` pins both
halves — the value *and* that it is not the 5e-324 denormal.

## 8. Real collections contain NaN transmit ephemeris — warned, not hidden

*Found 2026-08-09.* One surveyed Umbra collection has `TxTime`, `TxPos` and
`TxVel` as NaN on 32 of its 4844 vectors — the other 4812 are fine, and the
signal array is fully populated (non-zero energy). sarkit reads the same NaNs,
so this is the file, not our parser.

Those NaNs flow through the midpoint into `positions`, `velocities` and
`pulse_times`, and a backprojection consuming them yields NaN pixels — the
silent-wrong-answer failure mode the deferred paths in item 2 were all written
to avoid.

**Resolved as pass-through with a warn-once**, naming how many pulses in the
batch are affected and saying they will appear as blank image regions. The
alternatives were rejected as premature: throwing would refuse a collection
that is 99.3% good, and dropping the affected vectors would change the pulse
count mid-stream — legal under the rank-agnostic contract, but a real
behaviour change made on the evidence of a single file. Passing them through
loses nothing and makes the condition visible, which is enough to decide
properly if a collection ever turns up bad enough to need dropping.

The warning is once per operator instance rather than per tick, matching
invariant 5's warn-once convention; a collection with scattered bad vectors
would otherwise emit one per batch.

## 9. Validation against public collections

*2026-08-09.* The reader was checked against **231 credential-free public CPHD
files** — 113 Umbra and 111 Capella (CC BY 4.0), plus 7 more below — spanning
CPHD 1.0.1 and 1.1.0, stripmap / spotlight / sliding-spotlight, HH and VV, 23
satellites and collection dates from 2020 to 2026. Every file was parsed by
our `CphdFile` and independently by sarkit 1.10.1 **from the same local
bytes**, so any disagreement is a parser difference rather than a data
difference.

**Both pinned demo collects are included and both pass** — Atlanta (mode 1,
28.59 GiB) and Houston (mode 2, 110.31 GiB). They were missed by the first
sweep, whose diversity sampling took the smallest files per satellite-year and
so excluded the two largest, which happen to be the ones the demos actually
read. Both are `CF8` / FX / `SGN` -1 / no `AmpSF`, both agree with sarkit, and
Houston's 208,753 x 70,875 independently confirms the figure recorded in
[`INPUT_DATA.md`](../../../docs/planning/SAR/INPUT_DATA.md).

**Two paths no public file exercises were manufactured**, since a gap nothing
in the wild covers is where a parser bug survives longest:

- **Prefixed XML namespace** — see item 5. Output identical to the source
  collection.
- **Two channels.** Every one of the 231 real files is single-channel, so the
  per-channel `SignalArrayByteOffset` / `PVPArrayByteOffset` arithmetic and the
  operator's `channel` parameter had never been exercised against a file that
  could tell right from wrong. A real collection was rewritten with a second
  declared channel whose blocks follow the first, and — the part that makes it
  a test rather than a tautology — channel 1's first sample was patched to a
  distinct value. Reading channel 0 returns the original sample and channel 1
  returns the patched one, so the index genuinely selects. Without that patch
  a reader ignoring the index entirely would have looked correct.

Only the multi-GB signal blocks make these files large, and the parser needs
only the header, XML and PVP. Those byte ranges were fetched into sparse files
padded to the true length, so size and bounds checks behave exactly as on a
full download: 229 collections totalling ~117 GB nominal cost 12 GB on disk.
Files under 400 MB were taken whole so the signal path could be checked too.

**Result: no parser defects except item 7.** Across all 229 files, our reader
agreed with sarkit on CPHD version, signal format, domain, `SGN`, channel
count, per-channel vector and sample counts, and every `F8` PVP field, to
within 1e-9 relative. The only disagreement anywhere was the integer `SIGNAL`
field. On the 18 fully-downloaded CF8 collections the decoded samples are
**bit-exact** against sarkit.

What the survey says about coverage, which is the more consequential finding:

| | Capella (111) | Umbra (118) |
| --- | --- | --- |
| Signal format | `CI4`, all 111 | `CF8` 116, `CI2` 2 |
| `AmpSF` present | all 111 | 2 of 118 |
| `SGN` | `+1` | `-1` |
| Accepted by the mode 1 operator | **0** | **115** |

Two things follow. **Umbra open data is directly usable now** — 115 collections
need no mode 3 work at all, which means the reader can be demonstrated on real
public data rather than only on simulator output. **Capella is entirely gated
behind mode 3**, and not marginally: every file needs both `CI4` decoding and
`AmpSF`, so neither alone unlocks any of it.

`SGN` also splits cleanly by vendor, `-1` for Umbra and `+1` for Capella. The
reader reports it correctly in both cases; it is flagged because our simulator
only ever emits `-1`, so the `+1` path has never been exercised end to end
through image formation, and getting it wrong mirrors the image rather than
failing loudly.

**Sources ruled out**, so the next person does not re-search them: **ICEYE
Open Data** is credential-free and CC BY 4.0 but publishes only SLC and GRD
(COG / HDF5) — no phase history. **AFRL SDMS** (GOTCHA, GMTI) needs
registration, so it fails the credential-free bar, and its products are not
CPHD. **NGA six-library** is the reference C++ implementation and ships
schemas but no sample data; sarkit already gives us both an independent reader
and the official 1.0.1 / 1.1.0 XSDs. Umbra and Capella are, as far as this
search found, the whole of the credential-free CPHD world.

Deliberately not in CI: it needs several GB of network fetch. The probe,
survey and variant-builder scripts are throwaway and were not committed;
re-running the exercise means rewriting them, which is the right trade for a
check this infrequent.

## 10. No Python bindings

Every FMCW operator has a `python/` submodule; this one does not yet. Nothing
in the mode 1 path needs it, so it is left for the first Python consumer.
