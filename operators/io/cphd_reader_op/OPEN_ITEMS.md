# CPHDReaderOp — open items

Items this operator carries. The contract surface is complete **except for the
boundary announcement's encoding (item 11), which is mid-implementation and
does not match `OPERATOR_API.md` yet** — it cannot until the operator that
defines the shared encoding exists. Item 2, the entry that recorded a landed
slice narrower than `OPERATOR_API.md`, is closed. What otherwise remains is
conventions the contract permits several readings of, costs measured and
consciously not paid, and paths no available collection exercises. None block
the mode 1 demo.

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

The independent former in `tools/sar_image` no longer uses `TxPos` as a
stand-in APC. It uses the two-leg path
`½(‖x−Tx‖ + ‖x−Rcv‖ − ‖SRP−Tx‖ − ‖SRP−Rcv‖)`, which is the monostatic
formula when the two coincide (simulator fixtures) and the physically
correct delay when they do not. Comparing an Atlanta operator dump to a
cupy formation is therefore **midpoint-as-one-APC versus two-leg range**,
not Tx-only versus midpoint.

## 2. The paths deferred to mode 3 — closed, pulled forward to clear the checklist

*Closed 2026-08-09.* All four landed together, ahead of the mode 3 slice
[`SIGPROC_PLAN.md`](../../../docs/planning/SAR/SIGPROC_PLAN.md) scheduled them
into. The minimum-per-demo rule says implement what the next demo forces; the
[new-operator checklist](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md) says
a vertical slice never clears the gate. Both are right and they point opposite
ways here, so the reader was carried to full contract rather than being called
done at the mode 1 subset. What each now does:

- **Multi-file `path`.** A single file, a directory (lexicographic), or an
  explicit ordered list. Each file is its own collection: `pulse_range` applies
  per file, the frame is validated per file, the `coordinate_frame` descriptor
  is re-emitted at each collection's first tick, and the seam is announced as
  `collection_boundary` with `end_of_stream` reserved for the last
  unless `loop` is true (then the last seam wraps and is also
  `collection_boundary`). Every file is opened and checked at
  `start()`, so an unreadable third file fails before the first
  pulse rather than after two collections have been formed.
- **`CI4` and `CI2` sample decoding.** Both decode. `CF8` keeps its
  no-intermediate-copy path (item 6); the integer formats convert per sample,
  which is why `CI2` is the slowest per byte in the harness despite moving the
  fewest.
- **`AmpSF`.** Applied per vector. The scale factors live in the parameter
  block, so `read_signal_into` **requires** the `PvpBlock` to be handed to it
  whenever the file declares `AmpSF` and throws if it is missing or is the
  wrong batch. Making the caller supply it is what stops a silently unscaled
  read, which is the failure the old outright refusal was protecting against.
- **TOA-domain collections.** Read. `SC0`/`SCSS` define the sample axis in both
  domains — Hz and Hz/sample under `FX`, seconds and seconds/sample under
  `TOA` — so the arithmetic never differed and only the unit the contract
  ascribes to the port does. Untested against a real file: no public collection
  found in item 9's survey is TOA, and the simulator emits FX.

This closes the reader's contract surface. It does **not** close item 9's
coverage gap by itself — see the amended table there for what the survey says
now.

## 3. Frequency-axis constancy is enforced per batch, not per collection

`SC0` and `SCSS` are per-vector fields, but `sample_axis` is one shared row, so
the operator checks that they agree across the pulses of each batch and throws
otherwise. A collection that stepped its start frequency **between** batches at
exactly the batch boundary would slip through. Cheap to close by checking the
whole channel at `start()`; not done because it costs a full-file PVP pass at
open, which on a 28 GiB collection is not free, and no known collection
behaves that way. Revisit if one does.

*Updated 2026-08-26.* The check is a **relative ramp bound of 1e-5**, not
bit-identity and not last-place jitter. Public Umbra SNAPSHOT CPHDs (the six
mode-5 files) ramp `SC0` by ~0.02 Hz/pulse (~2e-12 relative at X-band) and
`SCSS` by ~4e-6 Hz/pulse (~1e-10 relative), monotonically, unique on every
vector of the first 100. That is a recorded-grid residual, not a stepped
sample axis. The previous 1e-12 bound refused Panama on pulse 1 of the first
batch. 1e-5 still refuses a genuine step (100 kHz on a 10 GHz `SC0`) and
leaves ~3× margin on a 5 s SNAPSHOT even if the whole collect is one batch
(~3e-6 relative on `SCSS`). The error names the field and prints both values.

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

## 6. Remaining hot-path cost — now measured, and it was not what was assumed

The tick's five `cudaMemcpy` calls **were** synchronous on the default stream
rather than async on the operator's own, and each of the five output tensors
is still a fresh device allocation. This entry previously named the
synchronous copies as the cost and deferred judgement to the performance
harness. The harness (`performance/cpp/perf_cphd_reader.cpp`) exists, and
attributing the tick puts the ranking the other way round at small batches:

| pulses x 512 | pvp read | signal read | 5x `cudaMalloc` | 5x H2D |
| --- | --- | --- | --- | --- |
| 4 | 0.006 | 0.013 | **0.090** | 0.028 |
| 32 | 0.004 | 0.034 | **0.089** | 0.047 |
| 256 | 0.011 | **0.243** | 0.085 | 0.193 |

(ms, median of 200 reps, page-cache-warm, RTX PRO 6000.)

Three readings, none of which the original entry would have predicted:

- **Allocation is a flat ~0.09 ms per tick, independent of batch size**, so it
  is half the tick at four pulses and 16% at 256. That is the per-tick-alloc
  red flag the checklist says to look for, and it is the reason the harness
  reports 0.3 GB/s at `xs` against 2.9 GB/s at `lg` — the small batches are
  measuring the allocator, not the reader.
- **The copies run at ~5.5 GB/s**, which is pageable-host-memory H2D speed
  almost exactly. Pinned staging buffers are worth roughly 3-4x on that line;
  going async on the operator's stream is worth nothing on its own, since
  nothing else is contending yet.
- **The read is not the bottleneck it was assumed to be** at these sizes —
  4.1 GB/s with the file in page cache. On a cold 28 GiB Atlanta pull the disk
  will dominate all three columns and none of this will matter.

*Paid 2026-08-24, once the mode 1 chain ran end to end.* Atlanta (11500 ×
37268 CF8, `CPHDReaderOp<double>`) spent **12.4 s in the reader** — not the
3.19 GiB of CF8, which isolated `read_signal_into` into a reused buffer does
in 1.71 s. Two host costs each tick: (1) `std::vector<complex<double>>(n)`
and `vector<char>(n)` **value-initialize ~90 MiB of fresh pages**, and (2)
blocking `cudaMemcpy` on the default stream. The operator now keeps
**unzeroed pinned** phase-history and decode scratch for its lifetime and
H2Ds on a GXF-managed non-blocking stream (`allocate_cuda_stream`; a
homemade `cudaStreamCreate` stream is rejected by `set_cuda_stream` and
logged an error on every port every tick), then `cudaStreamSynchronize`s
at the end of `compute()` so the next tick cannot overwrite history still
in flight. Ping-pong (overlapping decode of tick N+1 with H2D of tick N) was
measured and declined — extra complexity for ~0.8 s. `CphdFile` reuses an
uninitialized `new char[n]` scratch on the widen path the same way;
`vector<char>(n)` is gone there too.

*Paid 2026-08-26.* Mode-5 Ohio spent **8.2 s in `CPHDReaderOp<double>`**
after the pinned-staging fix, against 3.9 s of backprojection. The files
are CF8; the double host path widened every sample in a scalar loop
(~6 s) and H2D'd 16 bytes/sample. CF8 now reads stored bytes into the
existing pinned scratch, H2Ds 8 bytes/sample, and byte-swaps / widens /
applies AmpSF on the device (`cphd_cf8_decode.cu`). Float CF8 decodes
in-place on the output tensor; double H2Ds into a reused device CF8
buffer. Integer formats still use `read_signal_into`. `CphdFile`'s host
widen remains for tests and tools. No contract change. The HSDK 5
operator `compute()` had fallen back to host `read_signal_into` (Houston
1.0 m: ~0.16 s/tick decode vs ~0.12 s collect); it uses this GPU path
again.

*Paid 2026-08-28.* Houston 1.0 m / 0.5 m ingest is **one outstanding
sequential read**, not H2D and not ifstream buffering. Isolated cold
`pread` of a 131 MiB tick is ~1.9 GB/s on the SN850X whether the file
sits on overlay `/tmp` or ext4; cache-hot is ~6.3 GB/s. PVP seeks,
`POSIX_FADV_SEQUENTIAL`, and a 4 MiB `filebuf` do not move the cold
number. Four concurrent `pread`s of the same span reach ~4.4 GB/s (eight
~4.8). `CphdFile::read_at` now uses `pread` and splits spans ≥ 1 MiB
across four threads. Cold operator confirm: 1.0 m 16.35 s → 8.24 s
(signal 6.78 s), 0.5 m 33.92 s → 17.41 s (signal 14.56 s); both under
the 23.5 s dwell. EOS logs split the tick into pvp / signal / decode /
geo / sync.

**Still unpaid:** per-tick `make_device_tensor` (~0.09 ms, the small-batch
harness finding above). A pooled allocator is the remaining win; it is a
module-wide `make_device_tensor` question, not a reader-local one.
Two-slot ping-pong of `phase_history` was tried 2026-08-26 and reverted:
it removed RC's 0.27 s `cudaFree` but Ohio BP dropped ~8.7e11 → ~7.7e11
(same after `range_profiles` ping-pong was already gone). The device
sync from `cudaFree` was serializing decode off BP's GPU time.

*Two earlier costs closed, both measured rather than assumed:*

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
| Accepted at the mode 1 subset | 0 | 115 |
| Accepted now (item 2 closed) | **111** | **118** |

*Table amended 2026-08-09*, when item 2 closed. The survey's sharpest finding
was that **Capella was entirely gated**, and not marginally — every one of its
111 files needs both `CI4` decoding and `AmpSF`, so neither alone unlocked any
of it. That is what made deferring both to mode 3 expensive: it left half the
credential-free CPHD world unreadable while the reader was being called done.
All 229 now read.

`CI4` + `AmpSF` was re-verified numerically after the fact rather than assumed
from the fixtures: Capella collections were decoded three ways from the same
local bytes — our reader, `sarpy` (which applies `AmpSF` itself), and a manual
decode applying scale factors read by `sarkit` — and all three agree. The
generated `CI4` and `CI2` fixtures round-trip against the `CF8` collection they
were quantized from to within an LSB, which pins the decode; the three-way
check is what pins the *scaling*, since a fixture we quantized ourselves would
have agreed with our own convention either way.

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

## 11. The boundary value is a string, not the contract's `uint8` — mid-implementation

**This one is a live deviation from `OPERATOR_API.md`, not a reading of it.**
The D22 table specifies the value as `uint8`, an enum
`{end_of_stream, collection_boundary}` "named in `radar_common` so C++ and
Python agree, on the same `uint8_t` encoding the `control` port uses for its
four values". The reader currently writes the two names as a `std::string`.

`FormationControl` (the four `control` values) now exists in
`radar_common/descriptors/enums.hpp` — it landed with `BackprojectionOp`.
`StreamBoundary` as a `uint8` enum still does not. `ApertureManagerOp`
landed reading the same interim strings as this reader
(`radar_common/sdk/stream_boundary.hpp`). Switching both sites to
`uint8` together is still the close: add `StreamBoundary` beside
`FormationControl`, switch this operator's `metadata()` write, and
update `test_cphd_reader.cpp` plus the manager together.

This does **not** close the item. The wire format is still the placeholder
string encoding, not the contract's `uint8`.

Note this is separate from **R10**, the open team question about the key's
*namespace*. That is why the key sits at the SAR tier
(`holoscan.radar.sar.stream_boundary`) per
[`SIGPROC_PLAN.md`](../../../docs/planning/SAR/SIGPROC_PLAN.md). R10 governs
the prefix; this item governs the value.
