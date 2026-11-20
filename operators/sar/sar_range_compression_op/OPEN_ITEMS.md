# SARRangeCompressionOp — open items

Items this operator carries. **The contract surface is complete** as of
2026-08-24: all three output ports, all five `[live]` parameters, `nfft`
zero-padding in the middle of the rotated row, the leading `ifftshift`
(IFFT DC = stored sample `N/2`), the centred range axis, and both
descriptor-derived outputs are implemented, and nothing in
[`OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md) →
*SARRangeCompressionOp* is staged or throws.

One item here is a genuine team question (item 1, **I5**, migrated from
[`OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md) → *Open items*
now that this directory exists). The rest are conventions the contract leaves
open and measured costs consciously not paid.

## 1. I5 — the normalization default is implemented, not settled

`fft_norm` defaults to `backward`, exactly as the contract states, and because
this stage's transform is the *inverse* one that means **this operator applies
the 1/N**. Absolute output amplitude therefore reaches SICD's radiometric
fields, which is why the default is a radiometric choice rather than a spelling
detail. **Implementing the stated default is not resolving I5**: the open
question is whether `backward` is the right default measured against the SICD
convention and against what an independent reader expects, and nobody has
answered it.

What the code adds to the question is that the scaling is now pinned and
testable: `NormalizationLandsOnTheInverseTransform` asserts the absolute peak
against the analytic prediction for all three values, so a decision to change
the default is a one-line change plus a re-baselined expectation, not an
investigation. Changing it later is a contract change and needs the usual gate.

## 2. `bandwidth / (num_samples - 1)`, and why it is not `bandwidth / num_samples`

The contract derives `range_bins` from "bandwidth, `nfft` and
`propagation_speed`" without fixing the frequency spacing the bandwidth implies.
Implemented as `df = bandwidth / (num_samples - 1)`: the bandwidth is the span
from the first sample's frequency to the last one's, so N samples divide it into
N−1 steps. This is what CPHD's `Global/FxBand` against the per-vector `SC0` /
`SCSS` describes, and what the in-repo simulator
(`tools/sar_sim/sar_sim/signal.py` → `frequency_axis`) generates, so the two
agree by construction rather than by luck.

The alternative reading differs by one part in N. That is a range **scale**
error: it vanishes at the reference range and grows toward the swath edges, so a
single target at scene centre cannot see it, and neither can any test that only
checks a peak's bin index. `PeakMapsToDifferentialRangeViaRangeBins` therefore
places its target 96 bins off centre and asserts to half a bin.

Worth a second look when the first real collection is formed, since a file whose
`FxBand` is defined some other way would surface here.

## 3. The centring shift is a second pass, and it is cached scratch

`fftshift1D` is a gather, so the transform cannot write through it in place: the
transform lands in a scratch buffer and the shift copies out of it. The scratch
is cached by shape in the operator's `ConfigCache` rather than allocated per
tick.

It is **plain device memory, not stream-ordered** (`MATX_DEVICE_MEMORY`, not
`MATX_ASYNC_DEVICE_MEMORY`), and that is not a style choice. The cache outlives
every tick and is freed when the operator is destroyed, by which point the
Holoscan stream the allocation was made on is gone; a stream-ordered allocation
frees itself on that dead stream and segfaults during teardown. It cost an hour
to find, because the symptom was a crash after a clean run with correct output.
Every cached device artifact in this module has the same constraint.

Folding the shift into the transform's output indexing would remove the pass
entirely and is the obvious optimization; it is not taken because MatX's `ifft`
writes its own output and the alternative is a hand-written kernel replacing a
library call. Item 4's measurements say it is not currently the bottleneck.

## 4. A ~0.2 ms step appears when the output crosses ~2 MiB, and it is not compute

`performance/cpp/perf_sar_range_compression.cpp` on an RTX PRO 6000 Blackwell,
median steady-state per tick:

| Scenario | Output | median |
| --- | --- | --- |
| `md` 64×2048 | 1 MiB | 0.069 ms |
| `md` pad2 (same input, 2× transform) | 2 MiB | 0.261 ms |
| `md` pad4 (same input, 4× transform) | 4 MiB | 0.268 ms |
| `lg` 128×4096 | 4 MiB | 0.329 ms |
| `lg` ×16 channels | 64 MiB | 0.399 ms |

**Read the middle three rows together.** Going from 1 MiB to 2 MiB of output
doubles the transform work and costs 0.19 ms; going from 2 MiB to 4 MiB doubles
it *again* and costs 0.007 ms. A cost that appears once at a size boundary and
then stops scaling is not arithmetic — it is the per-tick `make_device_tensor`
(`cudaMalloc` / `cudaFree`) crossing out of the driver's small-allocation cache.
Above the step the operator behaves well: 64 MiB of output in 0.399 ms is
~670 GB/s of effective traffic.

This is the **same finding `CPHDReaderOp` made independently** (its
`OPEN_ITEMS.md` item 6: a flat ~0.09 ms of per-tick `cudaMalloc`), from a
different direction, and two operators agreeing on it is the argument for
pooled allocation in `make_device_tensor` rather than a fix here. That is a
module-wide change and stays queued behind `BackprojectionOp`, which will be
the third data point and the one with a real frame budget to serialize against.

Not paid here, deliberately: the operator allocates its output per tick like
every other operator in the module, and diverging from that locally would hide
the shared problem rather than fix it. Two-slot ping-pong of
`range_profiles` *and* of the reader's `phase_history` was tried 2026-08-26
and reverted. RC output reuse raced BP's in-flight kernel; reader reuse
removed the `cudaFree` device-sync that had been serializing decode off
BP's GPU time. Independent images Ohio BP dropped ~8.7e11 → ~7.5–7.7e11 px·pls/s
either way. A pooled allocator or a completion-event N-buffer is the
remaining path; two slots are not enough when GPU lag ≫ one host tick.

## 5. A prime `nfft` costs 170 ms the first time on a cold JIT cache

The `nfft` override is the only one that rebuilds anything expensive, so
`perf_sar_range_compression` measures four transitions separately, one sample
per graph session across several sessions, with the first session kept apart
(the cuFFT plan cache is process-global, so only session 0 pays a cold build).
On an RTX PRO 6000 Blackwell at `md` (64×2048), against a 0.067 ms steady tick:

| Transition | first session | later sessions (median) |
| --- | --- | --- |
| `2048 → 4096` (unseen, radix) | 0.83 ms | 0.36 ms |
| `2048 → 8192` (unseen, radix) | 0.84 ms | 0.37 ms |
| `2048 → 4099+` (unseen **prime**, fresh each session) | 5.2 ms | 3.5 ms |
| `4096 → 2048` (a length this instance already ran) | 0.088 ms | 0.088 ms |

**The 2x first-session figure was re-measured**, and it replaces an earlier
0.36 ms that was not a miss at all. The cuFFT plan cache is process-global, and
the harness ran a zero-padded scenario at 2 × `md` — the same length *and* the
same batch the 2x probe moves to — before the probe, so the "miss" measured a
warm plan. Reordering so the probes precede the padded scenario moved it to
0.83 ms. The 4x row was never affected: nothing else in the matrix runs 8192 at
this batch, and `lg`/`xl` reach those lengths only at their own pulse counts,
which are different plans. The correction changes the conclusion — the first
visit to an unseen length costs about the same at 2x and 4x, so the cost is a
plan build rather than anything that scales with length.

Two of those rows matter.

**The cached return costs 0.088 ms against a 0.073 ms steady tick**, so the
warn-once-and-cache rule delivers what it claims: re-entering a length the
instance has already run rebuilds nothing measurable. That is the only direct
check of the caching claim anywhere, unit tests included — they assert the
*output* is right, not that the second visit was cheap.

**A prime length is ~10× a power-of-two one, and on a cold CUDA JIT cache it is
~170 ms.** cuFFT has no radix decomposition for a prime, so it falls to
Bluestein, whose kernels are JIT-compiled for the arch on first use. Measured
directly: with `CUDA_CACHE_PATH` pointing at an empty directory the first prime
plan cost **172 ms**, and the same directory warm cost **7.6 ms** — same binary,
same length, same process shape. A fresh container, a driver update or a new GPU
generation therefore all resurface it.

This is worth knowing precisely because the contract names a large-prime
`num_samples` as the reason `nfft` exists. The guidance that follows is that
`nfft` should be set to a *fast* length rather than merely a sufficient one —
padding 4099 up to 4320 is cheaper than transforming 4099 — and that a caller
who does hit a prime pays it once per length, not per tick (`after_ms` for the
prime row is 0.27 ms, ordinary steady state for that size).

Nothing is fixed here: the operator honours the length it is given, which is
correct. If it should *warn* when handed a slow length, that is a contract
question about the `[live]` warning surface, not a bug.

**The scratch buffer holds two shapes, and that bound is what this table
measures.** The staging buffer used to be cached per shape without limit,
which pinned a full `(batch, pulses, nfft)` buffer for every length an
override ever visited. Bounding it to one entry fixed the memory but cost the
row above: a return to an already-run length reallocated, and went from
0.085 ms to 0.25 ms. Two entries hold the A → B → A pattern an override
usually follows while still bounding the worst case, which is why the return
row reads 0.088 ms rather than 0.25 ms. A genuine sweep across many lengths
evicts and pays a reallocation per length — a per-override cost, not a
per-tick one. Note the slots are a fixed array of `std::optional`, not a
vector: MatX overloads `tensor_t::operator=` to build a lazy assignment
expression, so any container operation that assigns an element leaves the slot
holding its old shape instead of the new tensor.

## 6. Emitting the range axis costs ~18 µs a tick, and blocks the host

`_noaxis` scenarios supply no waveform, so `range_bins` is never emitted;
differenced against their axis-emitting twin at `md` the axis path costs
**0.067 → 0.049 ms**, about 27% of a small tick.

The cost is not the 8 KiB copy. `fill_compressed_range_bins` rebuilds the axis
on the host every tick and then calls `cudaStreamSynchronize` to keep its
staging vector alive across the copy, which blocks the host on every kernel
already queued on that stream. The axis only changes when `nfft`, `waveform` or
`propagation_speed` changes, so caching the device buffer would remove both the
recompute and the sync.

Not done, for one reason worth stating: emitting a cached device tensor means
every tick's consumer receives the *same* buffer, and whether that is safe is a
module-wide question about downstream retention rather than a local
optimization. The FMCW range stage has the identical pattern in
`fill_range_bins`, so a fix belongs to both or neither.

## 7. A window override re-announces; an `nfft` override does not

`range_window` is emitted "whenever the window changes, including once at the
first tick", implemented against the *value* of the effective window: an
override that sets the window to a value equal to the current one is not a
change and does not re-announce. This is the contract read literally — an
override carrying the window already in force leaves nothing for a writer to
relabel — and it is what `resolve_override_param` reports, so the behaviour is
shared with every other `[live]` parameter in the module rather than special
here.

The alternative — re-announcing on every override message whatever its value,
on the grounds that a redundant announcement is cheap while a missed one is a
mislabelled product — would need this operator to bypass the shared helper,
since the helper compares values for all of its callers. Worth revisiting only
alongside a case where a writer actually missed a relabel.

Nothing else re-announces. `nfft` changes the output shape, which downstream
re-infers from the tensor, so there is no descriptor to re-emit.

## 8. `waveform` is the only override that does not warn on rebuild

The costly-rebuild warning fires for `range_window` and `nfft`, which rebuild
coefficients and transform plans. A `waveform` override rebuilds nothing — it is
read straight into the `range_bins` arithmetic each tick — so it is resolved
without the warn-once path. Recorded because the FMCW range stage *does* warn on
a waveform override, and the difference is real rather than an omission: there,
the waveform builds a cached dechirp reference.
