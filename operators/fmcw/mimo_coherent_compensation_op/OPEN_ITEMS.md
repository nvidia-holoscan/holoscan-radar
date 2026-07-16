# MIMOCoherentCompensationOp — Open Items

Deferred, agreed-upon work for this operator. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. DDM arbitrary / non-linear phase codes (deferred)

DDM mode is implemented for **standard linear-phase DDM** only: the
`doppler_offsets` compact form, and `phase_codes` that are linear-phase (a
constant per-chirp phase increment per TX, reduced to offsets internally). The
demux is a pure post-Doppler band gather.

Arbitrary / non-linear `phase_codes` are **rejected** with a clear error. Their
demux is a genuinely different algorithm — post-Doppler it is a circular
convolution (conjugate correlation of the code's spectrum), not a band
extraction — and truly non-linear codes do not form clean Doppler bands at all.
Deferred as a follow-on slice; needs a separate kernel and its own tests.

## 2. DDM imposed layout requirement (team-visible)

The linear-phase DDM slice **imposes** a layout not fully pinned down in the
original contract (now recorded in `OPERATOR_API.md`, DDMCodeDescriptor notes):

- `num_doppler` must be divisible by `num_tx` (equal sub-bands). Non-divisible
  Doppler lengths are rejected. A future slice could support unequal bands /
  explicit per-band widths.
- `num_doppler_per_tx = num_doppler / num_tx − guard_bands`; the top
  `guard_bands` fft-order bins of each band are discarded (guard between
  adjacent bands). This is one reasonable reading of "band width follows the
  offset/guard layout"; revisit if a different guard placement is needed.
- `doppler_offsets` are fractions of PRF in `[0, 1)` (converted to bins per
  tick), chosen over bins so the static descriptor need not know the runtime
  `num_doppler`.

## 3. Doppler axis convention

Both schemes assume the input `num_doppler` axis matches
`FMCWDopplerProcessorOp` with `center_zero_doppler = true` (the module
default) — TDM for the per-bin correction, DDM for the band gather / fftshift
bookkeeping. There is no `[live]` override on this operator to select another
ordering; mismatched upstream Doppler fftshift would require a contract
extension or an upstream convention change.

## 4. DDM waveform parameter is inert

For standard linear-phase DDM the demux is a pure Doppler-bin gather driven by
`ddm_code`; it does **not** use `waveform`. The parameter is kept required (and
its `[live]` override accepted + sticky) for contract parity with TDM and the
deferred arbitrary-code path, but overriding it does not change the DDM output.

## 5. Per-tick output allocation (module-wide — defer)

**Not operator-specific.** Emitted `rd_maps` use `make_device_tensor` every
`compute()` tick — the same synchronous `cudaMalloc`/`cudaFree` path as every
other marshaling / signal-processing operator. See
[`operators/OPEN_ITEMS.md`](../../OPEN_ITEMS.md) §1.

**Diagnostic (2026-07-16, `xl`, RTX PRO 6000 Blackwell):** after fixing a
kernel launch bug (one block per element, 256 redundant threads), nsys shows
the TDM correction kernel at ~6.7 ms (~1.28 TB/s) while the Holoscan-timed tick
median is ~14.5 ms — the remainder is output alloc/free, not kernel bandwidth.
No ping-pong or pooling work is planned for this operator slice; pick up with
the module-wide decision in §1 when latency budgets require it.
