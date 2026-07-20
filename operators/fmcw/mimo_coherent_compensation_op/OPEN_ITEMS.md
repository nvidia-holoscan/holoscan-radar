# MIMOCoherentCompensationOp — Open Items

Deferred work and **team-visible imposed decisions** for this operator.
Items in §A record choices the original contract left open but the current
implementation has pinned — revisit via an explicit `OPERATOR_API.md` contract
change, not by silent drift. Remove items when they land (note the commit in
the PR).

---

## A. Imposed decisions (contract was underspecified)

These are **not deferred** — they are what the landed operator does today.
`OPERATOR_API.md` was updated where noted; anything still ambiguous lives
here.

### A.1 `multiplexing_scheme` is required at runtime

Holoscan `Parameter` defaults must be a concrete value, so the build-time
default is an internal `MultiplexingScheme::kUnspecified` sentinel. `compute()`
rejects it with a clear error. There is no implicit TDM fallback.

### A.2 TDM inter-TX phase model

**Underspecified in contract:** exact formula relating `waveform.chirp_repetition_interval`,
`num_tx`, and the per-Doppler-bin correction.

**Pinned implementation:**

- `waveform.chirp_repetition_interval` is the **per-TX effective PRI** (TDM:
  `M ×` the raw on-wire inter-chirp spacing). See `FMCWWaveformDescriptor` notes.
- Inter-TX time offset for TX index `m` is `m × (PRI / M)` (i.e.
  `m × t_inter_chirp` with `t_inter_chirp = PRI / num_tx`).
- For each center-zero Doppler bin `d`, the correction factor is
  `exp(−j·2π·f_d·m·t_inter_chirp)` where `f_d` is the aliased Doppler frequency
  (Hz) of bin `d` under the input's `num_doppler` and `PRF = 1/PRI`.
- Applied as a **complex multiply** on the existing RD cell; correction
  factors are unit magnitude. Assumes misalignment is purely from inter-TX
  timing — not from per-chirp TX codes on the RD map (those are out of scope
  for TDM).
- Assumes uniform per-TX slow-time spacing; non-uniform schedules are
  rejected upstream by `MIMODeinterleaveOp` / `TXSchedule`.
- Output shape matches input: `num_doppler` is unchanged.

### A.3 DDM “code removal” is a band gather, not correlation

**Underspecified in contract:** “demux + per-chirp code removal + alignment”
could mean conjugate correlation in the Doppler domain.

**Pinned implementation (linear-phase slice):** for accepted codes, demux is a
**pure Doppler-bin gather** driven by a precomputed table. No multiply by
`phase_codes` on the RD data. “Removal” means: offsets define which input bins
belong to each TX band, and the gather reindexes them onto a common reduced,
center-zero Doppler axis. This is exact for standard linear-phase DDM where
codes only shift bands; it is **not** the algorithm for arbitrary codes (§B.1).

### A.4 DDM imposed band layout

Recorded in `OPERATOR_API.md` / `DDMCodeDescriptor` notes; details:

| Rule | Implementation |
| --- | --- |
| Equal bands | `num_doppler % num_tx == 0`; else reject. Each band width = `num_doppler / num_tx`. |
| Guard | `num_doppler_per_tx = band_width − guard_bands`. The **top `guard_bands` fft-order bins** of each band are discarded (inter-band guard). |
| Offset units | `doppler_offsets[m]` are fractions of PRF in `[0, 1)`; folded with `−= floor(o)`. Converted to fft-order start bin `round(o · num_doppler) mod D` per tick. |
| Output axis | Center-zero over width `W = num_doppler_per_tx`; demux table encodes fftshift bookkeeping so every TX lane shares the same reduced axis. |
| `num_tx` source | From `ddm_code.num_tx`, not from an input axis (DDM input has no TX dimension). |

### A.5 DDM `phase_codes` acceptance criteria

Only when `doppler_offsets` is absent:

- `phase_codes` length must be a multiple of `num_tx`; interpreted as
  `(num_tx, num_chirps)` row-major.
- **Mutually exclusive** with `doppler_offsets` (`DDMCodeDescriptor::validate`
  rejects both).
- Requires `num_chirps ≥ 2` per TX to infer a constant per-chirp phase step.
- Linear-phase test: `arg(code[k+1]·conj(code[k]))` must match the first pair
  within `1e-3` rad (wrapped) for all chirp pairs; else reject as arbitrary.
- Accepted codes are **reduced to `doppler_offsets`** internally (`arg(step) /
  2π`, folded to `[0, 1)`); the gather path is identical to supplying offsets
  directly.

### A.6 Doppler axis convention (both schemes)

Input `num_doppler` must match `FMCWDopplerProcessorOp` with
`center_zero_doppler = true` (module default). TDM correction and DDM demux
table both assume center-zero indexing on the input axis. No `[live]` toggle on
this operator to select another ordering.

### A.7 `waveform` on DDM (linear-phase slice)

Required and validated (build-time or `[live]` sticky override), but **does
not affect** the linear-phase gather demux. Kept for TDM parity and the
deferred arbitrary-code path (§B.1). Overriding `waveform` under DDM does not
change output.

### A.8 `ddm_code` on TDM

Unused on the TDM path — not validated when `multiplexing_scheme = tdm`. Callers
may leave `ddm_code` default-empty for TDM instances.

### A.9 Doppler / velocity ambiguity (÷M)

**Contract mentions** reduced unambiguous velocity under TDM and DDM but does
not specify disambiguation. **Not implemented** — no staggered-PRI, band-overlap,
or explicit velocity folding. Downstream consumers see the aliased per-TX
velocity axis from `FMCWDopplerProcessorOp` with per-TX effective PRI.

---

## B. Deferred follow-on work

### B.1 DDM arbitrary / non-linear `phase_codes`

Arbitrary / non-linear codes are **rejected** (§A.5). Their demux is a different
algorithm — post-Doppler circular convolution / conjugate correlation of the
code spectrum, not band extraction — and truly non-linear codes do not form
clean Doppler bands. Needs a separate kernel, tests, and likely uses `waveform`
and the full `phase_codes` matrix.

### B.2 Unequal DDM band widths

Current slice assumes equal `num_doppler / num_tx` bands. A future slice could
carry explicit per-band widths or offsets that do not divide `num_doppler`
evenly.

### B.3 Alternate guard placement

Guard discard is **top fft-order bins per band** (§A.4). Revisit if hardware /
waveform docs require edge guards, center guards, or per-offset guard widths.

### B.4 Per-tick output allocation (module-wide)

Not operator-specific. See [`operators/OPEN_ITEMS.md`](../../OPEN_ITEMS.md) §1.

**Diagnostic (2026-07-16, TDM `xl`, RTX PRO 6000 Blackwell):** after fixing a
kernel launch bug (one block per element), nsys shows the TDM kernel ~6.7 ms
vs harness tick median ~14.5 ms — remainder is output `cudaMalloc`/`cudaFree`,
not kernel bandwidth. Defer ping-pong to the module-wide decision.
