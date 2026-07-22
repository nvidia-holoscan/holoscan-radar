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
  within `1e-3` rad (wrapped) for all chirp pairs; codes passing this test are
  routed to the **gather path** (§A.3).
- Codes failing the linear-phase test are routed to the **correlate path**
  (§A.10) — no longer rejected.
- Accepted linear codes are **reduced to `doppler_offsets`** internally
  (`arg(step) / 2π`, folded to `[0, 1)`); the gather path is identical to
  supplying offsets directly.

### A.6 Doppler axis convention (both schemes)

Input `num_doppler` must match `FMCWDopplerProcessorOp` with
`center_zero_doppler = true` (module default). TDM correction and DDM demux
table both assume center-zero indexing on the input axis. No `[live]` toggle on
this operator to select another ordering.

### A.7 `waveform` on DDM (linear-phase slice)

Required and validated (build-time or `[live]` sticky override), but **does
not affect** the linear-phase gather demux or the non-linear correlate demux.
Kept for TDM parity. Overriding `waveform` under DDM does not change output.

### A.8 `ddm_code` on TDM

Unused on the TDM path — not validated when `multiplexing_scheme = tdm`. Callers
may leave `ddm_code` default-empty for TDM instances.

### A.9 Doppler / velocity ambiguity (÷M)

**Contract mentions** reduced unambiguous velocity under TDM and DDM but does
not specify disambiguation. **Not implemented** — no staggered-PRI, band-overlap,
or explicit velocity folding. Downstream consumers see the aliased per-TX
velocity axis from `FMCWDopplerProcessorOp` with per-TX effective PRI.

### A.10 DDM non-linear `phase_codes` correlate path

**Imposed for this slice.** Non-linear codes (those failing the 1e-3 rad
linear-phase test in §A.5) are now handled by a **conjugate-correlation** demux
instead of being rejected.

| Rule | Implementation |
| --- | --- |
| Detection | `ddm_classify_codes()` applies the same 1e-3 rad test; returns `kLinear` (gather path) or `kNonLinear` (correlate path). |
| Output shape | `(..., M_tx, N_rx, num_doppler, num_range)` — **full** Doppler axis preserved (no band reduction). |
| `guard_bands` | Must be **0**; correlate path rejects nonzero `guard_bands` with a clear error. |
| `num_chirps` match | `phase_codes.size() / num_tx` must equal input `num_doppler` at runtime; mismatch is rejected. |
| Algorithm | FFT circular correlation, range-tiled (~32 MiB persistent workspaces): pack (ifftshift to fft-order) → cuFFT fwd → per-TX (`Ŷ * W * 1/D` + cuFFT inv) → scatter (fftshift to center-zero). `W_m[p] = conj(c_m[(-p) mod D])`. Equivalent to `out[d] = (1/D) * sum_k Y[k] * conj(C_m[(k-d) mod D])` in fft-order. Input/output Doppler axes match `FMCWDopplerProcessorOp` center-zero. |
| Waveform | Required and validated (contract parity) but **unused** on the correlate path — carried for the same reason as §A.7. |
| Linear codes | Unchanged: detected automatically, routed to the existing gather path via `doppler_offsets`. |

---

## B. Deferred follow-on work

### ~~B.1 DDM arbitrary / non-linear `phase_codes`~~ — **LANDED**

Implemented as the correlate path (§A.10). Non-linear codes are no longer
rejected; they are demuxed via circular conjugate-correlation of the code
spectrum.

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

**Follow-up (2026-07-19, DDM-NL `xl`):** tiled cuFFT correlate GPU work
~8.0 ms vs harness median ~15.5 ms; per-tick `cudaMalloc` (~2.2–2.4 ms) +
`cudaFree` (~8–10.5 ms) on the ~4.3 GB full-Doppler output account for the
gap. Same §1 deferral — annotate only; no ping-pong landed here.

### B.5 Sparse DDM detect-before-demux (folded CFAR + per-detection recovery)

**Out of wave 1.** An alternate sparse MIMO–DDM recipe seen in reference
CUDA/MatX pipelines: CFAR on a **pre-demux** power map (often after Rx NCI
plus a Doppler-alias / “fold” average that shrinks the CFAR plane to
`num_range × (num_doppler / fold)`), then **per-detection** DDM recovery —
pick an unfolded Doppler candidate, gather each TX’s bin at
`(unfolded + doppler_offset[tx])`, apply the code-phase weight, and form the
virtual-channel snapshot for Bartlett / `BeamscanEstimatorOp`.

This is **not** a mode of `MIMOCoherentCompensationOp`. Wave 1 keeps
full-cube demux → optional clutter → `VirtualArrayAssemblyOp` → CFAR →
`SnapshotGatherOp`. The alternate path would mainly touch a DDM-aware
snapshot bridge (and optionally a fold-NCI preprocessor); COMP stays the
whole-cube demux.

**Why deferred:** trades detection performance for demux cost. Fold-then-CFAR
dilutes SNR when empty fold lanes are averaged in, is a different detector
than demux-first + `nci_axes` over virtual channels (worse Pd @ fixed Pfa in
the usual sense), and couples to extended Doppler unfold / disambiguation
already out of scope in §A.9. Worth revisiting only for sparse-path latency /
reference-parity workloads — see also
[`WORKFLOWS.md`](../../../docs/planning/WORKFLOWS.md) Notes.
