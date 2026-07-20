# MVDREstimatorOp — open items

Status: **NOT checklist-complete — one blocked parameter remains
(`spatial_smoothing`).** Everything else in the contract
(`OPERATOR_API.md` → MVDREstimatorOp) is implemented:

- sample covariance with `is_valid` masking, diagonal loading (`auto` / fixed),
  batched Hermitian inverse, Capon spectrum, dominant DOA;
- `max_sources` **[live]** + `doa_multi` (num_items, max_sources, 2) +
  `source_count` (num_items,) — the top-`max_sources` Capon-spectrum peaks
  (pass 0 = global Capon argmax so `doa_multi[·,0,·]` equals `aoa_estimates`;
  further peaks are strict 2-D local maxima, strongest first), unused rows NaN;
- `eigenvalues` (num_items, num_channels) — the ascending eigenvalue spectrum
  of the raw (pre-loading) sample covariance (batched `matx::eig`, NO_VECTOR).

All four optional outputs (`angle_spectrum`, `doa_multi`, `source_count`,
`eigenvalues`) carry `ConditionType::kNone`, so a consumer that only wants
`aoa_estimates` leaves them unwired without gating scheduling. They are always
emitted (see the eigenvalues perf note below).

## BLOCKED — team question (do not resolve unilaterally)

**`spatial_smoothing` [live] input form.** The contract lists a `[live]
spatial_smoothing` param (forward sub-array averaging to decorrelate coherent
multipath / raise covariance rank), but the input form for smoothing on the
module's *virtual* arrays is an open team decision — one of the
covariance-estimator open items flagged in `OPERATOR_API.md`
§"Covariance estimator open items". It is **not declared**, because declaring or
honoring it would pick an answer to that team question. When decided, add the
param + the sub-array covariance path here (it slots in front of the covariance
kernel; the rest of the pipeline is unchanged).

This is the sole reason the operator is not checklist-complete. The related
"where does the source count come from" question (fixed vs. MDL/AIC vs.
CFAR/cluster seed) does **not** block MVDR: for MVDR `max_sources` is simply the
number of Capon peaks to report (a reporting capacity, not a model order), so it
is implemented directly. That shared question still governs MUSIC/ESPRIT's
`num_sources`.

## Representation notes

- **`diagonal_loading` encoding.** The contract types this param as "float or
  auto". Pending the shared descriptor/YAML story, it is encoded here as a
  `double`: a negative value selects auto (δ scaled to
  `kAutoLoadingFactor · trace(R̂)/C`), a non-negative value is the absolute δ.
  Revisit when Python bindings / YAML converters land so a literal `auto`
  string round-trips.
- **Auto-loading factor.** `kAutoLoadingFactor = 1e-2` is a first cut. It
  guarantees invertibility (including rank-deficient `K < C`) without
  dominating a well-conditioned covariance, but has not been swept against
  resolution / SNR; tune with the analytic simulator.

## Performance follow-ups

Profiled on an RTX PRO 6000 (Blackwell GB202), `perf_mvdr_estimator --full`.
`capon_reduce_kernel` is ~97% of GPU time (nsys `cuda_gpu_kern_sum`); the
covariance kernel and the batched `matx::inv` are ~1.6% / ~1.3%. GPU-metrics
sampling shows the Capon kernel is **latency-bound**, not memory-bound: DRAM
throughput ≈0%, tensor cores unused, SM issue-rate tiny while warps are
resident (they stall). Findings:

- **Staging `R̂⁻¹` in shared memory measured *slower*** (lg 5.3→8.7 ms) and was
  reverted. Every thread reads the same `R̂⁻¹[c,c2]` in lockstep, so the global
  reads already broadcast from L1/L2; staging only added a copy + `__syncthreads`
  and cost occupancy. Do not retry this.
- **Inner-loop ILP (four independent accumulators) landed** — breaks the serial
  reduction dependency chain so FMAs pipeline. Measured: xl 20.0→14.9 ms
  (−26%), lg ~−4%, md flat, no regressions.
- **Remaining ceiling is occupancy + the one-thread-per-beam mapping.** The
  kernel uses ~66–68 registers/thread (≈3 blocks/SM), and `lg` launches only
  ~2.7 blocks/SM. The larger structural wins (not yet done): a warp-cooperative
  reduction (a warp per beam → more warps to hide latency + shorter per-thread
  chains), or a cuBLAS batched GEMM `M = R̂⁻¹·A` followed by a cheap reduction
  (tile over beams to bound the `O(num_items·num_channels·num_beams)` `M`).
- Covariance + inverse scratch is `O(num_items · num_channels²)`; chunk the item
  axis if a workload needs wider apertures at high detection counts.
- **`eigenvalues` runs a batched `matx::eig` (heevd) every tick, even when the
  port is unwired.** Measured overhead vs. the Capon-only pipeline is small
  (lg 5.3→5.4 ms ≈ +2%, xl 14.9→15.6 ms ≈ +4%, md flat; cold start grows with
  the cached heevd workspace). `peak_extract_kernel` (multi-source) is
  negligible — it re-reads the resident spectrum. If a workload never consumes
  `eigenvalues`, gating the `eig` on the port's connectivity is the obvious
  win, but Holoscan gives no clean in-`compute()` "is this output connected"
  signal (the codebase idiom gates optional outputs on a *parameter*, e.g.
  FMCWRangeProcessor's `waveform`); revisit if the diagnostic cost matters.

## Inherited caveats

- **`is_valid` bool transport (HSDK 4.3).** Shares the CFAR / SnapshotGather
  bool-`DLPack` transport caveat; `is_valid` is accepted as bool or uint8.
