# MIMODeinterleaveOp — Open Items

Deferred work and **team-visible imposed decisions** for this operator.
Remove items when they land (note the commit in the PR).

---

## A. Imposed decisions (contract was underspecified)

### A.1 First-wave TDM schedule uniformity

**Contract:** `tx_schedule` may carry `tx_order` or explicit `chirp_tx_index`;
downstream TDM compensation assumes uniform per-TX spacing.

**Pinned implementation:** `TXSchedule::expand_chirp_tx_index` rejects:

- Non-repeating patterns (the length-`M` cycle does not repeat identically
  across all `L` repetitions).
- Unequal TX counts (each TX must appear exactly `chirps_per_tx = slow_time / M`
  times in the interleaved axis).

Custom `tx_order` permutations of `[0, …, M−1]` are allowed when they repeat
uniformly. Arbitrary `chirp_tx_index` vectors are allowed only when they still
satisfy the same uniformity checks.

**Rationale:** `MIMOCoherentCompensationOp` TDM correction uses a fixed
inter-TX offset `PRI/M` between firings, not arbitrary dwell times per TX
(see that operator's README). Identity-`tx_order` vs firing-slot indexing is
a separate compensation gap (`OPEN_ITEMS.md` item 1 there).

### A.2 Slow-time divisibility

Input slow-time length `M·L` must be a positive multiple of `num_tx`. Otherwise
reject at gather-map build. `L` is inferred as `slow_time / num_tx`; not a
separate parameter.

### A.3 Gather map / output axis order

**Underspecified:** row-major layout of the deinterleaved `(M_tx, N_rx, L,
samples)` block.

**Pinned implementation:** for each output slot `(tx, l)`, gather from
interleaved slow-time index `k` where TX `tx` fired for the `l`-th time (see
`TXSchedule::build_gather_map`). Output trailing axes are
`(M_tx, N_rx, L, samples_per_chirp)` with channel-major `(tx, rx)` ordering
inside the stage layout parser. Dtype preserved (complex IQ or real IF).

### A.4 Relationship to `FMCWWaveformDescriptor` PRI

This operator does not read `waveform`. The caller must set
`waveform.chirp_repetition_interval` on downstream operators to the **per-TX
effective PRI** (`M ×` physical inter-chirp spacing). See `TXSchedule` notes in
`OPERATOR_API.md` and `applications/fmcw_mimo` for the simulator split between
physical vs effective PRI.

---

## B. Deferred follow-on work

### B.1 Non-uniform / staggered TDM schedules

Staggered / non-uniform TDM (TX dwell or count varies) is **out of the v1 scope**:
uniform round-robin only, and a non-uniform schedule is rejected with a clear
error (settled — STATUS.md L12). Supporting it would need a per-TX-timing
schedule descriptor plus a generalized `MIMOCoherentCompensationOp` compensation
model (itself an open design question). Tracked as a deferred future enhancement
on STATUS.md → Roadmap.

### B.2 Per-tick output allocation (module-wide)

Pure gather, strong ping-pong candidate. See [`operators/OPEN_ITEMS.md`](../../OPEN_ITEMS.md) §1.
