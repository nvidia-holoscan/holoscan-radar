# ClusterDBSCANOp — open items for team review

Concrete questions surfaced while implementing `ClusterDBSCANOp` against the
locked contract in [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md).
None of these silently change the contract; they record decisions taken with a
faithful default plus the question for the team. HSDK 5 / EA migration items
are indexed in [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream `[live] min_points` / `max_clusters` coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky”.

**Class:** Wiring settled (L16 / L17, 2026-08-21); Graph test coverage still
deferred on EA. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 and
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Decision.** `min_points` / `max_clusters` join the `points` trigger with
`Exact{}` and `on_missing SafeDefault{}`, each gated by `expect_<name>`
(default false). `sticky_<name>` (default true) latches the last Exact
match; false reverts to Params on a miss. The application stamps the
frame's transmit time on both the override and the feature matrix.
`expect_*=true` requires the matching `add_flow`
(`TEMPORAL_INPUT_UNCONNECTED` otherwise); wiring without the flag is
`RUNTIME_OPERATOR_REPLAY_MISMATCH`.

**Still deferred.**

- `LiveMinPointsStickyOverride` / `LiveMaxClustersStickyOverride` are still
  `GTEST_SKIP`’d — the Graph harness for a mid-stream POD override on this
  one-in / two-out operator is not written yet.
- `perf_cluster_dbscan` measures an inline `min_points` transition with the
  supported single-clock / exact-contract / single-partition shape.
- Params authoring covers selection (`MinPointsConfigurableFromParams` /
  `MaxClustersConfigurableFromParams`).

**Next.** Port the Graph live-override harness (the pattern
`test_fmcw_range_processor.cpp` uses for `nfft`) to this fixture and
unskip.

**Status:** open — test coverage only; the contract wiring is settled.
`max_clusters` perf isolation is still owed; `min_points` perf timing is
restored (§8).

---

## 2. `[live] epsilon` is Params-only (unbounded vector)

**Blocks checklist:** Tests → live `epsilon` mid-stream sticky override.

**Class:** Durable HSDK 5 (`PublishedPayload` admission). See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**What fails.** `Input<std::vector<double>>` is rejected at compile time.
`epsilon` is length `1` (isotropic) or `num_features` (per-axis); `num_features`
comes from the caller-assembled `points` matrix and is unbounded, so a fixed
POD like `SnapshotWindowOverride` does not fit.

**Interim.** `epsilon` stays on `ClusterDBSCANParams` only (`set_params` /
YAML). `LiveEpsilonStickyOverride` is `GTEST_SKIP`’d. Params coverage remains
(`EpsilonConfigurableFromParams`, anisotropic / isotropic unit tests). The
scalar lives (`min_points` / `max_clusters`) do join `contract()` behind
`expect_*` (§1); only `epsilon` is portless.

**Next.** Prefer `Input<Tensor>` (or a durable schema / bounded POD once a
module max-feature count is pinned) and restore sticky live + Graph tests.

**Status:** open — ported compute path; live vector port deferred.

---

## 3. Brute-force `O(num_points²)` neighbor graph — scaling (perf follow-up)

- **Context.** Neighbor counting, connected-components propagation, and border
  assignment are each an `O(num_points²)` sweep (no spatial index). This is
  correct and appropriate for the sparse detection counts this stage sees after
  CFAR (tens–low thousands of points), and matches the per-detection brute-force
  philosophy of the sibling bridge operators.
- **Impact.** Steady-time scales quadratically in the point count; the
  connected-components loop also issues one small D2H `changed`-flag sync per
  iteration (bounded by the graph diameter).
- **Follow-up (not a contract question).** A uniform-grid spatial index (or an
  atomic union-find with an adjacency pass) would cut both the neighbor sweeps
  and the per-iteration syncs for large point clouds. Deferred; `perf_cluster_
  dbscan` characterizes the current implementation so the upgrade has a
  baseline.
- **Status.** `landed with caveat` — brute-force is the shipped v1.

## 4. Per-tick scratch allocation (perf follow-up)

- **Context.** Each tick allocates `num_points`-sized scratch (is_core, two
  connected-components buffers, root flags, dense ids, CUB temp) plus the
  `labels` / `sizes` outputs. These are proportional to the (sparse) point
  count, not to any dense RD map, so the per-tick allocation is modest — but it
  is still per-tick.
- **Follow-up.** Pool the scratch (and optionally ping-pong `labels`) keyed on
  `num_points`, mirroring the CFAR dense-buffer pattern, if profiling shows the
  allocation matters. Deferred.
- **Status.** `landed with caveat`.

## 5. Performance harness (closed for Graph + scalar timing)

`performance/cpp/perf_cluster_dbscan.cpp` uses the HSDK 5 Graph source and
reports cold / steady / inline `min_points` override timing. `epsilon` remains
Params-only (§2); dedicated `max_clusters` override isolation remains in §1.

**Status:** closed for the Graph port and representative scalar override.

## 6. Python bindings

**Blocks checklist:** only if this PR were a bindings wave (module-wide
Python is deferred per checklist / STATUS).

**Class:** Our deferred work + EA incomplete Python surface. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6.

Existing `python/_cluster_dbscan_op_bindings.cpp` is still the HSDK 4.x
`Fragment` / `Arg` trampoline and is not built
(`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

**Status:** open — deferred; not required to land the C++ HSDK 5 port.

## 7. Scope vs. general-purpose DBSCAN (deliberate non-features)

- **Context.** Relative to general-purpose DBSCAN libraries (e.g.
  scikit-learn), the operator intentionally omits several features. Recorded so
  the scope is explicit and a consumer porting from sklearn does not assume
  parity.
- **Omissions.** (a) **Single distance metric** — the axis-scaled Euclidean
  ellipsoid pinned by the contract; no Minkowski-`p`, cosine, or
  precomputed-distance-matrix metrics. (b) **No `sample_weight`** — every point
  counts as 1 toward `min_points`. (c) **No core-sample introspection** — only
  `labels` + `sizes` are emitted; the core / border / noise distinction is not
  exposed (sklearn's `core_sample_indices_` / `components_` have no equivalent).
- **Impact.** None for the radar clustering path as contracted; matters only if
  a future consumer needs one of these.
- **Status.** `out of scope` — recorded for transparency; revisit per-feature if
  a consumer requires it (would be a contract change).

## 8. Parameterize the cluster-ordering policy (future enhancement)

- **Context.** Cluster ids are numbered by descending size (L10), which also
  fixes which clusters a `max_clusters` cap keeps (L7). That single fixed policy
  covers the current radar path, but the ordering rule is a natural candidate
  for a parameter once a consumer needs a different one.
- **Proposal (deferred).** A static `cluster_order` parameter — e.g.
  `size_desc` (default, current behavior), `root_asc` (the previous
  ascending-component-root numbering), or `discovery` (scikit-learn-style
  first-reached order). The cap semantics would follow the chosen order (drop
  the highest ids), so `size_desc` keeps the largest, `root_asc` keeps the
  lowest-root, etc. Would need the param, a small host branch selecting the
  argsort key (size vs. root vs. discovery index), and per-order tests.
- **Impact.** None today; the fixed `size_desc` policy is contract (L7 / L10).
  This only matters if a downstream consumer requires a different ordering.
- **Status.** `deferred` — tracked as a future enhancement; not a blocker and
  not a contract gap (the current order is pinned).
