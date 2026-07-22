# ClusterDBSCANOp — open items for team review

Concrete questions surfaced while implementing `ClusterDBSCANOp` against the
locked contract in [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md).
None of these silently change the contract; they record decisions taken with a
faithful default plus the question for the team.

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

- **Context.** Cluster ids are numbered by descending size (item 6 / L10), which
  also fixes which clusters a `max_clusters` cap keeps (item 1 / L7). That single
  fixed policy covers the current radar path, but the ordering rule is a natural
  candidate for a parameter once a consumer needs a different one.
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
