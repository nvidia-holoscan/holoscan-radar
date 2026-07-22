# ClusterDBSCANOp — open items for team review

Concrete questions surfaced while implementing `ClusterDBSCANOp` against the
locked contract in [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md).
None of these silently change the contract; they record decisions taken with a
faithful default plus the question for the team.

## 1. `max_clusters` — which clusters are "surplus", and what happens to their points

- **Context.** The contract says: *"Upper cap on the number of clusters reported
  in `sizes`; surplus dropped and a warning raised if exceeded. `sizes` is sized
  to the actual cluster count (≤ this cap)."* It pins the effect on `sizes` but
  not (a) which clusters are surplus when `num_clusters > max_clusters`, nor
  (b) what `labels` reports for points in dropped clusters.
- **Current behavior.** Clusters are numbered by **descending point count**
  (cluster `0` is the largest; ties broken by ascending component root). When
  capped, the clusters with the **highest** dense ids — now the **smallest** —
  are dropped, and their points are **relabeled to noise (`-1`)** so `labels`,
  `sizes`, and the reported cluster count stay mutually consistent (`sizes[k]`
  is valid for every `k` a `label` names). The cap therefore keeps the
  `max_clusters` **largest** clusters.
- **Impact.** Any downstream consumer that indexes `sizes` by `labels`. The
  policy is deterministic and keeps the most significant clusters.
- **Resolved (L7, 2026-07-22).** Keep the **smallest** surplus clusters (drop
  them to noise) — realized by numbering ids by descending size (see item 6,
  resolved as L10) so the existing "drop highest ids" cap keeps the largest.
  Chosen over the previous spatially-arbitrary ascending-root drop, and over
  "leave surplus `labels` populated / truncate only `sizes`" (rejected — it
  breaks the `sizes[label]` invariant). Making the *ordering rule* itself a
  parameter is deferred (see item 8). See STATUS.md → Settled decisions (L7).
- **Status.** `resolved` — cap keeps largest via descending-size numbering;
  landed + tested (`NumbersClustersByDescendingSize`, `MaxClustersKeepsLargest`,
  C++ and Python).

## 2. `epsilon` scalar-vs-vector encoding / ergonomics

- **Context.** The contract types `epsilon` as *"float scalar or
  `(num_features,)` vector"*. A single `[live]`-overridable Holoscan parameter
  cannot be both a scalar and a vector, so it is modeled as a
  `std::vector<double>`: length 1 → isotropic scalar, length `num_features` →
  per-axis. YAML/`Arg` therefore writes the scalar case as a one-element list
  (`epsilon: [0.5]`), not a bare `epsilon: 0.5`.
- **Impact.** Config ergonomics only; functionally complete (both isotropic and
  anisotropic neighborhoods work, and the override port reuses the proven
  `std::any` vector path).
- **Question.** Is the length-1-vector encoding acceptable, or should the
  operator additionally accept a bare scalar `epsilon` (a second parameter or a
  YAML-converter that promotes a scalar to `[scalar]`)?
- **Resolved (L8, 2026-07-22).** **Ratify the length-1-vector encoding**
  (`epsilon: [0.5]` for the isotropic case) — chosen over adding a bare-scalar
  converter. Rationale: the `[list]` idiom is already how every other vector
  parameter in the suite is written (`guard_cells`, `training_cells`,
  `nci_axes`, `spatial_smoothing`), so ratifying it is the *consistent* choice;
  a scalar-vs-vector special case just for `epsilon` would make it the odd one
  out (and, for consistency, would pull toward the same converter on every
  vector param — scope creep for a cosmetic YAML gain). Docs-only: the
  `OPERATOR_API.md` `epsilon` bullet now states the isotropic case is a
  one-element list. See STATUS.md → Settled decisions (L8).
- **Status.** `resolved` — length-1-vector encoding ratified; contract wording
  tightened to match.

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

## 5. Border-point assignment is deterministic (lowest-root neighbor)

- **Context.** DBSCAN has an inherent ambiguity: a border point (non-core, but
  within `epsilon` of a core point) that is reachable from cores in **two
  different clusters** may validly join either. The contract in
  [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) does not pin the
  tie-break, so the implementation had to choose one.
- **Current behavior.** Each border point joins the neighboring core with the
  **numerically smallest component root** (lowest member row index). The result
  is deterministic and independent of thread / iteration order (`labels` is a
  pure function of `points`, `epsilon`, `min_points`).
- **Impact.** The clustering *partition* matches any faithful DBSCAN, but the
  cluster a given inter-cluster border point lands in can differ from other
  implementations — e.g. scikit-learn assigns a border point to the first
  cluster whose expansion reaches it in index order, not the min-root one. Only
  inter-cluster border points are affected.
- **Question.** Is min-root the desired rule, or should border points go to the
  nearest core / the first-reached cluster? Whatever the answer, the contract
  should pin it so the behavior is not implementation-defined.
- **Resolved (L9, 2026-07-22).** **Ratify min-root** and pin it in the contract
  (chosen over nearest-core and over scikit-learn's first-reached rule).
  Rationale: only inter-cluster border points are affected (a rare edge case),
  the partition is identical either way, and downstream `DetectionReductionOp`
  is insensitive to which cluster a straddling point joins — while min-root is
  already deterministic and **order-independent** (preserving the
  `labels`-is-a-pure-function guarantee). First-reached is order-dependent and
  cannot be made order-independent on the GPU (rejected); nearest-core adds
  kernel distance-tracking + a secondary tie-break for no partition-level
  benefit. Docs-only: the `OPERATOR_API.md` `labels` bullet now states the
  min-root border rule. See STATUS.md → Settled decisions (L9).
- **Status.** `resolved` — min-root ratified; contract now pins the tie-break.

## 6. Cluster-id numbering order

- **Context.** The contract says `labels` are dense with noise `-1` but did not
  originally specify the order in which clusters are numbered.
- **Current behavior.** Clusters are numbered by **descending point count**
  (cluster `0` is the largest), ties broken by ascending component root (lowest
  member row index). `sizes` is row-aligned to that numbering (non-increasing).
- **Impact.** "Cluster 0 is the largest" is now guaranteed. Numbering still
  differs from scikit-learn's discovery order; the partition is identical, only
  the ids differ. Downstream code must not assume discovery-order ids.
- **Resolved (L10, 2026-07-22).** Pin **descending-size, ascending-root
  tie-break** in the contract (`OPERATOR_API.md` `labels`/`sizes`). Chosen over
  the prior ascending-root numbering — descending size makes the `max_clusters`
  cap keep the largest clusters (item 1 / L7) for free and gives a useful
  "cluster 0 is the largest" guarantee. Deterministic. Making the order a
  parameter is deferred (item 8). See STATUS.md → Settled decisions (L10).
- **Status.** `resolved` — descending-size numbering landed + tested.

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
