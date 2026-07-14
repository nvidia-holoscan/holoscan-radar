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
- **Current behavior.** Clusters are assigned dense ids `[0, num_clusters)` in
  ascending component-root (lowest member row index) order. When capped, the
  clusters with the **highest** dense ids are dropped, and their points are
  **relabeled to noise (`-1`)** so `labels`, `sizes`, and the reported cluster
  count stay mutually consistent (`sizes[k]` is valid for every `k` a `label`
  names).
- **Impact.** Any downstream consumer that indexes `sizes` by `labels`. The
  "drop highest ids / relabel to noise" choice is deterministic but arbitrary;
  a "keep the largest clusters" or "keep lowest-id clusters but leave labels
  untouched" policy would be equally defensible.
- **Question.** Is drop-highest-id + relabel-to-noise the intended semantics, or
  should the cap keep the largest clusters (by point count) and/or leave the
  surplus `labels` populated but simply truncate `sizes`?
- **Status.** `open` — implemented with the consistent default above.

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
- **Status.** `open` — length-1-vector encoding shipped; documented in the
  header and README.

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
