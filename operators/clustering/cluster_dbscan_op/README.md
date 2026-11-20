# `ClusterDBSCANOp`

## Purpose

Density clustering (DBSCAN) over a caller-assembled N-D feature matrix — radar-agnostic, so it attaches either before or after angle estimation. It consumes `FeatureAssemblyOp.points` and emits `labels` (and `sizes`) for `DetectionReductionOp` and reports. The fact a caller must not miss: `labels` is a pure function of `(points, epsilon, min_points)` — cluster `0` is the largest; noise is `-1`; a straddling border point joins the core neighbor with the **numerically smallest component root**.

## Responsibilities

- Applies per-feature normalization, finds density-connected components, and marks noise rows.
- Owns deterministic cluster ordering, optional size output, maximum-cluster truncation, and tick-local clustering overrides.
- Does not assemble physical features, choose epsilon units, or reduce clusters to representatives.

## When to use

Use after `FeatureAssemblyOp` when detections should be grouped by density in a normalized feature space without choosing the number of clusters in advance. Follow it with `DetectionReductionOp` when downstream processing needs one representative per cluster. Do not feed raw RD tensors or mixed-unit features without meaningful per-axis `epsilon` scaling.

```text
FeatureAssemblyOp.points → ClusterDBSCANOp → labels (+ sizes) → DetectionReductionOp
```

Declarations and parameter defaults: [cluster_dbscan_op.hpp](cluster_dbscan_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `points` | float32 / float64 | `(num_points, num_features)` | Required trigger/aligned input | Caller-assembled features (e.g. `[range, velocity, az, el]`); row order is `labels` row order |
| out | `labels` | int64 | `(num_points,)` | Required output | Cluster id per point; noise = `-1`; ids by descending size (tie: ascending component root) |
| out (optional, latched) | `sizes` | int64 | `(num_clusters,)` | Optional latched output | Point count per cluster, non-increasing; sized to the reported count |
| live | `epsilon` | float64 config Tensor | `(1,)` or `(num_features,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable; extent bounded by `max_features` |
| live | `min_points` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override (`<= 0` → `2 · active_features`); wire to enable |
| live | `max_clusters` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override (`<= 0` → uncapped); wire to enable |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_points` and `max_clusters` required; `max_features` optional for live `epsilon` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `epsilon` | float64 vector, length `1` or `num_features` | none (required) | `[live]` Exact host config Tensor | Neighborhood radius; per-axis zero ignores a feature; `[0.5]` is isotropic Cost: H2D of `1/ε²` or zero on a new `(epsilon, num_features)` key. |
| `min_points` | int64 | `0` → `2 · active_features` | `[live]` Exact | DBSCAN `minPts` (neighbors including self) Cost: Free (core threshold). |
| `max_clusters` | int64 | `0` (uncapped) | `[live]` Exact | Keep the largest `max_clusters`; warn once on overflow Cost: Free (relabel surplus to noise). |

## Algorithm

`compute()` resolves the `[live]` `epsilon` / `min_points` / `max_clusters`
overrides into one tick-local stage config, validates the input and derives
geometry in `build_cluster_stage_plan`, then runs the launchers in
`cluster_dbscan_stage.hpp`. `prepare_epsilon_inverse`,
`prepare_dbscan_workspace`, `find_component_roots`,
`compact_component_roots`, and `prepare_ranked_clusters` keep the shell at
orchestration level while retaining all tick scratch in one owning workspace.
Custom CUDA (no MatX expression); CUB exclusive-sum densifies component roots.
The normalized neighborhood test is:

```text
q is a neighbor of p when Σ_a ((p_a − q_a) · eps_inv2_a) ≤ 1
where eps_inv2_a = epsilon_a > 0 ? 1 / epsilon_a² : 0
```

That is the contract ellipsoid `Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1` over positive `epsilon` entries. A per-axis zero excludes that feature from the metric while preserving its value in the input row. Length-1 positive `epsilon` broadcasts to every axis.

1. **`mark_core`.** Count neighbors of each point **including itself**, stopping once the positive effective `min_points` threshold is reached. `is_core` is the same `count >= min_points` boolean; points that never reach it scan all `num_points` (worst-case `O(N²)`). Seed CC: core → own row index; non-core → `kNoiseRoot` (`-1`). Float with 2, 3 or 5 features uses a separate fixed-bound distance loop with the same ordered scalar arithmetic. Every double input and other float feature count uses the original generic kernel.

2. **`cc_step` loop.** Iterative min-label propagation over **core–core** edges, ping-pong buffers, plus one pointer jump `cc_in[cc_in[i]]`. Before evaluating a distance, skip a core neighbor whose snapshot label is at least the running minimum; it cannot lower that minimum. The predicate and per-iteration labels are unchanged. Converges to the minimum row index of each core component (the root). Each iteration D2H-syncs a `changed` flag. Cap: `num_points + 64` iterations, else `ClusterDBSCANOp: connected-components did not converge`.

3. **`assign_border`.** Each non-core point joins the **lowest-root** core neighbor (if any); otherwise stays noise. This is the pinned inter-cluster border tie-break.

4. **Dense ids.** Flag roots (`is_core && cc[i] == i`), CUB exclusive-sum → ascending-root dense ids, `finalize_labels` (noise → `-1`).

5. **Renumber by descending size.** Host argsort of the per-id histogram (`stable_sort` so size ties keep ascending-root order), then `remap_labels`. Cluster `0` is the largest. `labels` remains a pure function of `(points, epsilon, min_points)`.

6. **`max_clusters` cap.** If `max_clusters > 0` and `num_clusters` exceeds it, `cap_labels` drops ids `>= max_clusters` to noise (the **smallest** clusters, after step 5) and warns once: `ClusterDBSCANOp: N clusters exceed max_clusters=M, dropping K surplus cluster(s) to noise`.

7. **`sizes`.** The ordering histogram is reused: its sorted, possibly capped
   host counts are copied to the optional output; non-increasing.

Empty `points` → `labels` and `sizes` both shape `(0,)`.

## Shape and dtype rules

- **`labels`.** Same `num_points` as `points`.
- **`sizes`.** `num_clusters` after the cap (`≤ max_clusters` when set). Reallocated per tick.
- **`points`.** Rank 2, `num_features >= 1`. Precision is template `T`; no leading batch.
- **Extents.** Ctor requires `max_points` (labels pool) and `max_clusters` (sizes pool). Inputs with `num_points > max_points` fail before tick workspace allocation or GPU work; points are never truncated. Inputs within the bound retain the same clustering behavior, including quadratic worst-case work. `max_features` bounds the live `epsilon` payload only when an override arrives, so it may be unset when no override is sent. Oversized `sizes` loans still fail at allocation; the pool extent is distinct from the live `max_clusters` retention parameter.
- **Legal mid-stream change.** `num_points` / `num_features` inferred per tick. A `[live]` `epsilon` / `min_points` / `max_clusters` change changes membership without changing `labels` rank.
- **Rejected combinations:**

  | Condition | Message |
  | --- | --- |
  | `points` not rank 2 | `ClusterDBSCANOp: points must have shape (num_points, num_features)` |
  | `num_features < 1` | `ClusterDBSCANOp: num_features must be >= 1` |
  | Empty `epsilon` | `ClusterDBSCANOp: epsilon is required` |
  | `epsilon` length not 1 and not `num_features` | `ClusterDBSCANOp: epsilon must be a scalar (length 1) or match num_features` |

| Non-finite or negative entry | `ClusterDBSCANOp: epsilon values must be finite and >= 0` |
| Scalar zero or all-zero vector | `ClusterDBSCANOp: epsilon must select at least one active feature` |
  | `epsilon` override with no `max_features` | `ClusterDBSCANOp: MaxTensorExtents.max_features must be positive to accept an epsilon override` |
  | `num_points > INT_MAX` | `ClusterDBSCANOp: num_points exceeds the supported range` |
  | `num_points > max_points` | `ClusterDBSCANOp: num_points N exceeds MaxTensorExtents.max_points M` |
  | Missing ctor extent | `ClusterDBSCANOp: MaxTensorExtents.max_points must be a positive value (required for Tensor pool sizing)` (same for `max_clusters`) |
  | Non-device / non-contiguous / wrong dtype | `ClusterDBSCANOp: tensor received on input port 'points' …` |
  | No CUDA stream | `ClusterDBSCANOp: no CUDA compute stream` |

## Performance

**Cold.** First-tick scratch allocs and `1/ε²` H2D.

**Steady.** The CC neighbor loop skips distance evaluation when the snapshot neighbor label cannot lower its current minimum; pointer jump, exact per-iteration labels and convergence remain unchanged. Cache-hit reuse of the device `1/ε²` vector; `O(N² · F)` neighbor sweeps (mark_core, each CC step, border assign) plus a host argsort of `num_clusters` counts. CC iteration count is bounded by the graph diameter. Appropriate for post-CFAR tens–low-thousands of points; not a general-purpose large-N DBSCAN.

A paired Thor HSDK 5 check (5 warmups, 50 measured frames, three fresh pairs) measured large float marginal medians of 1.874515→1.722816ms, with a 156.630µs/8.325% paired reduction (8.091–8.861% across pairs). All nine selected protected endpoint rules passed; the same 38 native tests and exact labels/sizes exports across eight cases passed in both precisions. This is a workload-specific throughput result.

Latency tradeoffs remain: small double's fresh paired median slowed 10.028µs/2.089%. The capped float workload had recurring isolated slow frames; one fresh pair's maximum increased by 2.249ms even while its full 50-frame mean improved 6.18%. Setup, cold and override costs were mixed; these samples do not establish a tail-latency bound or universal improvement.

**Live rebuilds.** `min_points` / `max_clusters` are free. A new effective `(epsilon, num_features)` pair rebuilds and uploads `1/ε²`; an unchanged pair reuses the cached device vector. Every epsilon value still selects a new neighbor graph computation for that tick.

## Usage

```yaml
cluster_dbscan:
  epsilon: [3.0, 2.0]
  min_points: 1
  max_clusters: 0
```

```cpp
auto cluster = graph.op<ClusterDBSCANOp<float>>("cluster_dbscan", extents);
cluster.set_params(from_config("cluster_dbscan"));

graph.add_flow(feat->points, cluster->points, q);
graph.add_flow(cluster->labels, reduction->labels, q);
graph.add_flow(cluster->sizes, report->sizes, q);  // Optional.
```

`extents` must set `max_points` and `max_clusters`.
