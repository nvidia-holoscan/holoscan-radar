# `ClusterDBSCANOp`

Density clustering (DBSCAN) over a caller-assembled N-D feature matrix — radar-agnostic, so it attaches either before or after angle estimation. It consumes `FeatureAssemblyOp.points` and emits `labels` (and `sizes`) for `DetectionReductionOp` and reports. The fact a caller must not miss: `labels` is a pure function of `(points, epsilon, min_points)` — cluster `0` is the largest; noise is `-1`; a straddling border point joins the core neighbor with the **numerically smallest component root**.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `points` | float32 / float64 | `(num_points, num_features)` | Caller-assembled features (e.g. `[range, velocity, az, el]`); row order is `labels` row order |
| out | `labels` | int64 | `(num_points,)` | Cluster id per point; noise = `-1`; ids by descending size (tie: ascending component root) |
| out (optional, latched) | `sizes` | int64 | `(num_clusters,)` | Point count per cluster, non-increasing; sized to the reported count |
| live | `epsilon` | float64 config Tensor | `(1,)` or `(num_features,)`, rank-1 pageable host | Exact override; wire to enable; extent bounded by `max_features` |
| live | `min_points` | int64 | scalar | Exact override (`<= 0` → `2 · active_features`); wire to enable |
| live | `max_clusters` | int64 | scalar | Exact override (`<= 0` → uncapped); wire to enable |

```text
FeatureAssemblyOp.points → ClusterDBSCANOp → labels (+ sizes) → DetectionReductionOp
```

Contract: [`OPERATOR_API.md` § `ClusterDBSCANOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/clustering/README.md`](../README.md)

## Algorithm

`compute()` resolves the `[live]` `epsilon` / `min_points` / `max_clusters`
overrides into one tick-local stage config, validates the input and derives
geometry in `build_cluster_stage_plan`, then runs the launchers in
`cluster_dbscan_stage.hpp`. `prepare_epsilon_inverse`,
`prepare_dbscan_workspace`, `find_component_roots`,
`compact_component_roots`, and `prepare_ranked_clusters` keep the shell at
orchestration level while retaining all tick scratch in one owning workspace.
Custom CUDA (no MatX expression); CUB exclusive-sum densifies component roots.
Neighbor test, quoted from `dbscan_neighbor`:

```text
q is a neighbor of p when Σ_a ((p_a − q_a) · eps_inv2_a) ≤ 1
where eps_inv2_a = epsilon_a > 0 ? 1 / epsilon_a² : 0
```

That is the contract ellipsoid `Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1` over positive `epsilon` entries. A per-axis zero excludes that feature from the metric while preserving its value in the input row. Length-1 positive `epsilon` broadcasts to every axis.

1. **`mark_core`.** Count neighbors of each point **including itself** over all `num_points` (brute-force `O(N²)`). `is_core` when `count >= min_points`. Seed CC: core → own row index; non-core → `kNoiseRoot` (`-1`).

2. **`cc_step` loop.** Iterative min-label propagation over **core–core** edges, ping-pong buffers, plus one pointer jump `cc_in[cc_in[i]]`. Converges to the minimum row index of each core component (the root). Each iteration D2H-syncs a `changed` flag. Cap: `num_points + 64` iterations, else `ClusterDBSCANOp: connected-components did not converge`.

3. **`assign_border`.** Each non-core point joins the **lowest-root** core neighbor (if any); otherwise stays noise. This is the pinned inter-cluster border tie-break (STATUS.md L9).

4. **Dense ids.** Flag roots (`is_core && cc[i] == i`), CUB exclusive-sum → ascending-root dense ids, `finalize_labels` (noise → `-1`).

5. **Renumber by descending size.** Host argsort of the per-id histogram (`stable_sort` so size ties keep ascending-root order), then `remap_labels`. Cluster `0` is the largest (STATUS.md L10). `labels` remains a pure function of `(points, epsilon, min_points)`.

6. **`max_clusters` cap.** If `max_clusters > 0` and `num_clusters` exceeds it, `cap_labels` drops ids `>= max_clusters` to noise (the **smallest** clusters, after step 5) and warns once: `ClusterDBSCANOp: N clusters exceed max_clusters=M, dropping K surplus cluster(s) to noise` (STATUS.md L7).

7. **`sizes`.** The ordering histogram is reused: its sorted, possibly capped
   host counts are copied to the optional output; non-increasing.

Empty `points` → `labels` and `sizes` both shape `(0,)`.

## Shape and dtype rules

- **`labels`.** Same `num_points` as `points`.
- **`sizes`.** `num_clusters` after the cap (`≤ max_clusters` when set). Reallocated per tick.
- **`points`.** Rank 2, `num_features >= 1`. Precision is template `T`; no leading batch.
- **Extents.** Ctor requires `max_points` (labels pool) and `max_clusters` (sizes pool). `max_features` bounds the `epsilon` override payload and is checked only when an override actually arrives, so a graph that never sends one compiles and runs with `max_features` unset. There is **no** named tick reject when `num_points > max_points` or `reported_clusters > max_clusters` — oversize loans fail at allocate time.
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
  | Missing ctor extent | `ClusterDBSCANOp: MaxTensorExtents.max_points must be a positive value (required for Tensor pool sizing)` (same for `max_clusters`) |
  | Non-device / non-contiguous / wrong dtype | `ClusterDBSCANOp: tensor received on input port 'points' …` |
  | No CUDA stream | `ClusterDBSCANOp: no CUDA compute stream` |

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `epsilon` | float64 vector, length `1` or `num_features` | none (required) | `[live]` Exact host config Tensor | H2D of `1/ε²` or zero on a new `(epsilon, num_features)` key | Neighborhood radius; per-axis zero ignores a feature; `[0.5]` is isotropic |
| `min_points` | int64 | `0` → `2 · active_features` | `[live]` Exact | Free (core threshold) | DBSCAN `minPts` (neighbors including self) |
| `max_clusters` | int64 | `0` (uncapped) | `[live]` Exact | Free (relabel surplus to noise) | Keep the largest `max_clusters`; warn once on overflow |

YAML uses the one-element-list convention for isotropic `epsilon` (`epsilon: [0.5]`), consistent with other vector params (STATUS.md L8).

## SDK implementation notes

- **Construction.** `explicit ClusterDBSCANOp(const MaxTensorExtents& extents)`. Required: `max_points`, `max_clusters`. Optional `max_features` bounds an `epsilon` override payload. `labels` pool: `max_rank = 1`, span `max_points · sizeof(int64)`. `sizes` pool: `max_rank = 1`, span `max_clusters · sizeof(int64)`.
- **Triggering.** `contract()` is `OnEach{points}` plus an unconditional `Exact{}` + `on_missing SafeDefault{}` entry for every `[live]` override port. Wiring a port enables its override; an unwired entry reports `kNotConnected` and takes Params for that tick.
- **Optional output.** `sizes` is **latched**: the first unwired emit warns once, then later ticks skip its pool loan and H2D copy. The cluster-size histogram still runs because required `labels` uses it for deterministic descending-size numbering. See [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md), "Optional output convention".
- **Ports and queues.** `points` and live Inputs: `queue_depth(2)`. `points` is a device tensor; `epsilon` is a rank-1 pageable-host config Tensor. Both outputs: device tensors with the bounds above.
- **Streams and lifetime.** `context.cuda_stream()`. `points` is retained until the next tick. The hot path **does** synchronize: once per CC iteration (`changed` flag), plus cluster-count and size-histogram D2H for the host argsort and exact output shape.
- **Emission.** Required `labels` is committed and emitted first. If still active, optional `sizes` then loans its exact shape and copies the already-sorted host counts.
- **Workspace.** The device `1/ε²` vector is cached by effective `(epsilon, num_features)`. Other `make_device_tensor` temps (is_core, CC ping-pong, CUB scan, histogram, remap) remain per-tick operator workspace, not output loans; pooling them remains a profiling-driven optimization.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.ClusterDBSCANParams` version 1. Fields: `epsilon`, `min_points`, `max_clusters`.
- **Precision.** `ClusterDBSCANOp<float>`, `ClusterDBSCANOp<double>`.
- **Live override payloads.** `Input<int64_t> min_points` / `max_clusters` resolve with `resolve_live_override`. `epsilon` rides a rank-1 pageable-host float64 config Tensor, validated on arrival (rank, host memory, contiguity, dtype, `max_features` extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). Graph tests cover Exact matches, skewed misses, and reversion to Params for all three live families.

## Consumer lifetime notes

`labels` and `sizes` are fresh pool-loaned device buffers each tick. A consumer may retain them.

## Performance

**Cold.** First-tick scratch allocs and `1/ε²` H2D.

**Steady.** Cache-hit reuse of the device `1/ε²` vector; `O(N² · F)` neighbor sweeps (mark_core, each CC step, border assign) plus a host argsort of `num_clusters` counts. CC iteration count is bounded by the graph diameter. Appropriate for post-CFAR tens–low-thousands of points; not a general-purpose large-N DBSCAN.

**Live rebuilds.** `min_points` / `max_clusters` are free. A new effective `(epsilon, num_features)` pair rebuilds and uploads `1/ε²`; an unchanged pair reuses the cached device vector. Every epsilon value still selects a new neighbor graph computation for that tick.

Harness: `performance/cpp/perf_cluster_dbscan.cpp` (Graph). Axes: `(N, F)` ladder `xs`…`xl5` (256×2 through 8192×5; well-separated blobs). It measures the production Exact `min_points` override; `epsilon` and `max_clusters` overrides are not timed. Both outputs are wired, so timings include `sizes` publication. No measured numbers are recorded in-repo.

```bash
ctest --test-dir build -R perf_cluster_dbscan_smoke
```

## Tests and verification

| Check | Proves |
| --- | --- |
| `TwoBlobsAndNoise` | Synthetic blobs → correct membership; noise = `-1` (`TESTING.md`) |
| `AnisotropicPerAxisEpsilon` | Per-axis ellipsoid (`TESTING.md`) |
| `MinPointsSeparatesCoreFromNoise` / `DefaultMinPointsIsTwiceNumFeatures` | `minPts` including the `2·F` default (`TESTING.md`) |
| `NumbersClustersByDescendingSize` | Cluster `0` is largest; `sizes` non-increasing |
| `MaxClustersCapsAndDropsSurplus` / `MaxClustersKeepsLargest` | Cap drops smallest; warn; `labels`/`sizes` consistent (`TESTING.md`) |
| `BorderPointJoinsMinRootCluster` | A border point shared by two components takes the numerically smallest component root |
| `EmptyPointsEmitsEmptyOutputs` | `(0,)` / `(0,)` |
| `FloatAndDoubleAgree` | Both precisions |
| `*ConfigurableFromParams` / `Rejects*` | Params fields and epsilon/shape errors |
| `RejectsOpaqueLeadingBatch` | Rank is exactly `(num_points, num_features)` |
| `LiveEpsilon*` (Exact match, skewed capture, revert-to-Params, malformed, bound overflow) | Config-Tensor override: Exact join, revert-to-Params, payload rejects |
| `LiveMinPoints*` / `LiveMaxClustersRevertsToParamsAfterMiss` | Scalar Exact match/skew/miss behavior and one-tick reversion to Params |
| `UnwiredSizesDoesNotFailTheRunOrDisturbLabels` | `sizes` may be omitted; latching does not remove the ordering histogram labels require |

`TESTING.md` row-alignment is implicit (`labels.shape == (num_points,)`). Python tests exist but are not built.

```bash
ctest --test-dir build -R test_cluster_dbscan --output-on-failure
```

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

`extents` must set `max_points` and `max_clusters`. Python bindings are the HSDK 4.x trampoline and are **not built** on this branch.

## Contract conformance

Conforms on ports, live payload forms, the ellipsoid neighborhood,
core/`minPts` semantics, descending-size ids, min-root border tie-break,
`max_clusters` keeps-largest, and exact-sized latched-optional `sizes`.

Header comment block matches the implementation (config-Tensor `epsilon`, Exact scalar Inputs with unconditional `contract()` entries, descending-size numbering).
