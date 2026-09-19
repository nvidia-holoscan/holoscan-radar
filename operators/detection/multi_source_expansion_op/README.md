# `MultiSourceExpansionOp`

`MultiSourceExpansionOp` merges selectively evaluated multi-source angle results back into a complete sparse-detection row set. Unselected detections and selected detections not confirmed as multi-source retain their baseline angle. Confirmed rows expand to one row per resolved angle while duplicating the original detection index.

The operator knows nothing about CFAR, Beamscan, a particular super-resolution algorithm, or radar modality. It consumes ordinary device tensors and the common estimator outputs `aoa_estimates`, `doa_multi`, and `source_count`.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `detection_idx` | int64 | `(num_items, num_index_columns)` | Original row identity; 2–10 columns preserved verbatim. |
| in | `baseline_aoa` | float32 / float64 | `(num_items, 2)` | One fallback angle per original row. |
| in | `selected_idx` | int64 | `(num_selected,)` | Strictly ascending, unique original-row indices. |
| in | `resolved_aoa` | float32 / float64 | `(num_selected, 2)` | Dominant angle from the selected estimator. |
| in | `doa_multi` | float32 / float64 | `(num_selected, source_slots, 2)` | Source list; `source_slots <= max_sources`. |
| in | `source_count` | int32 | `(num_selected,)` | Valid sources in each selected estimator row. |
| out | `detection_idx_out` | int64 | `(num_output, num_index_columns)` | Original index repeated for every emitted source. |
| out | `aoa_estimates_out` | float32 / float64 | `(num_output, 2)` | Baseline or resolved angles aligned with output indices. |

All six inputs are required and form one zero-window Exact activation.

```text
original detection_idx + baseline_aoa
selected_idx + resolved_aoa + doa_multi + source_count
  → MultiSourceExpansionOp
  → detection_idx_out + aoa_estimates_out
```

Contract: [`OPERATOR_API.md` § `MultiSourceExpansionOp`](../../../docs/planning/OPERATOR_API.md) · group: [`operators/detection/README.md`](../README.md)

## Expansion algorithm

Input rows are visited in original order:

1. An unselected row emits once with its baseline angle.
2. A selected row with `source_count < 2` also emits once with its baseline. Counts zero and one deliberately mean “not confirmed”; the selected estimator's dominant angle is ignored.
3. A selected row with `source_count >= 2` emits `source_count` copies of its index. The first angle is `resolved_aoa`; subsequent angles are `doa_multi[1:source_count]`.

`resolved_aoa` is authoritative for the dominant result under the shared estimator contract, so `doa_multi[..., 0, :]` is not read. Unused multi-source slots need not be finite.

```text
original rows:       A       B       C
baseline angles:     a       b       c
selected_idx:                [1]
source_count:                [3]
resolved/multi:               b0, b1, b2

output rows:         A       B       B       B       C
output angles:       a       b0      b1      b2      c
```

The exact output size is

```text
num_output = num_items + Σ_selected max(source_count - 1, 0)
```

and must not exceed `MaxTensorExtents.max_points`.

## Shape, dtype, and alignment rules

- Precision is fixed by the template instantiation. All angle tensors use the same `float` or `double` dtype.
- Every input must be contiguous and CUDA-accessible. Index tensors are int64 and `source_count` is int32.
- `detection_idx` accepts two range/Doppler columns plus up to eight preserved leading-index columns. Every column is copied unchanged.
- `selected_idx` must be strictly ascending, unique, non-negative, and less than `num_items`, making the merge deterministic and linear.
- `num_selected == 0` is legal; estimator outputs have zero leading rows and all original rows fall back. `num_items == 0` produces rank-2 zero-row outputs.
- Confirmed emitted angles must be finite. Baseline fallback angles are copied as supplied; upstream baseline-validity policy remains authoritative.
- Both outputs preserve the common Exact timestamp.

Rejected combinations and messages include:

| Condition | Message |
| --- | --- |
| invalid index rank/width | `detection_idx must have shape (num_items, 2 [+ n_batch]) with at most 10 columns` |
| original rows beyond capture | `num_items exceeds max_items` |
| selected rows exceed originals | `num_selected cannot exceed num_items` |
| baseline/resolved/count mismatch | named expected-shape diagnostic |
| invalid multi-source shape/capacity | `doa_multi must have shape (num_selected, max_sources, 2) within ctor bounds` |
| duplicate, unsorted, negative, or out-of-range selection | `selected_idx must be strictly ascending, unique, and in range` |
| invalid source count | `source_count must be within [0, doa_multi.shape(1)]` |
| non-finite confirmed angle | `confirmed source angles must be finite` |
| output capacity exceeded | `expanded rows exceed max_points` |
| missing stream | `MultiSourceExpansionOp: no CUDA stream` |

## Parameters and sizing

There are no runtime parameters or live overrides. Configuration is captured output sizing:

| `MaxTensorExtents` field | Meaning |
| --- | --- |
| `max_items` | Maximum original rows and selected-ordinal workspace. |
| `max_sources` | Maximum second axis accepted from `doa_multi`. |
| `max_points` | Maximum expanded rows; normally `max_items × max_sources`. |

The index pool is bounded as `(max_points, 10)` int64 and the angle pool as `(max_points, 2)` real. Runtime outputs remain exact-sized.

## SDK implementation notes

- **Construction.** All three extents are required and positive. The constructor captures pool spans and allocates ordinal/error workspaces.
- **Triggering.** `OnAligned` names all six inputs at zero window and every input uses `Exact{}`. Partial estimator results cannot activate the op.
- **Planning.** One device thread validates selection order, counts, and confirmed angles while building per-row expansion counts and estimator ordinals.
- **Synchronization.** One device-to-host readback returns validation and exact output count; a prefix sum assigns stable output ranges.
- **Emission.** Both loans are prepared before scatter. The angle output publishes before `detection_idx_out`, the final required boundary.
- **Streams and lifetime.** Planning, scan, and scatter use `context.cuda_stream()`. All inputs are retained through activation.
- **Precision.** Explicit instantiations are `MultiSourceExpansionOp<float>` and `MultiSourceExpansionOp<double>`.

## Consumer lifetime notes

Both outputs are fresh pool loans and may be retained. Repeated index rows are intentional: `FeatureAssemblyOp` gathers the same range, velocity, and score for each angular source while consuming the row-aligned expanded angles.

## Performance

Planning is `O(num_items + num_selected × source_count)` and scatter is `O(num_output × num_index_columns)`. One synchronization validates and exact-sizes the output.

`performance/cpp/perf_multi_source_expansion.cpp` sweeps original rows, selected fraction, sources per selected row, and index width while draining both outputs and reporting warmup, setup/cold, and steady timing.

```bash
ctest --test-dir build-perf-check -R perf_multi_source_expansion_smoke
```

## Tests and verification

`tests/cpp/detection/test_multi_source_expansion.cu` covers mixed ordering, two through maximum-source plans, zero/one-source fallback, empty inputs and selections, malformed selected indices, inconsistent counts, non-finite confirmed angles, and both precisions.

`tests/cpp/integration/test_selective_aoa_operators.cpp` verifies actual emitted index and angle tensors after a modality-independent screen and common-estimator mock. It covers widened indices, duplicated rows, fallback, deterministic ordering, Exact alignment, and timestamps.

```bash
ctest --test-dir build -R 'test_multi_source_expansion|test_selective_aoa_operators' --output-on-failure
```

## Usage

```cpp
auto expansion = graph.op<MultiSourceExpansionOp<float>>(
    "multi_source_expansion",
    MaxTensorExtents{.max_items = max_detections,
                     .max_points = max_detections * max_sources,
                     .max_sources = max_sources});

graph.add_flow(cfar->detection_idx, expansion->detection_idx, q);
graph.add_flow(beamscan->aoa_estimates, expansion->baseline_aoa, q);
graph.add_flow(screen->selected_idx, expansion->selected_idx, q);
graph.add_flow(estimator->aoa_estimates, expansion->resolved_aoa, q);
graph.add_flow(estimator->doa_multi, expansion->doa_multi, q);
graph.add_flow(estimator->source_count, expansion->source_count, q);
graph.add_flow(expansion->detection_idx_out, features->detection_idx, q);
graph.add_flow(expansion->aoa_estimates_out, features->aoa_estimates, q);
```

## Contract conformance

The operator is estimator- and modality-independent. It preserves original order and complete index rows, accepts only confirmed multiplicity (`source_count >= 2`), and falls back to one baseline row otherwise. It intentionally does not cluster, choose estimators, reinterpret counts, or deduplicate angles.
