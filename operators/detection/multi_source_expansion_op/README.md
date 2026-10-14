# `MultiSourceExpansionOp`

## Purpose

`MultiSourceExpansionOp` merges selectively evaluated multi-source angle results back into a complete sparse-detection row set. Unselected detections and selected detections not confirmed as multi-source retain their baseline angle. Confirmed rows expand to one row per resolved angle while duplicating the original detection index.

## Responsibilities

- Merges selected multi-source results with the complete baseline detection stream while preserving original row order.
- Duplicates detection indices for confirmed resolved sources and emits matching angle rows plus source-slot metadata.
- Does not decide which rows require multi-source processing or estimate any directions itself.

## When to use

Use after selective multi-source estimation to merge resolved rows back with the original detection stream and expand each reported source into its own row. It is appropriate when only a subset was routed through an expensive estimator but downstream consumers require one uniform detection/angle table. Do not use it when each input item has only the baseline direction or when selected-row alignment is unavailable.

```text
original detection_idx + baseline_aoa
selected_idx + resolved_aoa + doa_multi + source_count
  → MultiSourceExpansionOp
  → detection_idx_out + aoa_estimates_out
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `detection_idx` | int64 | `(num_items, num_index_columns)` | Required trigger/aligned input | Original row identity; 2–10 columns preserved verbatim. |
| in | `baseline_aoa` | float32 / float64 | `(num_items, 2)` | Required trigger/aligned input | One fallback angle per original row. |
| in | `selected_idx` | int64 | `(num_selected,)` | Required trigger/aligned input | Strictly ascending, unique original-row indices. |
| in | `resolved_aoa` | float32 / float64 | `(num_selected, 2)` | Required trigger/aligned input | Dominant angle from the selected estimator. |
| in | `doa_multi` | float32 / float64 | `(num_selected, source_slots, 2)` | Required trigger/aligned input | Source list; `source_slots <= max_sources`. |
| in | `source_count` | int32 | `(num_selected,)` | Required trigger/aligned input | Valid sources in each selected estimator row. |
| out | `detection_idx_out` | int64 | `(num_output, num_index_columns)` | Required output | Original index repeated for every emitted source. |
| out | `aoa_estimates_out` | float32 / float64 | `(num_output, 2)` | Required output | Baseline or resolved angles aligned with output indices. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_items`, `max_sources`, and `max_points` required | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| None | — | — | — | This operator has no Params; behavior is set by constructor captures and inputs. |

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

## Performance

Planning is `O(num_items + num_selected × source_count)` and scatter is `O(num_output × num_index_columns)`. One synchronization validates and exact-sizes the output.

`performance/operators/detection/multi_source_expansion/adapter.cpp` sweeps original rows, selected fraction, sources per selected row, and index width while draining both outputs and reporting warmup, setup/cold, and steady timing.

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_multi_source_expansion_smoke
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
