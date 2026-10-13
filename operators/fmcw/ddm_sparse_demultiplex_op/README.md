# `DDMSparseDemultiplexOp`

## Purpose

Gathers detected folded DDM cells, separates TX bands, applies code and ambiguity-fold correction, and assembles virtual snapshots. It is the sparse counterpart to dense demultiplexing, virtual assembly, and snapshot gathering.

## Responsibilities

- Owns linear band lookup, fold-aware TX permutation, phase removal, neighborhood gathering, and geometry.
- Keeps snapshot, validity, and geometry ordering coherent.
- Does not support nonlinear/fractional-bin codes, infer `fold_idx`, run CFAR, or suppress clutter.

## When to use

Use after `DDMPowerFoldOp` and CFAR when only detected DDM cells need TX separation and virtual snapshots. Supply `fold_idx` when targets may lie outside the principal reduced velocity interval. Use dense demultiplexing when the full separated cube or physical-velocity filtering is required; do not use this operator for TDM.

```text
rd_maps + folded detections [+ fold_idx] -> DDMSparseDemultiplexOp -> snapshot + geometry
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(N_rx,D,R)` | Required trigger/aligned input | Pre-demultiplex DDM maps. |
| in | `detection_idx` | int64 | `(N,2)` | Required trigger/aligned input | Folded Doppler/range rows. |
| in (optional) | `fold_idx` | int64 | `(N,)` | Optional Exact side input; `SafeDefault{0}` | Ambiguity integer in `[0,M_tx)`. |
| live | `wavelength` | float64 scalar | scalar | Optional Exact one-tick override; Params fallback on miss | Wavelength for geometry. |
| live | `snapshot_window` | `DDMSparseWindowOverride` | POD `{doppler, range}` | Optional Exact one-tick override; Params fallback on miss | Folded neighborhood half-extents. |
| out | `snapshot` | complex64 / complex128 | `(N,num_virtual,num_snapshots)` | Required output | Demultiplexed virtual snapshots. |
| out (optional, always) | `is_valid` | uint8 | `(N,num_snapshots)` | Optional `always` output | Edge-validity mask. |
| out | `array_geometry` | `WireArrayGeometry` | descriptor-Tensor | Required output | Virtual positions and wavelength. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`, `max_detections` or `max_items`, `max_snapshots`; optional `max_virtual_channels` | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `ddm_code` | `DDMCodeDescriptor` | none (required) | static | Linear integer-bin code layout. |
| `tx_positions` | `vector<double>` | none (required) | static | Row-major TX positions. |
| `rx_positions` | `vector<double>` | none (required) | static | Row-major RX positions. |
| `overlap_policy` | `OverlapPolicy` | `dedup` | static | Virtual collision handling. |
| `wavelength` | float64 | none (required) | `[live]` Exact | Positive geometry wavelength. |
| `snapshot_window` | `vector<int64_t>` | `{0,0}` | `[live]` Exact | Folded-surface half-extents. |
| `doppler_edge` | `SnapshotEdgeMode` | `wrap` | static | Folded-Doppler edge policy. |
| `range_edge` | `SnapshotEdgeMode` | `shift` | static | Range edge policy. |

Contract: [`OPERATOR_API.md` section `DDMSparseDemultiplexOp`](../../../docs/planning/OPERATOR_API.md) - group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

The linear DDM planner maps each folded coordinate and TX to a source bin. `fold_idx` cyclically permutes TX association for aliased targets, code phase is removed, and the neighborhood is gathered through the virtual-array plan. Zero fold is correct only in the principal reduced interval.

## Shape, dtype, and timing rules

Inputs are exact ranks three, two, and optionally one. Fold rows match detection rows and lie in `[0,M_tx)`. Codes must be linear and integer-bin aligned. Optional fold and live values join `detection_idx` with `Exact{}`. Geometry and mask publish before snapshot.

## Configuration and caching

DDM lookup/phase tables are cached by code and Doppler shape; virtual assembly is cached by positions and overlap policy. Live wavelength changes metadata only, while a live window changes exact shape within captured bounds.
