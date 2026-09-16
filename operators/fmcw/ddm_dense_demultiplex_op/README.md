# `DDMDenseDemultiplexOp<T>`

Separates simultaneous-TX DDM maps from `(N_rx, D, R)` into an explicit TX axis. Linear offsets or linear phase codes use the reduced gather path; nonlinear phase codes use the full-axis conjugate-correlation path. An optional physical velocity axis is transformed alongside the map.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(N_rx, D, R)` | Center-zero coded DDM map; trigger |
| in | `velocity_bins` | float32 / float64 | `(D,)` | Optional Exact physical axis matching `rd_maps` |
| out | `rd_maps_out` | complex64 / complex128 | `(M_tx, N_rx, D_out, R)` | Required demultiplexed map |
| out (cond.) | `velocity_bins_out` | float32 / float64 | `(D_out,)` | Optional/latched; emitted only when the input axis is present |

The only parameter is static, required `ddm_code`; no waveform or multiplexing switch is exposed. Linear output shape is `(M_tx, N_rx, D/M_tx-guard_bands, R)` and nonlinear output keeps `D`. For the linear path, `velocity_bins_out` is the centered `D_out`-element slice of the input axis and keeps its physical spacing; for nonlinear correlation it is an identity copy. The operator does not unfold the reduced DDM velocity ambiguity.

`contract()` triggers on `rd_maps`; optional `velocity_bins` joins by `Exact{} + SafeDefault{}`. Both active output loans are prepared before publication, then `velocity_bins_out` is emitted before `rd_maps_out`. That causal order is load-bearing when the map immediately triggers a downstream consumer with the axis as an Exact sideband. A declined `velocity_bins_out` emit is absorbed and latches off its later copy work without suppressing `rd_maps_out`.

Construction requires `max_tx`, `max_rx`, `max_doppler_bins`, and `max_range_bins`. The map pool uses the full captured cube bound and the axis pool uses `max_doppler_bins` real elements. Inputs and outputs are CUDA-device tensors; the source map and optional axis are retained across the asynchronous stream work.

The implementation reuses `radar_common/dsp/mimo_coherent_compensation_stage.*`.

Contract: [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) · group: [`operators/fmcw/README.md`](../README.md)
