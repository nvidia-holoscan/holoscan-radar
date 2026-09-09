# `TDMDenseCompensationOp<T>`

Applies firing-slot-aware, per-Doppler-bin TDM inter-TX phase correction to `(M_tx, N_rx, D, R)` complex maps. Output `rd_maps_out` preserves shape and dtype.

Parameters: required `waveform` (`[live]`, Exact, one tick) and static `tx_schedule` (empty means identity round-robin). The implementation reuses `radar_common/dsp/mimo_coherent_compensation_stage.*`; correction tables are cached by waveform, shape, and firing slots.

Contract: [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) · group: [`operators/fmcw/README.md`](../README.md)
