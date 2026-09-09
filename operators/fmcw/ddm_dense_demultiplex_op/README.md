# `DDMDenseDemultiplexOp<T>`

Separates simultaneous-TX DDM maps from `(N_rx, D, R)` into an explicit TX axis. Linear offsets or linear phase codes use the preserved gather path; nonlinear phase codes use the preserved full-axis conjugate-correlation path.

The only parameter is static, required `ddm_code`; no waveform or multiplexing switch is exposed. Linear output shape is `(M_tx, N_rx, D/M_tx-guard_bands, R)` and nonlinear output keeps `D`. The implementation reuses `radar_common/dsp/mimo_coherent_compensation_stage.*`.

Contract: [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) · group: [`operators/fmcw/README.md`](../README.md)
