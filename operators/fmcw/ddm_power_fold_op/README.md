# DDMPowerFoldOp

`DDMPowerFoldOp<T>` prepares the detect-before-demultiplex sparse DDM path. It
maps every linear DDM TX band to the common reduced Doppler axis and sums power
over all TX/RX looks without allocating a dense `(M_tx,N_rx,W,R)` cube.

## Contract

- Input `rd_maps`: device complex `(N_rx,D,R)`.
- Output `power_map`: device real `(D/M_tx-guard_bands,R)`.
- Required static `ddm_code`: linear, exact-bin, equal-width M-band layout.

The gather table is shared with `DDMDenseDemultiplexOp`; therefore guard
exclusion and nonuniform band permutations are identical. Nonlinear codes,
fractional or non-M-band offsets, duplicate bands, nondivisible `D`, and guard
widths that remove the whole band are rejected. CFAR should use
`nci_axes=[]` and `preintegrated_looks=M_tx*N_rx`. That Gamma model is exact
only for independent looks; correlated channels need empirical calibration.
Discarding `G=guard_bands` bins from each width-`B=D/M_tx` band creates a
blind-velocity fraction `G/B`; applications should choose `G` from the
slow-time window's main-lobe and transition width rather than treating it as
free sensitivity tuning.

Both float and double specializations run entirely on the CUDA stream.
