# BackprojectionOp — open items

This operator landed as a **mode-1 slice**. It does not clear
[`NEW_OPERATOR_CHECKLIST.md`](../../../docs/planning/NEW_OPERATOR_CHECKLIST.md).
Do not resolve team-blocked items here.

## Implementation follow-ups

1. **OSRB relicensing of copied NVRadar kernels.** The CUDA kernels in
   `nvradar_kernels.cuh` are transcribed from NVRadar
   (`bp_kernels.py` at SHA `37c1cb29355bcfe256ae728f51dd47071847d949`) and
   carry Apache-2.0 in this tree pending OSRB. Implementation did not wait.

2. **`nvradar_*` copies.** All 18 CUDA backends are copied and dispatched.
   Default is `nvradar_block_anchored_expanded_throughput_phase_lut_vec`.
   LUT fill is NVRadar `build_phase_lut` (`pc_partial * (rmin + b·dr)`),
   not MatX's. `nvradar_block_anchored_expanded_throughput_phase_lut_vec_z0`
   ignores `z` (planar scenes only).

3. **MatX wrap (`matx_*`).** Wrapped by reference when
   `matx/kernels/sar_bp.cuh` is on the include path (this operator's
   TU is C++20 for that header). `matx_taylor_fast` still throws until
   the header exposes `TaylorFast`. Even-`N` origin/phase on `matx_*`
   is half a bin wrong until the MatX pin includes
   [MatX #1226](https://github.com/NVIDIA/MatX/pull/1226) (`b568960`).
   Tests accept a 1-pixel neighbor on even `N`. Do not add an `R0` or
   profile-shift compensation.

4. **I1 (use of bundled `sar_bp`).** The wrap uses the SDK's bundled
   MatX. A C++20 bump, if this translation unit needs it, is local to
   this operator — not a MatX pin bump. Still experimental.

5. **`initial_image` and `geometry_change = scroll`.** Declared and
   rejected this slice.

6. **`sinc` / `nearest` interpolation.** Named in the enum, rejected.

7. **Uniform-lattice expand is on device.** Host `expand_sample_lattice` is
   tests and small lattices. The formation path expands
   `SampleGeometry::sample_position` on the GPU: NVRadar writes the x/y/z
   planes directly (no packed 1.61 GiB host buffer / H2D at Atlanta
   8192x8192 double); MatX still gets packed xyz, also expanded on device.
   `sample_points` remains a device-to-device copy of the caller tensor.

## Not ours to close

- **R10** — metadata key namespace. This operator consumes uint8
  `control` only and does not read `stream_boundary`.
- **I5** — range-compression normalization default. Lives on
  `SARRangeCompressionOp`.
