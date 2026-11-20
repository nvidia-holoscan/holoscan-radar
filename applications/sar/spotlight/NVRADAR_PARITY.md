# Atlanta vs NVRadar — forced matches

Historical source-branch walk-back list for the spotlight Umbra Atlanta demo against NVRadar's
`spaceborne-spotlight-sar` example (same CPHD, same lut-vec kernel).
This is a record of source-branch parity experiments, not an HSDK 5
runbook. The HSDK 5 app has not been validated on the real Atlanta file. That is what tells us which match was load-bearing.

NVRadar tree is `/tmp/nvradar` and is not in this repo. Their
converter + example: `examples/synthetic-aperture-radar/spaceborne-spotlight-sar/`.

Do not treat this file as a contract. The SAR contract is
[`docs/planning/SAR/OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md).
Nothing here authorizes changing it.

## Still on (walk these back)

1. **Image grid = their `meshgrid`.** `spotlight.yaml` `sample_geometry`:
   8192², `ps = 0.6992513927571693` m, IARP ENU, `z = 0`, origin =
   NW pixel centre `(-4095.5, +4095.5)·ps`, `axis_v = south`,
   `axis_u = east`. Verified `|Δ| = 0` vs their `xx`/`yy`/`z=0`.
   **Undo:** restore the 5 km ECEF lattice (origin = IARP − 2500 N −
   2500 E, `spacing = 5000/8191`, `axis_v` = north ECEF, `axis_u` =
   east ECEF). Must undo (2) in the same step or APC stays ENU on an
   ECEF grid.

2. **APC / SRP rotated into that ENU.** `enu_origin_ecef` in
   `spotlight.yaml` inserts `EcefToSceneEnuOp` (Bowring WGS-84 + ECEF→ENU
   at the IARP, same as their converter). Reader stays ECEF; BP
   `coordinate_frame` is `enu`. Pulse-0 ENU matches their `.dat` xyz
   after a float32 cast.
   **Undo:** drop `enu_origin_ecef` and set `backprojection.coordinate_frame: ecef`.
   The op in `custom_ops.hpp` can stay; it is inert without the YAML key.

3. **`phase_center` rounded through float32.** `quantize_phase_center_f32`
   in `spotlight.yaml`. After the rotate, each APC component is
   `double(float(x))` so the kernel sees the same bits as their `.dat`
   xyz (tensor dtype stays float64 because the app is `double`).
   Reference / SRP is not rounded. Side effect: `R0 = ‖APC − SRP‖`
   then equals their kernel `‖xyz_f32‖` (SRP is the origin), which is
   the *consistent* pair — not the earlier failed “their R0 + our
   double APC” mix.
   **Undo:** set `quantize_phase_center_f32: false` (or drop the key).

4. **`Np` trimmed to 11500.** `pulse_range: [91273, 102773)` — same
   start as their 1° keep, last 66 pulses dropped the way
   `run_pipeline.py` aligns to `pulse_batch_size=100`.
   **Undo:** restore `[91273, 102839)`.

5. **100-pulse batches.** `cphd_reader.pulses_per_tick: 100` matches
   their `pulse_batch_size`. Same 11500 pulses; launch shape only
   (`115 × 100` instead of `44 × 256 + 236`).
   **Undo:** restore `pulses_per_tick: 256`.

6. **Their kernel `vph` — off as of 2026-08-24.** Contract RC now
   `ifftshift`s native FX (IFFT DC = stored sample `N/2`) and
   `reference_frequency` is that sample, not `SC0`. Their conditioner
   `vph` still pairs with `SC0`; leaving `nvradar_vph_path` set would
   be the inconsistent pair. Re-enable only with `f_ref = SC0`.
   File is not in git.
   **Historical experiment (not wired in the HSDK 5 app):** the source branch
   could set `nvradar_vph_path` and restore `reference_frequency` to `SC0`.

Items 1–2 did **not** clear the ghost. Item 3 did not move the peak.
Item 4 same peak `(3437, 4726)`, mag 728.6 → 729.0, mean |Δ| 0.087.
Item 5 same peak, max |Δ| 2×10⁻⁴ vs the 256-batch dump (accum order).
Item 6 moved the peak `(3437, 4726)`/729 → `(4073, 5709)`/2.09×10⁵
(~286×; √N ≈ 193 is only the ortho scale). PNG:
`/tmp/sar/atlanta-nvradar-vph.png`.

**Walk-back 1+2, item 6 kept.** Old 5 km ECEF lattice, no ENU rotate,
same 11500 / 100-pulse `vph`. PNG `/tmp/sar/atlanta-ecef-vph.png`.
Peak `(4842, 4823)` / 2.23×10⁵ — same pixel as the early
ifftshift-on-ECEF trial, now at ortho scale (that run was 1160 ≈
2.23×10⁵ / √N). Scene is a focused airport, not ghosts. The
NVRadar meshgrid is **not** required for a good image; the `vph`
shift is.

**Walk-back 1–5, item 6 kept.** Original 5 km ECEF lattice, 256-pulse
batches, full 1° window `[91273, 102839)` (11566), no ENU, no f32
APC. `vph` file extended to all `.dat` pulses
(`/tmp/sar/nvradar-vph-11566.c64`). PNG
`/tmp/sar/atlanta-shift-only.png`. Peak still `(4842, 4823)`, mag
2.24×10⁵ vs 2.23×10⁵ on the 11500-pulse ECEF+vph form. Scene
stays a focused airport. Items 1–5 are not required.

## Already the same (do not "undo" — they were not forced)

These agreed before the grid work. Reverting them would be a new
divergence, not a walk-back.

- Kernel: `nvradar_block_anchored_expanded_throughput_phase_lut_vec`
- `Nfft = 37268` (no pad). Their kernel `minf` is `SC0` with their
  `vph`; our `reference_frequency` is stored sample `N/2` with
  contract RC. Same rotate, different carrier spelling.
- `dr` / `rmin` = natural `c/(2·df·N)`, centred `−⌊N/2⌋·dr`
- LUT: `φ[b] = pc_partial·(rmin + b·dr)`
- `R0` formula `‖APC − MCP‖` (`lut_vec` does not take MCP)
- 1° keep *start* `91273` (their converter cone). Length is item 4.
- SRP frozen at the IARP on this collect (their MCP = 0 is the same point)
- `vph` layout: interleaved complex64, pulse-major
- `phase_center` layout: planar `(x[], y[], z[])`, stride `Np`

## Still different (candidates once the image is clean)

- **`vph` radiometry (`fft_norm`)** — contract RC now `ifftshift`s too
  (2026-08-24), so the remaining factor is `√N` (`ortho` vs `backward`).
  I5. Not the ghost.
- **`phase_center` / `R0` precision** — only while item 3 is off.

## Tried and reverted (not in the tree)

Do not put these back unless a new diagnostic asks for them.

- **Extra `ifftshift` in `SARRangeCompressionOp` before the 2026-08-24
  contract change** (their conditioner on our old ECEF lattice, with
  `f_ref` still `SC0`). Peak moved (4109, 5952)/828 → (4842, 4823)/1160.
  The rotate is now the contract; pair it with stored sample `N/2`, not
  `SC0`.
- **Force kernel `R0 = ‖float32(APC)‖`** while leaving APC float64
  (`match_nvradar_r0` on the ENU path). Worse: peak mag 728 → 42,
  image defocused. Their `R0` is paired with *their* quantized APC;
  mixing the two is an inconsistent `(phase_center, R0)` pair.

## Walk-back order (suggested)

Once a form matches their PNG:

1. Item 6 is already off (contract RC + `N/2` `f_ref`).
2. Turn off item 5 (`pulses_per_tick: 256`) alone.
3. Turn off item 4 (restore `[91273, 102839)`) alone.
4. Turn off item 3 (`quantize_phase_center_f32: false`) alone.
5. Restore the 5 km ECEF lattice **and** drop the ENU rotate (items 1+2 together).
6. If that breaks it, put only the ENU rotate back (ECEF numbers, their
   crop/spacing) — that splits “frame” from “pixel pitch / row 0”.
