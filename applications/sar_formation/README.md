# SAR formation

Mode-1 spotlight: one collection, one image, dumped as `.npy` and
shown in Holoviz when `visualize: true`. The dump is also the offline
PNG checkpoint (`tools/sar_image/form_image.py`).

```text
CPHDReaderOp → SARRangeCompressionOp → BackprojectionOp → TensorDumpOp
                                              └→ SarImageToLutOp → HoldLastImageOp → HolovizOp
reader.positions / reference_position ─────────────▲
StreamBoundaryControlOp (app-owned) ───────────────▲
```

`range_bins` is not wired. Mode 1 is the full centred window, so
`BackprojectionOp.range_bin_spacing` (from CPHD `SCSS` and
`num_samples`) and the centred origin default are enough. RC has no
waveform, so it does not emit the axis.

The **demo** reads the pinned Umbra Atlanta collect (28.59 GiB) and
forms the 1° closest-approach frame onto the NVRadar spaceborne
Atlanta lattice: 8192² IARP ENU, 0.699 m, ~5.73 km, sample `(0, 0)`
at the north-west pixel. The dump `.npy` is that lattice
`(num_v, num_u)` with `v` increasing south and `u` east. The PNG
is that array with no resample. Matches forced onto the NVRadar
example — and how to walk them back — are in
[NVRADAR_PARITY.md](NVRADAR_PARITY.md). The generated point-target
fixture is smoke / CI only.

## Atlanta (mode 1 demo)

Download once. The app streams a local file and never reaches the
network; `pulse_range` seeks the frame.

```bash
mkdir -p /tmp/sar
curl -L -o /tmp/sar/2026-02-20-03-03-38_UMBRA-07_CPHD.cphd \
  https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/sar-data/task-data/0f1231be-26d3-4b98-96ab-f675a5375c14/2026-02-20-03-03-38_UMBRA-07/2026-02-20-03-03-38_UMBRA-07_CPHD.cphd

./build/applications/sar_formation/cpp/sar_formation

python3 tools/sar_image/form_image.py \
    --image /tmp/sar/2026-02-20-03-03-38_UMBRA-07_CPHD_image_0.npy \
    -o /tmp/sar/atlanta.png
```

`mode01.yaml` sets `visualize: true`. The window is greyscale
log-magnitude (same 30 dB / 99.8-percentile window as the PNG),
held for `visualize_hold_s` seconds after the image emits so it
does not close with the graph. On a host with no display the app
warns and skips the window. `headless: true` renders Holoviz
offscreen. The dump is full-resolution complex; Holoviz windows on
full-resolution `|z|` then box-averages the longest side to 2048.

`mode01.yaml` (staged next to the binary) names that path, the 1°
`pulse_range` trimmed to their 11500-pulse batch (`[91273, 102773)`),
and that ENU lattice. The reader
still emits ECEF; the app rotates APC / SRP into scene ENU at the
IARP so the kernel grid and positions share a frame. The graph is
`double`. Override the path with a CLI argument if the file lives
elsewhere. Re-render the dump with
`tools/sar_image/form_image.py --image …npy`. The PNG is a 1:1
greyscale raster of the formed samples (8192² here), windowed 30 dB
below the 99.8th-percentile magnitude so a specular does not crush
the scene. No axes or pad. Row 0 of the dump is north; this tool
does not flip it.

## Smoke fixture

```bash
python3 tools/sar_sim/generate_cphd.py \
    tools/sar_sim/recipes/mode01_smoke.yaml -o /tmp/sar

./build/applications/sar_formation/cpp/sar_formation \
    mode01_smoke.yaml /tmp/sar/mode01_smoke.cphd
```
