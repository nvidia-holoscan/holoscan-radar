# SAR formation

Mode-1 spotlight: one collection, one image, shown in Holoviz when
`visualize: true`. Mode 2 is video SAR (sliding window). Mode 5 is
N independent images. `dump: true` writes a `.npy`
(off by default); that dump is the offline PNG checkpoint
(`tools/sar_image/form_image.py`).

```text
CPHDReaderOp → SARRangeCompressionOp → ApertureManagerOp → BackprojectionOp
                                              [→ TensorDumpOp]
                                              [→ SarImageToLutOp → HoldLastImageOp → HolovizOp]
                                              [→ TimelineRasterOp → HolovizOp]
reader.positions / reference_position ─────────────▲
```

Modes 1 and 5 wire `ApertureManagerOp` with `max_aperture_pulses: 0`,
`emit_interval_pulses: 0`, `emit_mode: reset` (one image per
collection, no retention). `weight` stays unconnected (all `+1`).
Mode 2 retains a sliding window (`max_aperture_pulses > 0`,
`emit_mode: continue`) and wires `manager.weight` so eviction is
`−1`. Mode 5 inserts `CollectionBoundaryStagerOp` on the
`range_profiles` edge so the next lattice is staged into the
manager on `collection_boundary`, and `range_bin_spacing` /
`reference_frequency` still go straight to `BackprojectionOp`
(those are not manager ports).

`range_bins` is not wired. `BackprojectionOp.range_bin_spacing`
is derived from CPHD `SCSS` and `num_samples`, and from
`range_compression.nfft` when that key is set, or from app-owned
`nfft_pad: next_fast` (mode 5: smallest 5-smooth length covering
every collect). Mode 1 leaves both unset (native N). RC has no
waveform, so it does not emit the axis.

The app uses Holoscan's `EventBasedScheduler` with a shared worker
pool sized to the operator count (one worker per operator, not a
pinned thread each). That lets the reader load batch *N+1* while
backprojection is still in `compute()` on batch *N*. Tests and the
perf harness keep the default single-thread `GreedyScheduler`.

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

# Offline PNG from a dump (set dump: true in the YAML first):
python3 tools/sar_image/form_image.py \
    --image /tmp/sar/2026-02-20-03-03-38_UMBRA-07_CPHD_image_0.npy \
    -o /tmp/sar/atlanta.png
```

`mode01.yaml` sets `visualize: true`. The window is greyscale
log-magnitude, 0 dB at the displayed grid's 99.8th-percentile
magnitude, floor at its 25th percentile (never wider than 30 dB).
A small N-up compass (bottom-right) and a border chevron mark
the mean phase-centre direction (row 0 = north, col 0 = west).
Holoviz is given those 8-bit codes as `R8_UINT` (COLOR_LUT
indexes 0–255; a float raster was the washed live window).
`pass_through` is not available on this display. Held for
`visualize_hold_s` seconds after the image emits so it does not
close with the graph. On a host with no display the app warns
and skips the window.
`headless: true` renders Holoviz offscreen. `dump: true` writes the
full-resolution complex lattice; omit `output.path` to put it next
to the CPHD as `<stem>_image_{index}.npy`. Holoviz box-averages
`|z|` to the window (`visualize_size`) and windows that grid on
the GPU so the raster is 1:1 (`R8_UINT` cannot bilinear). The PNG
path still windows the full dump.

`mode01.yaml` (staged next to the binary) names that path, the 1°
`pulse_range` trimmed to their 11500-pulse batch (`[91273, 102773)`),
and that ENU lattice. The reader
still emits ECEF; the app rotates APC / SRP into scene ENU at the
IARP so the kernel grid and positions share a frame. The graph is
`double`. Override the path with a CLI argument if the file lives
elsewhere. Re-render the dump with
`tools/sar_image/form_image.py --image …npy`. The PNG is a 1:1
greyscale raster of the formed samples (8192² here). Each file
windows on its own histogram (99.8th percentile at 0 dB, 25th
percentile as the floor, never wider than 30 dB) so a specular
does not crush the scene and a low-contrast collect does not look
washed. No axes or pad. Row 0 of the dump is north; this tool
does not flip it.

## Smoke fixture

```bash
python3 tools/sar_sim/generate_cphd.py \
    tools/sar_sim/recipes/mode01_smoke.yaml -o /tmp/sar

./build/applications/sar_formation/cpp/sar_formation \
    mode01_smoke.yaml /tmp/sar/mode01_smoke.cphd
```

## Mode 2 — Houston video SAR

Full 23.5 s downtown Houston dwell as a sliding window (INPUT_DATA.md).
The app streams a **local** central-subband CPHD; subset once, never
the network. Live YAML is ~1.0 m ground-range GSD (~27 GiB) after
0.5 m ran ~4× behind collect time on this GPU.

```bash
python3 tools/sar_sim/subset_cphd.py \
    tools/sar_sim/recipes/mode02_houston_1p0m.yaml \
    --source /tmp/sar/2024-10-30-16-10-33_UMBRA-06_CPHD.cphd \
    -o /tmp/sar

./build/applications/sar_formation/cpp/sar_formation mode02.yaml
```

`mode02.yaml` retains 8262 pulses (`2020 × 4.09 / 1.0`, the 4.09 m
window scaled), emits every 4096 pulses (~2.18 Hz at PRF 8942 Hz;
four 1024-pulse reader ticks, `emit_mode: continue`),
and wires `ApertureManagerOp.weight` so the trailing edge leaves at
`−1`. No `aperture_window`: a sliding window is unweighted (~−13 dB).
The scene is 6.8 km (`lattice.counts` × `spacing` at 1.0 m).
`lattice.max_side: 6144` is the formed pixel side (~1.11 m GSD);
**that is the number to lower on a slower GPU** (same km, coarser
pixels, cost ~ N²). `visualize_size` is only the Holoviz window.
Watch the overlay's `behind` line: if it climbs, drop `max_side`.
`nfft_pad: next_fast`, `playback_rate: 1`, `loop: true` (wrap is
`collection_boundary` / emit-and-reset; the window refills over
the next ~0.92 s), `visualize_hold_s: -1`.

Two Holoviz windows: formed `|z|` and the timeline. The
formed-image overlay and the timeline show collect-vs-wall lag
when playback is paced (`on time` / behind / ahead). Modes 1
and 5 omit that overlay. A small N-up compass (bottom-right)
and a border chevron mark the mean phase-centre direction
relative to the scene (row 0 = north, col 0 = west). Dump stays
the formed complex lattice.

0.5 m is the same extent at twice the samples and four times the
pixels: `/tmp/sar/mode02_houston_0p5m.cphd` (recipe
`mode02_houston_0p5m.yaml`, 33505 samples), lattice 13600² at
0.5 m, `max_aperture_pulses: 16524`.

## Mode 5 — six unique-IARP spotlights

N independent images on the same spine as mode 1. The reader
`loop`s the settled six Umbra SNAPSHOTs (Panama Canal, Melbourne,
Bingham Copper Mine, Kourou, Buenos Aires, Intel Ohio One). Each
seam is `collection_boundary`; Holoviz keeps the last emit until
the next image replaces it. `visualize_hold_s: -1` holds the window
until SIGINT. `playback_rate: 1` paces to collect time (sleep
while ahead of radar; no sleep while behind). `0` is
unconstrained. The collect-vs-wall lag overlay is mode 2 only.
Optional
`duty_cycle` < 1 inserts a cooldown
`T_c · (1/D − 1)` between collections. `pulses_per_tick: 1024`
is the mode-5 batch (Ohio is 22 ticks); 100 was NVRadar Atlanta
parity, not a mode-5 constraint.

Pull first (~10.5 GiB). The script skips files already present
and refuses the set if two IARPs collide (read from each file's
XML).

```bash
python3 applications/sar_formation/pull_cphd.py --dest /tmp/sar/mode5

./build/applications/sar_formation/cpp/sar_formation mode05.yaml
```

Each scene is formed onto the largest IARP-centred east-north
square that fits that collect's ground-range × ground-azimuth
support, sampled at the coarser of the two GSDs (range
`(c/2B)/cos ψ`, azimuth `λ/(2Δθ)` from the first/last valid
`TxPos`). `lattice.max_side: 0` keeps native N (Ohio ~11848²);
a positive cap coarsens GSD on the same square.
`lattice.pixel_scale` multiplies native N per collect (list
aligned with `cphd.path`, or one scalar for all): `1` is native,
`0.5` is half the side (quarter the pixels), same square.
Explicit `counts` / `spacing` still force an IARP-centred ENU grid.
Holoviz (`visualize_size`, default 1024) box-averages that
formed image to the window so R8_UINT is 1:1. `dump: true` writes
the physics lattice as `.npy` (`output.path`, `output.max_files`);
it is off by default because a native-N dump is seconds of D2H
(Ohio ~2 GiB / 4.6 s). `nfft_pad: next_fast` sets the IFFT
length to the smallest 5-smooth integer covering every collect
(Ohio 22499 → 22500, not the next power of two); `load_config`
mirrors it into every collect's `range_bin_spacing`. Holoviz
windows the displayed grid on the GPU; the PNG path still
windows the dump. The overlay is the scene name from
`cphd.names` with the formed lattice `H x W` on the same line,
then the host time spent in `BackprojectionOp` for that image
and the implied `pixels * pulses / s`. A second Holoviz window
(`SAR timeline`) is a 20 s scrolling wall-clock Gantt: dwell bars
on the platform lane (one color per collect), a matching tick on
the product lane at each image emit with the last-packet→image
latency beside it, and duty-cycle cooldown as a dimmer bar when
`duty_cycle` < 1 (mode 5 default `1` leaves that empty). After
GLFW maps it, the strip is moved to the bottom-left of the
desktop work area (X11 / XWayland; native Wayland is a no-op).
Holoviz (and a PNG of the dump) resets the dB window on every
emit from that image's own histogram.
CI stays on generated fixtures; this YAML is not a CTest.
