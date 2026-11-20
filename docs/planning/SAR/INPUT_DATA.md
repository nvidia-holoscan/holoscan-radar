# Holoscan Radar Module — SAR Input Data Plan

## Purpose

What data the SAR wave's tests and demos run on, where it comes from, and what it costs to obtain. The contract lives in [OPERATOR_API.md](OPERATOR_API.md), the chains in [WORKFLOWS.md](WORKFLOWS.md), the assertions in [TESTING.md](TESTING.md), and the sequencing in [SIGPROC_PLAN.md](SIGPROC_PLAN.md); this document is the data those depend on.

**Status (HSDK 5, 2026-09-29):** The C++ suite generates its small CPHD
fixture offline. Optional [data preparation tools](../../../tools/sar_data/README.md)
now provide descriptive manifests and the Houston subset recipe for the
shipped five-app real-data presets. Public collections and interactive X11
have not been validated on this worktree. The source-branch planning and data
surveys below remain historical; use each [app guide](../../../applications/sar/README.md)
for current commands. Team-blocked questions remain open in
[TEAM_BLOCKED.md](TEAM_BLOCKED.md). Where this document and [TESTING.md](TESTING.md) disagreed on simulator ownership or fixture policy, the decisions below are the later ones; TESTING.md was amended to match on 2026-08-06, so the two now agree.

## Two sources, two jobs

Input data serves two purposes that pull in opposite directions, so it comes from two places rather than one compromise.

| Source | Job | Why not the other source |
| :--- | :--- | :--- |
| **In-repo simulator** | Metric assertions — resolution, geolocation, phase, accumulation | Ground truth is known by construction; real collections have no truth to assert against beyond a delivered product |
| **Public collections** | Convention correctness and demos | A synthetic test generated under the implementation's own assumption validates the assumption into existence, not the code |

The second row is the same argument [SIGPROC_PLAN.md](SIGPROC_PLAN.md) makes for the deramped branch being evidence-gated, applied to input data generally.

## Policy

**No CPHD binaries in git.** Every fixture — CI smoke and demo alike — is generated from a committed recipe at test or run time. Committing generated binaries is team-anti-preferred, and the point-target fixtures generate fast enough that committing them buys nothing.

**The simulator lives in this repository and is public.** An external simulator cannot be a dependency of a public module when users are expected to generate their own test input, and the NVRadar simulator is not public. This supersedes [TESTING.md](TESTING.md)'s "first-class test dependency living outside this repository" and its pinned-version/bit-for-bit-regeneration requirement, which existed to make an external dependency safe.

**Two scene modes, chosen by what the data is for.** A point-target scene for CI and metric assertions, generating in seconds; an image-derived scene for demos, with a hard cap on reflector count so generation stays within a few minutes. The realism a demo needs and the known truth a test needs are different requirements, and one scene model serving both would be slow for CI and too sparse to look like anything.

**`sarpy` for CPHD writing**, in both the simulator and the public-data subsetter. It confines a Python-only dependency to the generate/pull path — no operator, test binary, or application links against it.

## Public collections, per workflow

All are credential-free: a plain HTTPS or `aws s3 --no-sign-request` pull, no account. Scope is the first ship — spotlight, video SAR, tiled strip, continuous strip, and independent images plus opt-in autofocus, per [WORKFLOWS.md](WORKFLOWS.md) → *Image-formation workflows*. Wide swath, sub-aperture looks, circular SAR, and 3D / tomographic volume are deferred and unpinned.

| Workflow | Collection | Full size | Status |
| :--- | :--- | :--- | :--- |
| spotlight image | Umbra, Hartsfield-Jackson Atlanta — the scene the MatX `sar_bp` example uses | 28.59 GiB | pinned, header verified |
| video SAR | Umbra, downtown Houston | 110 GiB | pinned, **fully validated** — see below |
| tiled strip | Capella stripmap, Imperial Valley CA — ~15.77 s dwell, CI4 with `AmpSF` (same file as continuous strip) | 3.452 GiB | pinned, header verified 2026-08-30 |
| continuous strip | Capella stripmap, Imperial Valley CA — same collect as tiled strip; `continuous_strip.yaml` scrolls a 4096-row window | 3.452 GiB | pinned, header verified 2026-08-30 |
| independent images | six unique-IARP Umbra CPHDs (see below). Capella is **not** required | ~10.5 GiB | **settled 2026-08-25** |

The two Capella pins carry weight beyond their modes: they are the only `CI4` collections here, and the only ones with `AmpSF` present, so they are what covers the reader's integer-sample and per-vector-scale paths. They also split CPHD minor version (1.0.1 vs 1.1.0) and are `SGN` +1 with travelling `SRPPos` (`SRPFixed=false`). Both Umbra collections are `CF8` without `AmpSF`, `SGN` −1, frozen SRP.

### Spotlight — Atlanta

Bucket `umbra-open-data-catalog` (us-west-2), key:

```text
sar-data/task-data/0f1231be-26d3-4b98-96ab-f675a5375c14/2026-02-20-03-03-38_UMBRA-07/2026-02-20-03-03-38_UMBRA-07_CPHD.cphd
```

`CPHD/1.1.0`, `CF8`, 102,839 vectors x 37,268 samples, 42.01 deg graze, IARP 33.6400 / -84.4300. The key is taken from the MatX `sar_bp` example README rather than transcribed, so "the scene that example images" is exact rather than approximate — which matters because this collection is also the parity reference for `BackprojectionOp`.

## Independent images — unique-IARP Umbra collects

Independent images is N independent products, so the demo has to *look* like a catalogue of scenes, not a double-run of spotlight. Listing the same CPHD twice proves `collection_boundary` and proves nothing a viewer can see.

**Demo set (settled 2026-08-25):** six real Umbra CPHDs, one IARP each. All `CF8` / FX / `SGN` −1 / `SRPFixed=true`, all `SNAPSHOT` / `UMB-SPOTLIGHT-35-1`. This is the first-ship pin; do not swap a collect without a broken key or an explicit reopen. Tanna Island and David Fuentes were dropped as too small. Atlanta stays on spotlight — a 16.63 s / 28.59 GiB collect is a different product class, not a larger catalogue tile. Houston stays on video SAR.

Dwell is the **radar collect**, not the download: Umbra `METADATA.json` `collects[].startAtUTC`–`endAtUTC`.

| Scene | IARP | Size | Dwell | Vectors × samples |
| :--- | :--- | ---: | ---: | :--- |
| Panama Canal | 8.97°N, 79.58°W | 801 MiB | 2.0 s | 9693 × 7218 |
| Melbourne | 37.85°S, 144.91°E | 1.07 GiB | 2.39 s | 9946 × 8704 |
| Bingham Copper Mine | 40.52°N, 112.15°W | 1.20 GiB | 2.0 s | 11134 × 9147 |
| Kourou, French Guiana | 5.23°N, 52.77°W | 1.74 GiB | 2.72 s | 13451 × 9682 |
| Buenos Aires | 34.60°S, 58.38°W | 1.93 GiB | 3.23 s | 17205 × 11700 |
| Intel Ohio One | 40.12°N, 82.71°W | 3.77 GiB | 5.0 s | 22475 × 16183 |

All six are snapshot spotlights. Intel Ohio One is the large-image slot: same SKU as the others, ~2× Buenos Aires in file size and ~1.5× in dwell. The site has a second IARP ~10 km west (40.08°N, 82.81°W, 2.54 GiB / 3.80 s); that cluster is **not** in the set — one plant, one IARP.

Keys (bucket `umbra-open-data-catalog`, smallest CF8 in each IARP cluster):

```text
sar-data/tasks/Panama Canal, Panama/2d482820-501d-4a21-9c8a-b0213637a4f4/2023-09-12-02-36-33_UMBRA-04/2023-09-12-02-36-33_UMBRA-04_CPHD.cphd
sar-data/tasks/Melbourne, Australia/0a23b3bc-96f2-48cc-8214-c10cef0d48f3/2023-07-14-23-20-06_UMBRA-05/2023-07-14-23-20-06_UMBRA-05_CPHD.cphd
sar-data/tasks/Bingham Copper Mine/b276284e-c8c6-4fb5-8aac-0916bc6ef4d6/2023-10-20-04-22-56_UMBRA-04/2023-10-20-04-22-56_UMBRA-04_CPHD.cphd
sar-data/tasks/Kourou, French Guiana/b40dc22c-c15f-4295-b71f-3a1db2007bbd/2023-02-22-12-54-56_UMBRA-04/2023-02-22-12-54-56_UMBRA-04_CPHD.cphd
sar-data/tasks/Buenos Aires, ARG/a7a03ef5-6bb6-4f85-aceb-638685a9b3e7/2023-02-08-02-11-21_UMBRA-05/2023-02-08-02-11-21_UMBRA-05_CPHD.cphd
sar-data/tasks/Intel Ohio One/95d6685e-a4af-4a87-ba0b-b4a43e636e81/2023-10-11-02-20-47_UMBRA-05/2023-10-11-02-20-47_UMBRA-05_CPHD.cphd
```

**A committed pull script ships with the demo.** The app still reads only local files; the user runs the script first. It lives under `tools/sar_data/pull_cphd.py`, takes `--app independent_images` (this manifest), or `--app tiled_strip` / `--app continuous_strip` (Imperial Valley, shared dest). Downloads over HTTPS / `aws s3 --no-sign-request`, skips files already present, and **on independent images refuses the set if two IARPs collide** (read from each file's XML, not from the folder name; collision is ECEF distance < 1 km).

**CI** stays generated: a handful of small simulator CPHDs with different IARPs and different target layouts. No CI job pulls public data.

**The visual checkpoint is Holoviz per collection.** Independent images `loop`s, so Holoviz replaces the last emit as each image forms (`visualize_hold_s: -1` was the source-branch hold request; indefinite post-run hold is not yet implemented in the HSDK 5 app). `dump: true` writes `.npy` (off by default); `output.max_files` caps a looping dump.

## Video SAR — the Houston collect, full dwell

**The video SAR demo uses the entire 23.5 s collect.** A short burst is cheaper but it undersells the mode: the value of video SAR over the other modes is that the scene evolves, and evolution is what a longer dwell buys.

Bucket `umbra-open-data-catalog` (us-west-2), key:

```text
sar-data/tasks/Houston_TX/8f28779a-7cfa-412d-9856-4d4d08079198/2024-10-30-16-10-33_UMBRA-06/2024-10-30-16-10-33_UMBRA-06_CPHD.cphd
```

`CPHD/1.1.0`, `FX` domain, `SGN` -1, `CF8`, no `AmpSF`, `SC0` and `SCSS` both constant (`SCSS` 15,031.265 Hz). 208,753 vectors x 70,875 samples; the signal block alone is 110 GiB. 23.50 s dwell at 8,942.2 Hz PRF, 22.3 deg of aspect (-16.93 to +5.38 about closest approach, which falls at vector 158,915), 458.8 km slant range, 53.42 deg graze, 3.06 cm wavelength. Collected 09:48 local over downtown Houston, so the scene has daytime traffic in it.

### The subset recipe

The 110 GiB signal block is never downloaded whole. Resolution is set by a **contiguous central frequency subband** and frame length by a **pulse window**, chosen to match:

| Choice | Value | Gives |
| :--- | :--- | :--- |
| central subband | samples `[33389, 37485)` — 4,096 of 70,875 | 61.6 MHz, `del_r` 2.435 m, **4.09 m ground range** |
| pulse window | 2,020 vectors — 0.226 s | 0.214 deg of aspect, **4.09 m cross range** |
| image grid | 2048 x 2048 at 2.5 m, anchored at the IARP | 5.12 km, 1.6x oversampled |

The 4.09 m pin above used a 2048² / 2.5 m / 5.12 km grid. **The shipped `video_sar.yaml` is a local subband of the same full dwell at ~1.0 m ground-range GSD** (`tools/sar_data/recipes/houston_1m.yaml`, 16753 samples, ~27 GiB) on a 6.8 km IARP-centred ENU square (`lattice.counts: 6800`, `spacing: 1.0`), formed at `lattice.max_side: 5632` (~1.21 m GSD). Lower `max_side` on a slower GPU (same km, coarser pixels). Aperture 8262 pulses is `2020 × 4.09 / 1.0`. The 0.5 m subset (`houston_0p5m.cphd`, 33505 samples, 13600², `max_aperture_pulses: 16524`) missed collect time by ~4× on the demo GPU and is kept on disk, not in the live YAML.

A subband is phase-consistent as a lower-bandwidth collect provided the centre frequency and bandwidth are derived from the **actual sample grid** (`SC0 + (i0 + (n-1)/2) * SCSS`, `n * SCSS`) rather than from the global `FxBand`, which on this collect implies a sample spacing 0.06% off the true `SCSS`.

**One fixed grid is valid for the whole dwell**, because SRP travel across the collect is exactly 0.0 m — a true staring spotlight. That is what makes frames pixel-registered without resampling, and it is worth asserting rather than assuming: a sliding-spotlight or stripmap collect would need the grid to follow the SRP. The 5.12 km grid is inside the sampled range window on 208,751 of the 208,753 vectors.

### Measured budget

Every number below was measured, not estimated from an assumed bandwidth.

**Pull once; frame rate is then free.** Non-overlapping windows tile the dwell exactly once, so the whole dwell at 4.09 m is the *minimum* download for full-collect video: 6.84 GB in 208,753 range requests. Overlapping windows reuse vectors already on disk, so a higher frame rate costs compute only.

| Stage | Cost | Notes |
| :--- | :--- | :--- |
| PVP block | 78.5 MB, 5.3 s | once per collect, independent of frame count |
| Signal subband, full dwell | 6.84 GB, **7.1 min** | 208,753 range requests at 452–547 req/s |
| Forming 410 frames (75% overlap, 17.7 fps) | 214 s wall, 14 s GPU | 0.52 s/frame wall is file writing, not compute |

The pull is **request-rate bound, not bandwidth bound** — 6.84 GB in 7.1 min is only ~16 MB/s, nowhere near link capacity. Concurrency is therefore the tuning knob, and it has an optimum rather than a monotone: 16 connections without keep-alive gave 29 req/s, 48 without keep-alive 87 req/s, 64 with keep-alive 263 req/s, **128 with keep-alive 456–547 req/s**, and 256 collapsed back to 166 req/s. Pin 128 with connection reuse; do not raise it.

### What the demo shows

Across the full collect the aspect sweeps 22.3 deg, so layover rotates and specular returns flash — tall structures visibly swing and linear features brighten and fade, while the street grid stays registered. That is physically correct on a flat `Z = 0` grid and typical of real video SAR, and it is what makes the sequence unmistakably a video rather than a repeated still.

**Movers are real but are the subtle layer.** Analysis over a 3.6 s burst found 20 unambiguously moving vehicles across the 5.12 km scene, tracked over up to 16 consecutive frames with sub-pixel fit residuals and landing on the freeway corridors. They read clearly only after the static background is removed. The demo does not ship that background-subtraction view; traffic is not expected to be obvious in raw magnitude.

Two findings from that analysis are worth carrying into the operator work:

**A mover's apparent velocity here is exactly `2 * v_azimuth`, along the azimuth axis.** For this collect `omega * R / V` measures 0.961 to 1.000 across the dwell (1.0000 at closest approach), and at that value the radial term in a constant-velocity target's apparent image motion cancels: the displacement `v_r * R / V` grows at exactly the rate the geometry rotates it away. This is a property of near-broadside spotlight collection, not of this scene, and it is the discriminator that separates traffic from specular glint sliding along a curved structure — glint must move tangent to its structure, a mover need not. Any mover-related test or demo annotation should use it rather than a naive frame-difference.

**The azimuth axis rotates 12.7 deg over the collect** (93.3 to 105.8 deg in image coordinates). A single constant azimuth reference works for a short burst and silently loses detections at the ends of a full one, so the reference is per-frame. The frame emitter records it per frame for this reason.

### A data-quality finding

Two of the 208,753 vectors — indices 171,540 and 182,384 — carry `TOA1`/`TOA2` around -19 km where every other vector reads about -2.3 km. Their `SIGNAL` flag is valid and their sample data is present, so this is bad per-vector metadata rather than bad signal, and it is invisible until something takes the intersection of the TOA support across all vectors and gets a negative extent.

The practical impact here is nil. The value is as evidence: this is a real, public, otherwise clean collection with corrupt PVP in it, which argues concretely for the per-vector sanity checking `CPHDReaderOp` is contracted to do, and against deriving anything global from an unguarded min/max over a PVP field.

## Tiled strip — Imperial Valley stripmap (tiled strip)

Same collect as continuous strip. Bucket `capella-open-data` (us-west-2; objects also answer at the REST hostname `capella-open-data.s3.amazonaws.com`), key:

```text
data/2024/11/29/CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602/CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602.cphd
```

Credential-free: `aws s3 cp --no-sign-request` or HTTPS GET. 3,706,536,768 bytes (3.452 GiB). `CPHD/1.1.0`, `CI4`, FX, `SGN` +1, `AmpSF` present, `STRIPMAP` / VV / right-looking. 159,051 vectors × 5,760 samples. TxTime dwell **15.766 s** at 10,088.1 Hz PRF. IARP 32.8423°N, 115.5275°W (HAE 10 m) — Imperial Valley, California. Graze 73.11 deg, slant 616.8 km, FxC 9.650 GHz, bandwidth 200 MHz. `SC0` 9,550,000,277.42 Hz and `SCSS` 34,722.221 Hz both constant. `SRPFixed=false`: SRP travels 103.8 km (platform 113.8 km at 7,217 m/s). Ground-range swath ≈ 10.3 km (TOASaved); the formed cross-track is the IFFT window (~14.8 km). Fields and canals; a ~10:1 strip. `nvec` only divides evenly by 3; the demo drops one vector (`pulse_range: [0, 159050)`) for ten ~10.4 km along-track tiles.

The demo pulls the file whole (no subsetter): `python3 tools/sar_data/pull_cphd.py --app tiled_strip --dest data/sar`. `--app continuous_strip` is the same key and dest. Tiles are along-track ROIs of a parent lattice that follows the SRP chord (along-track on `v`); every tile forms the parent's full cross-track so joins line up. `lattice.max_side` caps each tile's along-track axis, not `u`; `SarMosaicOp` holds finished squares. Taylor `range_window` / `aperture_window`; `weight` wired. Palmdale/Rosamond is not this YAML.

## Continuous strip — Imperial Valley stripmap (continuous strip)

Same file as tiled strip. `continuous_strip.yaml` sets `geometry_change = scroll` on a
parent-width window `lattice.max_side` deep along `v` (buffer depth, not
a coarsening cap). `SarMosaicOp` blits each window into the parent so
Holoviz is the strip (same landscape fit as tiled strip). `weight` is
unconnected (untapered ~−13 dB). `emit_interval_pulses: 256` with
`pulses_per_tick: 1024` — the app stager slices each reader cube so
formation sees the interval. Pull: `python3 tools/sar_data/pull_cphd.py --app continuous_strip --dest data/sar`.

## Palmdale / Rosamond — optional CPHD 1.0.1 spare

Not the shipped `tiled_strip.yaml`. Same bucket, key:

```text
data/2021/7/27/CAPELLA_C02_SM_CPHD_HH_20210727210317_20210727210321/CAPELLA_C02_SM_CPHD_HH_20210727210317_20210727210321.cphd
```

738,094,880 bytes (703.9 MiB). `CPHD/1.0.1`, `CI4`, FX domain, `SGN` +1, `AmpSF` present, `STRIPMAP` / HH / right-looking. 43,746 vectors × 4,148 samples; signal block 692.2 MiB. TxTime dwell **4.265 s** at 10,257.1 Hz PRF. IARP 34.7988°N, 118.0771°W (HAE 657 m), labeled Palmdale; the SRP track is Rosamond (~1.4 km), Plant 42 / Palmdale Airport 15–19 km off-swath. Graze 52.26 deg, slant 664.9 km, FxC 9.650 GHz, bandwidth 100 MHz. `SRPFixed=false`: SRP travels 23.76 km. Kept as a small CPHD 1.0.1 + `CI4` + `AmpSF` spare; the live tiled-strip demo is Imperial Valley.

## The offline reference path, and what it is not

The MatX `sar_bp` example was used to form every frame described above: a `.sarbp` file per frame, written directly from the range-read subband, formed at 34 ms per 2048 x 2048 frame (440 Gbp/s, `fltflt` precision, `pixel-z zero`, RTX PRO 6000 Blackwell).

**This is a golden-image path, not a build dependency.** `matx::experimental::sar_bp` is experimental and is absent from the MatX bundled with the Holoscan SDK this module builds against, which item I1 in [OPERATOR_API.md](OPERATOR_API.md) already records; nothing in the wave may depend on it. Unchanged by the 2026-08-12 measurement under I1: a newer MatX has it, but pulling one in is paused until the Holoscan 5.0 conversion lands, and it stays in the `experimental` namespace there regardless. Two consequences:

**The shipped application does not use `.sarbp` or a pulse store at all.** It streams a CPHD through `CPHDReaderOp`. Since it cannot stream the 110 GiB original, the demo needs a **local subset CPHD** written by the subsetter — about 7 GB for the full Houston collect at 4.09 m, versus about 2.3 GB had we taken an 8 s excerpt. That is the real cost of the full-collect decision, and it is a one-time cached pull rather than a per-run one.

**Independent re-formation is a golden for the accumulation path.** Each of the 410 frames above was backprojected from scratch over its own window, with nothing carried between frames. The shipped chain does the opposite — `ApertureManagerOp` retains vectors and emits `weight = -1` to evict, with `BackprojectionOp` accumulating across ticks — so diffing incremental accumulation against independent re-formation directly tests the cross-tick accumulation convention, including the long-slide floating-point drift [WORKFLOWS.md](WORKFLOWS.md) flags as wanting periodic re-forming.

The cost asymmetry that motivates the incremental path is also now quantified: incremental costs `2 * stride` pulse-backprojections per frame against `window` for re-formation, breaking even at 50% overlap and saving 2x at the 75% overlap used here.

## Open questions

Tracked here rather than in [TEAM_BLOCKED.md](TEAM_BLOCKED.md), since none needs a team decision — they need work.

1. **Recipe / geometry source of truth.** A recipe names a collection, a subband, a window and a grid. Whether the grid is declared in the recipe or derived from the collection's own geometry decides whether a recipe is portable between collections.
2. **Demo numeric budgets for the simulator path.** The image-derived scene needs a reflector cap chosen against measured generation time, the same way the Houston budget above was measured rather than assumed.
3. **The two Capella keys — closed 2026-08-30.** Palmdale (1.0.1 spare) and Imperial Valley (tiled strip and continuous strip) re-probed over range-GET (ASCII header, XML, first/last PVP `TxTime` / `SRPPos` / `SC0` / `SCSS`) and transcribed above. Size and TxTime dwell match the original survey (~704 MiB / ~4.3 s, ~3.45 GiB / ~15.8 s). The live tiled-strip demo is Imperial Valley.
4. **Fixture regeneration determinism.** With the simulator in-repo the bit-for-bit requirement loses its original justification, but CI still wants a generated fixture to be stable across platforms; what "stable enough" means is unsettled.
