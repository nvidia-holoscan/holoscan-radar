# Holoscan Radar Module — SAR Input Data Plan

## Purpose

What data the SAR wave's tests and demos run on, where it comes from, and what it costs to obtain. The contract lives in [OPERATOR_API.md](OPERATOR_API.md), the chains in [WORKFLOWS.md](WORKFLOWS.md), the assertions in [TESTING.md](TESTING.md), and the sequencing in [SIGPROC_PLAN.md](SIGPROC_PLAN.md); this document is the data those depend on.

**Proposed alongside the in-review SAR contract.** Nothing here resolves a team-blocked item — the open rows stay in [TEAM_BLOCKED.md](TEAM_BLOCKED.md). Where this document and [TESTING.md](TESTING.md) disagreed on simulator ownership or fixture policy, the decisions below are the later ones; TESTING.md was amended to match on 2026-08-06, so the two now agree.

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

## Public collections, per mode

All are credential-free: a plain HTTPS or `aws s3 --no-sign-request` pull, no account. Scope is the first ship — modes 1–5 plus opt-in autofocus, per [WORKFLOWS.md](WORKFLOWS.md) → *Image-formation modes*. Modes 6–9 are deferred and unpinned.

| Mode | Collection | Full size | Status |
| :--- | :--- | :--- | :--- |
| 1, spotlight image | Umbra, Hartsfield-Jackson Atlanta — the scene the MatX `sar_bp` example uses | 28.59 GiB | pinned, header verified |
| 2, video SAR | Umbra, downtown Houston | 110 GiB | pinned, **fully validated** — see below |
| 3, tiled strip | Capella stripmap, ~4.3 s dwell, CI4 with `AmpSF` | ~704 MiB | collection identified; **exact key needs re-verification** |
| 4, continuous strip | Capella stripmap, ~15.8 s dwell, CI4 with `AmpSF` | ~3.45 GiB | collection identified; **exact key needs re-verification** |
| 5, independent images | reuses the mode 1 and 3 collections | — | no separate pin needed |

The two Capella pins carry weight beyond their modes: they are the only `CI4` collections here, and the only ones with `AmpSF` present, so they are what covers the reader's integer-sample and per-vector-scale paths. Both Umbra collections are `CF8` without `AmpSF`.

### Mode 1 — Atlanta

Bucket `umbra-open-data-catalog` (us-west-2), key:

```text
sar-data/task-data/0f1231be-26d3-4b98-96ab-f675a5375c14/2026-02-20-03-03-38_UMBRA-07/2026-02-20-03-03-38_UMBRA-07_CPHD.cphd
```

`CPHD/1.1.0`, `CF8`, 102,839 vectors x 37,268 samples, 42.01 deg graze, IARP 33.6400 / -84.4300. The key is taken from the MatX `sar_bp` example README rather than transcribed, so "the scene that example images" is exact rather than approximate — which matters because this collection is also the parity reference for `BackprojectionOp`.

## Mode 2 — the Houston collect, full dwell

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

**Movers are real but are the subtle layer.** Analysis over a 3.6 s burst found 20 unambiguously moving vehicles across the 5.12 km scene, tracked over up to 16 consecutive frames with sub-pixel fit residuals and landing on the freeway corridors. They read clearly only after the static background is removed, so the demo should ship a **median-subtracted residual view** as a viz mode rather than relying on traffic being visible in raw magnitude.

Two findings from that analysis are worth carrying into the operator work:

**A mover's apparent velocity here is exactly `2 * v_azimuth`, along the azimuth axis.** For this collect `omega * R / V` measures 0.961 to 1.000 across the dwell (1.0000 at closest approach), and at that value the radial term in a constant-velocity target's apparent image motion cancels: the displacement `v_r * R / V` grows at exactly the rate the geometry rotates it away. This is a property of near-broadside spotlight collection, not of this scene, and it is the discriminator that separates traffic from specular glint sliding along a curved structure — glint must move tangent to its structure, a mover need not. Any mover-related test or demo annotation should use it rather than a naive frame-difference.

**The azimuth axis rotates 12.7 deg over the collect** (93.3 to 105.8 deg in image coordinates). A single constant azimuth reference works for a short burst and silently loses detections at the ends of a full one, so the reference is per-frame. The frame emitter records it per frame for this reason.

### A data-quality finding

Two of the 208,753 vectors — indices 171,540 and 182,384 — carry `TOA1`/`TOA2` around -19 km where every other vector reads about -2.3 km. Their `SIGNAL` flag is valid and their sample data is present, so this is bad per-vector metadata rather than bad signal, and it is invisible until something takes the intersection of the TOA support across all vectors and gets a negative extent.

The practical impact here is nil. The value is as evidence: this is a real, public, otherwise clean collection with corrupt PVP in it, which argues concretely for the per-vector sanity checking `CPHDReaderOp` is contracted to do, and against deriving anything global from an unguarded min/max over a PVP field.

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
3. **The two Capella keys need re-verifying and transcribing** into the recipes, with header probes recorded as for the two Umbra collections.
4. **Fixture regeneration determinism.** With the simulator in-repo the bit-for-bit requirement loses its original justification, but CI still wants a generated fixture to be stable across platforms; what "stable enough" means is unsettled.
