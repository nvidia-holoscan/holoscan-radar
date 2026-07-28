# FMCW interchangeable AoA estimators (`fmcw_aoa_estimators`)

**The point of this app:** show how naturally **Holoscan** lets you drop a whole
family of operators that solve the *same problem different ways* into **one
running graph** and swap among them **at runtime** — with only the selected one
paying compute each frame. You connect every candidate with the same `add_flow`
you'd use for any edge, add a few-line dynamic-flow callback, and the scheduler
does the rest: no dispatch logic, no branch-heavy `compute()`, no data copies,
no restart. Here that family is the five sparse angle-of-arrival (AoA)
estimators — **Beamscan, MVDR, MUSIC, Root-MUSIC, and ESPRIT** — but the
takeaway is the pattern: **interchangeable operators are close to free in a
Holoscan graph**, so you get an apples-to-apples comparison on the same live
scene, one estimator's worth of compute per frame.

The same app ships in **C++ and Python**. They share the graph shape, the YAML
config, and the selection policy; they differ only in *how* the one-of-N branch
is realized, because of Holoscan Python-binding limits (see
[Two realizations](#two-realizations-c-and-python)). This README covers both.

- [Why Holoscan makes this easy](#why-holoscan-makes-this-easy)
- [The five estimators](#the-five-estimators)
- [How runtime selection works](#how-runtime-selection-works)
- [Selection policy](#selection-policy)
- [Pipeline](#pipeline)
- [Two realizations: C++ and Python](#two-realizations-c-and-python)
- [Display](#display)
- [App-local operators](#app-local-operators)
- [Build & run](#build--run)
- [What you should see](#what-you-should-see)

## Why Holoscan makes this easy

In a Holoscan graph, operators are just nodes and `add_flow` edges name the
ports they connect. That model is what makes swapping methods feel natural
rather than like a framework you have to fight:

- **Wiring a candidate is one edge.** Connecting an estimator is the ordinary
  `add_flow(producer, estimator)` / `add_flow(estimator, consumer)` you use
  everywhere else — no adapter, no wrapper, no special "pluggable" base class.
  Adding a *sixth* estimator is one `make_operator` and two `add_flow`s.
- **Selection is a short callback, not a scheduler.** Holoscan's **dynamic flow
  control** lets one operator decide which of its outgoing edges fires this tick.
  You hand `set_dynamic_flows` a few lines that pick the chosen edge; the
  framework then runs only that branch's `compute()` and skips the rest. There's
  no dispatch `switch`, no `if method == ...` buried inside an operator, and no
  manual enable/disable of nodes.
- **Nothing gets repackaged.** The selected estimator reads the same tensor off
  the same port as every other candidate, so switching methods moves zero data
  and needs zero glue between operators.
- **Config rides along.** Each operator pulls its own parameters straight from
  YAML with `from_config(...)`, so an estimator's knobs (scan grids, source
  counts, angle convention) travel with it. Swapping estimators never means
  re-plumbing configuration.

The result: Holoscan treats "five ways to do it" almost exactly like "one way."
The *only* thing you owe it is the obvious agreement between the candidates —
same port names and tensor shapes (`snapshot` in, `aoa_estimates` out, plus the
`array_geometry` / `angle_convention` params) — and the plain-tensor data plane
means even that is just ordinary ports, not a bespoke message type.

**Two demo specifics** let a single graph serve all five estimators in 2-D:

- **One geometry for everyone.** A single filled *y-z uniform rectangular array*
  (URA) feeds all five: its two orthogonal apertures resolve azimuth (y) **and**
  elevation (z), so the grid-scan estimators scan the full `(az, el)` grid and
  Root-MUSIC / ESPRIT take their native 2-D paths. (A ULA is still valid and
  collapses each estimator back to azimuth-only.)
- **One producer, two look-counts.** Beamscan is single-snapshot; the
  covariance/gridless four want multiple looks per detection. So the multi-look
  snapshot feeds MVDR/MUSIC/Root-MUSIC/ESPRIT and Beamscan gets a single center
  look — the one spot the branches differ, handled explicitly (a `K=1` slice in
  C++, a second `K=1` gather in Python).

## The five estimators

All five map the same `snapshot` to the same `aoa_estimates`; they differ only
in method and cost.

| Estimator | Class | One-line idea |
| --- | --- | --- |
| **Beamscan** | grid scan | Conventional beamformer; scan the grid, take the power peaks. Single-snapshot. |
| **MVDR** | grid scan | Minimum-variance (Capon) beamformer; sharper peaks via the sample covariance inverse. |
| **MUSIC** | grid scan | Eigen-subspace null-spectrum; scan the grid for noise-subspace nulls. |
| **Root-MUSIC** | gridless | Same subspace, but solve for directions by polynomial rooting — no grid. |
| **ESPRIT** | gridless | Rotational-invariance (shift) subspace method; closed-form directions. |

Grid-scan estimators are drop-in with a shared `(az, el)` grid; the gridless two
need no grid at all. Swapping among them at runtime is exactly what this app
demonstrates.

## How runtime selection works

Three ingredients turn "five operators in a graph" into "one runs per frame,
chosen live":

**1. Static fan-out.** At compose time the snapshot producer is connected to
*every* estimator branch. The edges all exist; they are just not all active.

**2. Dynamic flow control activates one edge per tick.** Holoscan lets an
operator decide, at runtime, which of its outgoing edges to fire this tick via
`set_dynamic_flows(...)`. The callback picks the edge whose downstream operator
matches the currently selected estimator, so only that branch executes — the
other four operators never run that frame:

```cpp
set_dynamic_flows(router,
  [](const std::shared_ptr<holoscan::Operator>& op) {
    auto r = std::static_pointer_cast<EstimatorRouterOp>(op);
    const std::string& target = r->active_target();          // e.g. "music_estimator"
    auto flows = op->find_all_flow_info(
        [&target](const std::shared_ptr<holoscan::Operator::FlowInfo>& f) {
          return f->next_operator->name() == target;
        });
    op->add_dynamic_flow(flows);                             // fire only that edge
  });
```

**3. Selection policy is decoupled from transport.** The thing that *decides*
which estimator is a small app-local operator (`EstimatorRouterOp` in C++, an
`EstimatorSelector` object in Python). It owns only the policy — rotate / hold /
file — and forwards the data untouched. Keeping "which" separate from "the
tensors" is what keeps the estimators themselves oblivious to the switching:
they are ordinary operators that happen to be gated.

Because every estimator feeds the *same* `aoa_estimates` input of the feature
assembler, and exactly one runs per frame, the assembler always sees exactly one
estimate — no merging, no per-estimator plumbing downstream.

## Selection policy

The policy is configured under the `estimator_router` YAML block and is shared
by both apps:

- **`mode: rotate`** (default) — fixed dwell: cycle through the five estimators,
  holding each for `rotate_period` frames. Fully deterministic and unattended; a
  full cycle is `5 * rotate_period` frames. The default `rotate_period: 40` runs
  each estimator for 40 frames (~4 s at 10 Hz) before advancing.
- **`mode: hold`** — pin one estimator (`hold: <name>`); ignore external input.
- **`mode: file`** — poll `control_path` each frame. Writing a recognized name
  (`beamscan | mvdr | music | root_music | esprit`, or `1`–`5`) latches that
  estimator (sticky) until changed; an empty/absent/unrecognized file falls back
  to the rotation. This is how you get **real-time control that still works
  unattended** — leave it rotating, override on demand:

  ```bash
  echo music > /path/to/control_file   # or: echo 3 > /path/to/control_file
  ```

`log_every: N` additionally logs the active estimator every N frames (0 = only
on change).

## Pipeline

```text
SimAdcSourceOp                 (GPU-simulated SIMO ADC cube: N_rx, chirps, samples)
  -> FMCWRangeProcessorOp
  -> ClutterFilterOp             (single_delay MTI: notches the zero-Doppler column)
  -> FMCWDopplerProcessorOp      (rd_maps: N_rx, doppler, range)
  -> CFARDetectorOp              (NCI over the RX axis; detection_idx = dopp, range)
  -> SnapshotGatherOp            (snapshot: N_det, N_ch, K  [+ is_valid])
  -> [selection]                 (exactly one estimator, chosen at runtime)
       { SnapshotK1Slice -> Beamscan | MVDR | MUSIC | Root-MUSIC | ESPRIT }
  -> FeatureAssemblyOp           (points: [range, velocity, az, el])
  -> ClusterDBSCANOp
  -> DetectionReductionOp        (medoid representative per cluster)
  -> ClusterReportOp             (logs cluster centroids + representatives)
```

Setting `cfar_detector.peak_only: true` drops the clustering tail (FeatureAssembly
/ DBSCAN / Reduction / ClusterReport): CFAR already emits one detection per
target, so the per-detection angle is logged straight from `DetectionReportOp`
instead. The estimator branch is unaffected — it is the DOA stage, not
clustering.

## Two realizations: C++ and Python

The two apps share the graph shape, config, and selection policy. Only the
one-of-N mechanism differs, because of two Holoscan Python-binding limits:

1. A Python fragment cannot DLPack-wrap a **bool** tensor, so `is_valid` cannot
   be delivered to a Python operator.
2. A Python operator cannot **emit an empty (0-detection) device tensor** that a
   downstream C++ operator accepts as CUDA-accessible (cupy reports a null
   pointer in `__cuda_array_interface__` for any 0-size array).

**C++ — one router, gated edges.** A single `EstimatorRouterOp` sits on the data
plane. It forwards `snapshot`/`is_valid` unchanged, advances the policy, and
exposes `active_target()`; the `set_dynamic_flows` callback above activates just
that estimator's edge. Beamscan's single-look input comes from a
`SnapshotK1SliceOp` that center-slices the multi-look snapshot to `K=1`.

**Python — two gathers, two callbacks, no data-plane router.** To keep the
bool/empty-tensor traffic entirely in C++ operators, the Python graph drops the
router and the slice. Instead it instantiates **two `SnapshotGatherOp`s** — one
multi-look for the covariance/gridless four, one `K=1` for Beamscan — and fans
each out to its estimators plus a drain. Selection is expressed purely through
**dynamic-flow callbacks that touch no tensors**:

```python
def cov_callback(op):
    target = COV_TARGETS.get(selector.select(), "cov_drain")
    op.add_dynamic_flow(op.find_all_flow_info(lambda f: f.next_operator.name == target))

def beam_callback(op):
    target = BEAM_TARGET if selector.select() == 0 else "beam_drain"
    op.add_dynamic_flow(op.find_all_flow_info(lambda f: f.next_operator.name == target))

self.set_dynamic_flows(gather_cov, cov_callback)
self.set_dynamic_flows(gather_beam, beam_callback)
```

Two callbacks now query the *same* policy, so a shared `EstimatorSelector` keeps
them in lockstep: it computes the frame's selection once, and only advances the
frame counter after **all** registered callbacks have queried it, so both agree
regardless of the order Holoscan invokes them. The bool `is_valid` outputs are
drained (Holoscan logs a benign `Unsupported DLPack data type (code: 6)` per
emit); the estimators run without the mask (treated as all-valid), which matches
the C++ behavior closely enough for the demo.

The net effect is identical: exactly one estimator computes per frame, chosen by
the same rotate/hold/file policy.

## Display

With `visualization.enabled: true` (and a reachable display, or `headless: true`
for offscreen rendering) the app opens a Holoviz window built from the
app-agnostic `radar_app/viz` operators. Everything is drawn with native Holoviz
primitives — no software-rasterized panel images:

- **Map title:** reads `Range-Doppler - <ESTIMATOR>`, naming the estimator
  running this frame, so you can see at a glance which one is active as the
  router advances. It is a dynamic Holoviz TEXT layer re-emitted each frame by a
  small app-local op (`EstimatorLabelOp`) that reads the current selection and
  draws it where `ClusterOverlayOp` would otherwise put the plain title.
- **Top-left:** the scenario clock (`t = <n> s`, current simulated time).
- **Left:** the non-coherently integrated range–Doppler map (turbo colormap) —
  the same surface CFAR thresholds, since it runs `nci_axes: [0]` over the RX
  axis, so the picture and the overlaid detections come from the same data —
  framed with a
  border, axes, tick labels and a title, with CFAR detections drawn as filled
  cluster-colored dots. Colors are assigned by range rank, so a target keeps its
  color across frames and matches its markers in the side panels. The dark
  vertical stripe at Doppler 0 is the `single_delay` MTI clutter notch
  (expected).
- **Right (stacked):** range-vs-time and velocity-vs-time rolling plots of the
  per-cluster representative, plus a boresight azimuth/elevation scatter fed by
  whichever estimator ran this frame — so as the router advances you watch each
  estimator's DOA estimates land on the same targets in turn.

The `visualization:` block also carries the detection-marker geometry
(`marker_radius`, `marker_outline`, `marker_segments`) and `vsync`, which
selects a FIFO present mode (frame-rate capped, but far more robust to
window-resize swap-chain errors than IMMEDIATE).

If visualization is enabled but no display is reachable, the app logs a warning
and runs headless automatically (disable that probe with
`skip_display_check: true`). The cluster report log runs in both modes. Set
`AOA_CAPTURE_DIR=<dir>` (with `headless: true`) to dump composited frames for
offline inspection (PPM in C++, PNG in Python).

## App-local operators

Kept colocated with the app (not shared operators or contract changes), because
they exist only to serve *this* demo's switching pattern:

- **`EstimatorRouterOp`** (C++) — pass-through of `snapshot`/`is_valid` that owns
  the rotate/hold/file policy and exposes `active_target()` for the dynamic-flow
  callback.
- **`EstimatorSelector`** (Python) — the same policy as a plain object, queried
  by the two gather callbacks; frame-synchronized so both callbacks agree.
- **`SnapshotK1SliceOp`** (C++) — center-slices a multi-look `(N, C, K)` snapshot
  to `(N, C, 1)` for the single-snapshot Beamscan branch.
- **`EstimatorLabelOp`** — emits the live `Estimator: <NAME>` Holoviz caption
  reflecting the active estimator each frame.

## Build & run

**C++:**

```bash
cmake -S . -B build -DAPP_fmcw_aoa_estimators=ON
cmake --build build --target fmcw_aoa_estimators -j"$(nproc)"
./build/applications/fmcw_aoa_estimators/cpp/fmcw_aoa_estimators \
    [path/to/config.yaml]   # defaults to the installed fmcw_aoa_estimators.yaml
```

**Python** (same graph shape; `config.py` + `custom_ops.py` colocated). Run it
against the build tree:

```bash
ulimit -s 32768   # Holoscan warns below this stack size
HOLOSCAN_RADAR_BUILD_DIR=$(pwd)/build \
  python3 applications/fmcw_aoa_estimators/python/fmcw_aoa_estimators.py \
  [path/to/config.yaml]
```

With no config argument, both default to the shared
`cpp/fmcw_aoa_estimators.yaml`. A lighter two-target smoke config lives at
`python/fmcw_aoa_estimators_smoke.yaml`.

## What you should see

Each frame logs its CFAR detections and `N clusters`, and the selector logs
`frame <f> -> <name> estimator` as it advances through all five. Cluster
centroids/representatives land near the configured target range, velocity, and
`(az, el)` **regardless of which estimator is active** — which is the whole
point: five different methods, one graph, one switch, same answer.

### Target motion

Four of the five targets move **non-radially**: each carries a transverse
velocity component of 0.32–0.46 m/s on top of its radial one, so its azimuth and
elevation sweep over the run rather than staying pinned. Target 3 is the single
purely radial reference, holding az = 0.000 / el = 0.080 throughout — useful as
a fixed point to read the others against.

The sweeps are substantial: target 1 crosses 0.150 → 0.013 rad in azimuth
(7 grid cells) as it closes from 245 m to 83 m, and targets 2 and 4 each move
~0.04 rad in elevation. Both inbound targets deliberately sweep *toward*
boresight — an inbound target's angular rate goes as `v_t / r`, so one sweeping
outward would run off the ±0.3 rad scan grid before the run ended.

Transverse motion is small next to these ranges, so each target's radial rate
drifts by under 0.2 Doppler bins across the whole run. That is what lets the
range/Doppler geometry below survive non-radial motion intact.

### The two close passes

Three of the five targets are well separated in speed (−6.5 / +1.5 / +9 m/s);
the other two each shadow one of them through a close pass, so you can see what
the angle stage does as two objects converge in range and Doppler. Both run
1.5 m/s faster than the target they shadow — 2.5 velocity bins, still inside the
2.0 m/s `cluster_dbscan.epsilon` so the pair can merge — closing a 15–27 m range
gap to under 0.05 m, then reopening it.

- **Pair A** (target 4 shadows target 2, coinciding at t = 10 s) is separated by
  0.318 rad in azimuth at closest approach, past the array's 0.253 rad first
  null — angle alone should carry it through the crossing.
- **Pair B** (target 5 shadows target 1, coinciding at t = 18 s) differs by only
  0.025 rad in azimuth, so no estimator on this array can split it; every cell
  reports a blended angle. Elevation differs by 0.057 rad, just past the
  clustering epsilon, so whether it merges around closest approach depends on how
  far the detection cloud spreads.

Note `mvdr_estimator` / `music_estimator` are configured `max_sources: 1`, so
they report one DOA per cell even where the geometry would allow two — raising
that is the natural next experiment.

Note `mvdr_estimator` / `music_estimator` are configured `max_sources: 1`, so
they report one DOA per cell even where the geometry would allow two — raising
that is the natural next experiment.
