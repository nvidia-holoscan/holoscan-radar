# BackprojectionOp — open items

This operator landed as a **mode-1 slice** plus mode-4 scroll. It does not clear
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

5. **`initial_image`.** Declared and rejected this slice.
   `geometry_change = scroll` landed with mode 4 (ring buffer,
   two slab launches, logical-order emit).

6. **`sinc` / `nearest` interpolation.** Named in the enum, rejected.

7. **Uniform-lattice expand is on device.** Host `expand_sample_lattice` is
   tests and small lattices. The formation path expands
   `SampleGeometry::sample_position` on the GPU: NVRadar writes the x/y/z
   planes directly (no packed 1.61 GiB host buffer / H2D at Atlanta
   8192x8192 double); MatX still gets packed xyz, also expanded on device.
   `sample_points` remains a device-to-device copy of the caller tensor.

8. **Host pulse tiling of kernel launches.** `launch_nvradar` /
   `launch_matx` split a tick into 1024-pulse tiles (`kKernelPulseTile`).
   Not a contract parameter: the operator still consumes the whole tick.
   Tiles `+=` into the same accum; `prepare_pulses` / LUT fill /
   float-scene downcast stay on the full tick. Planar APC/R0 keep the
   tick stride; packed VPH is a pointer offset. Ticks with `Np <= 1024`
   stay one launch. 256 was slower on Ohio (~1.5%): each tile does a
   global `simg +=`, so a tile equal to the in-kernel smem chunk flushed
   the image once per chunk.

9. **Four items found in the 2026-10-30 `dpc/sar-autofocus` push
   review.** All predate that branch (`b591e829`). The write-only
   `range_bins_connected_` member was removed on 2026-10-31 with the rest
   of the review's nits; these three were left. Only the first has a
   measured runtime cost.

   - **`locality_warned_` is cleared on every scroll tick**
     (`backprojection_op.cu:630` / `:711`), so the Taylor locality check
     re-arms continuously instead of warning once. The check costs two
     blocking device-to-host synchronisations, which on the default
     Taylor backend therefore run every tick of a scrolling collect
     rather than on the first. Measured, not inferred. Latch it per
     geometry instead of per emit.
   - **`sample_points_connected_` latches on arrival, not on wiring**
     (`backprojection_op.cu:296`): `if (sample_points) sample_points_connected_ = true;`.
     The port is `[live]` with `SafeDefault{}`, so a tick with no match
     leaves the flag false, and it is read in four places (`:466`,
     `:558`, `:720`, `:850`) that decide between points mode and
     geometry mode. The operator therefore behaves as geometry-mode
     until the first lattice happens to arrive. **Not fixable as
     written**: HSDK 5 exposes no input-connectivity query — there is no
     `is_connected()` on `Input<Tensor>` or on the spec — so `setup()`
     cannot read the wiring. Closing this needs either an SDK affordance
     or an explicit constructor flag on the operator; the latter is a
     contract change and belongs with the team.
   - **A scroll tick that misses its `[live]` `sample_geometry` throws**
     (`backprojection_op.cu:323` and `:441`). On a miss, `new_geometry`
     falls back to `params().sample_geometry` — the *original* lattice —
     which no longer equals the scrolled `effective_geometry_`, so
     `geometry_override` goes true and `scroll_translation_rows` is
     asked to move backward along v. That is the hard error at `:119`.
     The whole point of `SafeDefault{}` is that a missed tick is legal,
     so a scrolling instance should treat a miss as "no change" rather
     than as a reversion to Params.
10. **Element counts are narrowed to `int` in three places**
    (`backprojection_stage.hpp:558`, `:576`, `:1104`). The counts are
    `int64_t` and the lattices they describe are large — Atlanta is
    8192x8192 — so these sit closer to the 2^31 element ceiling than
    anything else in the operator. Not reached by any current
    configuration.

11. **`locality_warned_` is no longer re-armed on every scroll tick**
    (fixed 2026-11-01). A scroll advances the lattice by whole rows and
    does not change the locality property, so the per-tick reset turned a
    warn-once into a warn-forever and re-ran a check costing two blocking
    device-to-host synchronizations per tick on the default Taylor
    backend. `continuous_strip` scrolls every tick.

## Team-blocked

### R11 — the leading batch axis on `image` is historical, half-built, unreachable

Raised 2026-10-30 while reviewing `dpc/sar-autofocus` for push. Nothing here is
from that branch; it is recorded because the review had to understand it and the
next reader should not have to re-derive it. **Do not resolve this in code.**

**What the operator section says.** [`SAR/OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md),
all of it inside `## Historical SAR design reference` — which matters, and is
taken up below:

- **`:543`** — **Out:** `image` is `(..., num_v, num_u)` in both sample-geometry
  modes, and "Leading batch axes are **preserved**, one independent accumulator
  per batch element".
- **`:631`** (Notes → *Batch semantics*) — "Leading axes are preserved as
  independent images, each with its own accumulator — the rank-agnostic reading
  of invariant 4, and what a caller wants for independent sub-collections. To
  integrate **across** a leading axis instead, flatten it into the pulse axis
  upstream; the operator will not guess which is meant."
- **`:632`** — "A batch is a batch of **collections**, not of **scenes**, because
  the geometry carries no leading axis." One lattice, `N` independent
  accumulators. The tomographic case (`N` lattices, one pulse stream) is
  explicitly *not* spellable this way and is `N` instances.

**What the code does.** Accumulation implements the contract; the emit does not.

- `backprojection_op.cu:376` — `const BatchedShape cube = collapse_batch(*profiles, "range_profiles")`.
  `cube` is not a SAR concept: `collapse_batch` is the shared helper in
  `radar_common/tensor/tensor_utils.hpp:60` that splits any rank >= 2 tensor into
  `rows` (second-to-last axis), `cols` (last), and `batch` (product of everything
  in front). Here `rows = num_pulses`, `cols = num_range_bins`. The name is FMCW
  data-cube jargon carried over; for SAR the tensor is a 2-D matrix.
- `:589` / `:592` — `ensure_matx_accum` / `ensure_accum` allocate `cube.batch`
  accumulators.
- `:686` — the accumulate loop runs every batch element and passes
  `static_cast<int>(b)` into `launch_matx_window` / `launch_nvradar_window`, which
  selects the accumulator. So they are genuinely independent images, not one
  shared sum.
- `:829` / `:832` — **the split.** `emit_accum(*state_, dst, 1, num_v, num_u, stream)`
  hardcodes a batch of `1`, and the declared output shape is `{num_v, num_u}`.
  Everything past accumulator 0 is computed at full cost and discarded. This is
  the half: accumulation follows `:543`, the emit follows `:29`.
- `:646` — the batch concept is threaded through more than one port:
  `weight` must satisfy `wbatch == cube.batch` or the tick throws.
- `:738` — `unemitted_pulses_ += num_pulses * cube.batch`. Under the
  independent-accumulator model this is wrong by a factor of `cube.batch`: it
  counts every batch element's pulses toward the emit cadence of a single image,
  so with batch 4 and `emit_interval_pulses` set, the cadence fires four times
  earlier than accumulator 0 has earned. Whether this is a second bug or a
  remnant of an older "batch means more pulses" reading is not established.

**Why it is unreachable today.** `ApertureManagerOp.range_profiles_out` is rank-2
`(num_output_pulses, num_range_bins)` and `CPHDReaderOp.phase_history` is rank-2,
so no graph in this repository produces a leading axis and `cube.batch` is always
`1`. Verified 2026-10-30 against every SAR app config. The emit gap is therefore
latent, not active — but it means the feature cannot be used even deliberately:
a caller who sent a rank-3 cube would get correct accumulation and silently lose
every image but the first.

**The HSDK 5 section already answers this; the historical section and the code
were never updated to match.** Everything above (`:543`, `:631`, `:632`) sits
under `## Historical SAR design reference` (`:98`), and the document opens by
saying what that section is worth:

| Where | Says |
| --- | --- |
| `SAR/OPERATOR_API.md:9` (HSDK 5 preamble) | The historical section's "HSDK 4 transport statements, proposed/acceptance labels, **opaque `...` axes**, sticky overrides, and metadata stream boundary are historical and do not override this section." |
| `:15` | "There is no implicit batch axis." |
| `:29` (*Implemented operator ports*) | `BackprojectionOp<T, TGeom>` → `image`: complex T `(num_v,num_u)`. Rank 2, no leading axis — which is exactly what the emit at `:832` builds. |
| `:91` (HSDK 5 migration table) | "Named axes replace opaque leading batch; fat outputs capture `MaxTensorExtents`." |
| [`AGENTS.md`](../../../AGENTS.md) invariant 4 | "Do not treat an open-ended leading `...` as an opaque module batch axis — multi-item throughput is an application concern (loop / fan-out)." |

Against that, the statements for a batch axis are `:543` / `:631` / `:632`, the
carry-over paragraph at `:110` ("rank-agnostic batching, with leading axes
treated as opaque batch dims") and the invariant restatement at `:1055`
("Leading axes stay opaque and preserved") — five, not four. Every one of them
sits inside the historical section, so `:9` disclaims them all by name. So the emit at `:832` is not a gap in the
contract at all: it implements `:29`. **Read this item as "the code is right and
three paragraphs plus some dead compute are stale", not as "the contract is
undecided".**

Two things keep it from being purely editorial. First, `:543` / `:631` / `:632`
is substantive prose rather than `(...)` notation — `:631` is aware of the
tension with invariant 4 and argues around it, calling per-batch accumulators
"the rank-agnostic reading of invariant 4" — so a reader who takes `:9` to cover
only the shape spelling will still read it as live, which is how this review
read it for most of a day. Second, the per-batch accumulators are still
allocated and filled, so the code half-asserts the feature even though its
output does not.

**Do not delete it without reading D19.** [`TEAM_BLOCKED.md`](../../../docs/planning/SAR/TEAM_BLOCKED.md)
→ D19 defers (rather than rejects) letting `sample_points` carry a leading axis
the data does not, making the output batch the outer product of the two. That is
the change that would collapse the `N`-instance tomographic volume into one
operator, and it is spelled in terms of this batch axis. Killing the axis
forecloses D19's current formulation.

**The options, with what each costs.**

1. **Leave it.** Cheapest, and nothing depends on it either way. Keeps three
   paragraphs that read as live describing a shape the operator does not emit,
   so the next reader pays this cost again.
2. **Make the code match `:29`.** Reject a leading axis at validation, delete
   the per-batch allocation and loop, and strike `:543`/`:631`/`:632` (or mark
   them superseded in place, like the other historical prose). Removes dead
   compute and makes the historical section stop contradicting the port table.
   Also settles `:738` by construction, since `cube.batch` becomes 1 by
   definition. Forecloses D19 as currently written.
3. **Promote the batch axis into the HSDK 5 section**, i.e. change `:29` to
   `(..., num_v, num_u)` and finish the emit. A real contract change, reviewed
   as one. The accumulators already exist and are already filled, so the code
   cost is small, but it reopens the conflict with invariant 4 and `:15` and
   commits to maintaining a path nothing exercises or tests.

**Recommendation, not a resolution.** Option 2 is what the HSDK 5 section
already says, so it is the default unless someone wants the feature. The part
that is genuinely a team call is D19: it is spelled in terms of this axis, and
option 2 requires it to be re-expressed (most likely as `N` instances, which is
already how `:632` handles the tomographic case). Whoever picks up D19 should
pick up this item with it; they cannot be settled separately.

## Not ours to close

- **R10** — metadata key namespace. This operator consumes uint8
  `control` only and does not read `stream_boundary`.
- **I5** — range-compression normalization default. Lives on
  `SARRangeCompressionOp`.
