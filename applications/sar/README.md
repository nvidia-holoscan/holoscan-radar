# Standalone SAR applications

The six HSDK 5 C++ and Python SAR workflows share the CPHD reader, range compression,
aperture manager, and backprojection operators. Five are image-formation workflows; the
sixth, autofocus, adds the opt-in phase gradient autofocus estimation/reformation chain over a retained, replayed aperture. Each executable
owns its graph wiring, config validation, build gate, YAML, metadata, and
fixture smoke. See the [workflow guide](../../docs/WORKFLOWS.md)
for the named image-formation workflows and their relationships. Shared Graph driving, geometry conversion, reporting, mosaics,
and CUDA visualization live in `applications/common/radar_app/sar/`.

| Application | Main behavior |
| --- | --- |
| [spotlight](spotlight/README.md) | One reset image |
| [video_sar](video_sar/README.md) | Retained sliding aperture and signed eviction weights |
| [tiled_strip](tiled_strip/README.md) | Staged tile lattices and parent mosaic |
| [continuous_strip](continuous_strip/README.md) | Batch slicing, scroll geometry, and mosaic |
| [independent_images](independent_images/README.md) | Per-collection geometry, spacing, and frequency |
| [autofocus](autofocus/README.md) | Retained aperture, conjugate-grid PGA estimate, corrected product formation |

## Python applications

Every SAR workflow has a colocated Python graph and CLI manifest. Python wires
the processing operators and application stages; both languages share native
configuration validation, pools, mosaics, reporting, HUD/timeline, and run
driving. Existing Mermaid diagrams describe both implementations. The CLI
selects Python when `--language` is omitted; pass `--language cpp` explicitly
to run the C++ reference.

```bash
./holoscan-radar run spotlight --language python
./holoscan-radar run video_sar --language python
./holoscan-radar run tiled_strip --language python
./holoscan-radar run continuous_strip --language python
./holoscan-radar run independent_images --language python
./holoscan-radar run autofocus --language python
```

Pass a preset through `--run-args=config.yaml`; a SAR input override is
`--run-args="config.yaml --cphd /path/to/input.cphd"`. Real-data presets still
need their documented CPHD files. Python does not download collections.

For programmatic use inside the SDK environment:

```python
from holoscan_radar.applications.spotlight import build_graph

workflow = build_graph("spotlight.yaml", cphd_override="/data/input.cphd")
# workflow.graph is the SDK Graph. workflow.options exports equivalent compile
# options for inspection; run() uses the shared native driver's placements.
images = workflow.run()
```

`run()` runs once, returns the reported image count, and raises on runtime
failure. Each workflow owns independent HUD, counters, and replay support.
SIGINT/SIGTERM cooperatively stop and join native execution before propagating
`KeyboardInterrupt`. SDK calls release the GIL while running.

For an installed wheel, set `HOLOHUB_DATA_DIR` at runtime when preset data lives
outside the build machine's data directory. For example, a preset input
`sar/collection.cphd` resolves to `/datasets/radar/sar/collection.cphd` with
`HOLOHUB_DATA_DIR=/datasets/radar`. The wheel does not include CPHD collections.
An explicit absolute input or `cphd_override` remains supported. Existing input
files relative to the caller or beside the YAML take precedence over the runtime
data root; an unset or empty variable retains the compiled fallback.

Public SAR operators use mapping parameters, matching their YAML spelling.
`CPHDReaderOp`, `ApertureManagerOp`, and `BackprojectionOp` expose explicit
`F32GeomF32`, `F32GeomF64`, and `F64GeomF64` factories. The apps choose
`F32GeomF64` by default, preserving ECEF precision; `sample_precision: double`
selects `F64GeomF64`. Other SAR operators expose F32/F64 factories.
`TensorDumpOp` is not a Python factory; Python SAR apps keep `dump: true`
because the shared graph runtime adds the C++ operator itself. Python live enum ports retain the
[documented SDK limitation](../../DEVELOPER.md).

The provider suite compares complex C++/Python fixture products in both sample
precisions, covers retention and geometry changes, exercises the autofocus
branches, and runs headless rendering. These checks establish application and
binding parity; they do not qualify real-collection image quality.

## Choose a run path

For immediate offline validation, run `./holoscan-radar test` from the
repository root **on the host**. The CLI builds the module in a container and
runs the registered suite, including the fixture smokes for all six SAR apps and headless
visualization checks. CTest generates the CPHD fixture in C++ and checks image
counts and dimensions. These are wiring and numerical smokes, not validation
of image quality on a real collection. For targeted SAR CTest commands inside
a development container, see the individual app guides.

The shipped YAML files are **real-data presets**. Download or subset the CPHD
files in the host `data/sar/` directory using [optional data preparation tools](tools/README.md),
then run through the Holoscan CLI or directly inside the DevContainer. Relative
CPHD paths first use an existing caller-relative or YAML-relative file, then
runtime `HOLOHUB_DATA_DIR`, then the data directory compiled into the application
(default: the repository `data/` directory). The repository mount exposes the
files in both environments. Each app guide gives a complete
download-and-run command block. Read each app guide for
the exact collection, geometry, output, and display prerequisites. The
[CPHD reader](../../operators/io/cphd_reader_op/README.md) and
[SAR operator references](../../operators/sar/README.md) describe the library
contracts. Public headers/schemas own declarations and defaults; [architecture](../../docs/ARCHITECTURE.md) covers shared rules.

Visualization is best-effort: if rendering falls behind, both SAR display
connections discard older frames and show newer images. Formation, image
reporting, and optional tensor dumping keep their normal delivery policy.
The paced CPHD reader, range/manager, and backprojection run in separate
partitions so collect-vs-wall lag is
the live playback clock (TxTime versus now), not GPU time on the controller.
Looping presets (`loop: true`) stop with Ctrl-C in the launch terminal; the
C++ driver handles SIGINT/SIGTERM, while Python retains Python signal ownership;
both stop and join native execution cooperatively. One in-flight backprojection may finish before the process exits.
When visualization is enabled, a cached status panel describes the displayed
image: collection name, formed dimensions, backprojection work, GPU processing
latency and throughput. The sensor look annotation stays over the image.
The docked timeline defaults to 112 pixels high and polls batch events at 10 Hz.
Set `app.visualization.timeline.placement: window` for a separate timeline.

Interactive startup shows a cached PREPARING canvas until the first formed
image arrives. The progress panel reports observed checkpoints and unique
received versus GPU-completed pulses toward the next image. Autofocus adds
auxiliary formation, estimation and final-formation checkpoints. These are
observed stages, not a heartbeat or estimated completion time.

The history shows cyan `Pulse batch received` and green
`Pulse batch processed` events on a 20-second axis ending at `Latest batch`.
Connectors pair a batch's receipt with its observed GPU completion. History
advances with batch events, while the progress readout refreshes at 1 Hz.
Optional paced playback lag remains available. Timeline ticks do not render
or download the cached main image. See the
[shared visualization guide](../common/radar_app/viz/README.md#docked-sar-timeline)
for display controls, event semantics and limits.

The former `applications/sar_formation/` path and `sar_formation` executable
were replaced by the five image-formation workflows above; autofocus adds
a separate workflow. There is no compatibility launcher.
The host `./holoscan-radar test` command includes generated CPHD smokes and
headless visualization. The CPHD fixture is generated once by C++.
See the [SAR operator guides](../../operators/sar/README.md) for
owner-local limitations and [SAR input data](tools/README.md) for optional real
collections and [SDK development](../../DEVELOPER.md) for the pinned image.

## View saved images

With `dump: true`, the app writes complex `.npy` products independently of the
GUI. The continuous-strip workflow instead dumps its published mosaic view;
when that view is decimated, samples contain amplitude with zero phase.
It does not export the full-resolution accumulator. See the
[continuous-strip output contract](continuous_strip/README.md#output-and-display).
Convert a saved image to a PNG on the host from the repository root:

```bash
uv run applications/sar/tools/view_image.py path/to/image.npy
```

The PNG is written beside the input; no display is needed. `uv` supplies the
optional plotting dependencies. Add `--show` for interactive viewing or use
`clean.npy corrected.npy --compare` for panels sharing one intensity scale.
The [viewer guide](tools/README.md#view-saved-images) describes
range settings and array orientation. Save logs within the checkout print the
viewer command for the actual image filename.

## Troubleshooting

### Vulkan or display initialization fails on any platform

For x86, Jetson, and GB10 systems, start with the
[HSDK 5 visualization troubleshooting table](https://docs.nvidia.com/holoscan/sdk-user-guide/modules/visualization#troubleshooting).
It covers display forwarding and authorization, NVIDIA Vulkan driver visibility
inside containers, CUDA/Vulkan adapter matching, hybrid-GPU presentation, and
multi-GPU placement. Match the reported error to those checks before applying
the GB10 memory-cache workaround below.

### Vulkan initialization fails on DGX Spark / GB10

`VK_ERROR_INITIALIZATION_FAILED` from `create_device_for_surface` or
`create_swapchain` can accompany host NVIDIA-driver allocation failures. Check
`journalctl -k` for `NV_ERR_NO_MEMORY` and `kgrctxAllocMainCtxBuffer` near the
application failure. On GB10, these errors can occur even when `free -h` reports
substantial available memory; a successful X11 connection alone does not rule
out this driver allocation failure. Rebuilding the container does not reclaim
host memory caches.

On the **host**, try NVIDIA's documented cache-reclaim workaround, then restart
the application:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
./holoscan-radar run continuous_strip
```

This flushes filesystem caches; it does not delete downloaded CPHD files.
It is a debugging workaround, not a diagnosis for every Vulkan error. See
[NVIDIA's DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture).

### Formation falls behind and a required output queue fills

The reader may wait for downstream capacity when `cphd_reader.wait_for_downstream`
is enabled. Required outputs in the formation stages fail the run when their
queues fill. A fatal required-emit error also requests cooperative cancellation,
so a reader already retrying an emit can exit and the application can report
the original error and its playback-rate/image-size guidance. Visualization
uses separate drop-oldest queues and may discard stale display frames.
