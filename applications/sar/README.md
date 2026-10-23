# Standalone SAR applications

The five HSDK 5 C++ SAR formation workflows share the CPHD reader, range
compression, aperture manager, and backprojection operators. Each executable
owns its graph wiring, config validation, build gate, YAML, metadata, and
fixture smoke. Shared Graph driving, geometry conversion, reporting, mosaics,
and CUDA visualization live in `applications/common/radar_app/sar/`.

| Application | Former mode | Main behavior |
| --- | --- | --- |
| [spotlight](spotlight/README.md) | 1 | One reset image |
| [video_sar](video_sar/README.md) | 2 | Retained sliding aperture and signed eviction weights |
| [tiled_strip](tiled_strip/README.md) | 3 | Staged tile lattices and parent mosaic |
| [continuous_strip](continuous_strip/README.md) | 4 | Batch slicing, scroll geometry, and mosaic |
| [independent_images](independent_images/README.md) | 5 | Per-collection geometry, spacing, and frequency |

## Choose a run path

For immediate offline validation, run `./holoscan-radar test` from the
repository root **on the host**. The CLI builds the module in a container and
runs the registered suite, including all five SAR fixture smokes and headless
visualization checks. CTest generates the CPHD fixture in C++ and checks image
counts and dimensions. These are wiring and numerical smokes, not validation
of image quality on a real collection. For targeted SAR CTest commands inside
a development container, see the individual app guides.

The shipped YAML files are **real-data presets**. Download or subset the CPHD
files in the host `data/sar/` directory using [optional data preparation tools](../../tools/sar_data/README.md),
then run through the Holoscan CLI or directly inside the DevContainer. Relative
CPHD paths resolve under `HOLOHUB_DATA_DIR` (default: the repository `data/`
directory); the repository mount already exposes these files in both environments. Each app guide gives a complete
download-and-run command block. Read each app guide for
the exact collection, geometry, output, and display prerequisites. The
[CPHD reader](../../operators/io/cphd_reader_op/README.md) and
[SAR operator references](../../operators/sar/README.md) describe the library
contracts. The [current HSDK 5 SAR contract](../../docs/planning/SAR/OPERATOR_API.md)
takes precedence over older planning material.

Visualization is best-effort: if rendering falls behind, both SAR display
connections discard older frames and show newer images. Formation, image
reporting, and optional tensor dumping keep their normal delivery policy.
The paced CPHD reader, range/manager, and backprojection run in separate
partitions with dedicated `ThreadPlacement` lanes so collect-vs-wall lag is
the live playback clock (TxTime versus now), not GPU time on the controller.
Looping presets (`loop: true`) stop with Ctrl-C in the launch terminal; the
shared Graph driver owns SIGINT/SIGTERM so host `holoscan run` does not need
`docker kill`. One in-flight backprojection may finish before the process exits.
When Holoviz is enabled, the formed-image window composites the source-branch
on-image HUD (scene name, formed size, backprojection rate, last-pulse latency,
paced collect-vs-wall lag, and APC look chevron) as text and
geometry layers. A second `SAR timeline` window is a 20 s scrolling Gantt of
collection dwells, image-emit ticks with last-pulse latency labels, and a 1 Hz lag HUD.

The former `applications/sar_formation/` path and `sar_formation` executable
were replaced by these five entries. There is no compatibility launcher.
The host `./holoscan-radar test` command includes generated CPHD smokes and
headless visualization. The CPHD fixture is generated once by C++.
See the [SAR open-items roll-up](../../docs/planning/SAR/OPEN_ITEMS.md) for
tracked follow-ups and [SAR input data](../../docs/planning/SAR/INPUT_DATA.md) for optional real
collections and [SDK development](../../SDK_DEVELOPMENT.md) for the pinned image.
