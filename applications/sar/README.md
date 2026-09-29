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

For immediate offline validation, configure and build with testing enabled,
then run `ctest --test-dir build -R '^Sar_.*_(smoke|viz)$' --output-on-failure`.
CTest generates one CPHD fixture in C++, invokes all five applications, and
checks output image counts and dimensions as well as headless visualization.
This is a wiring and numerical smoke, not validation of image quality on a
real collection.

The shipped YAML files are **real-data presets**. Download or subset the CPHD
files in the host `data/sar/` directory using [optional data preparation tools](../../tools/sar_data/README.md),
then run through the Holoscan CLI with `data/sar/` mounted at the YAML's
`/tmp/sar` path inside the container. Each app guide gives a complete
download-and-run command block. Read each app guide for
the exact collection, geometry, output, and display prerequisites. The
[CPHD reader](../../operators/io/cphd_reader_op/README.md) and
[SAR operator references](../../operators/sar/README.md) describe the library
contracts. The [current HSDK 5 SAR contract](../../docs/planning/SAR/OPERATOR_API.md)
takes precedence over older planning material.

The former `applications/sar_formation/` path and `sar_formation` executable
were replaced by these five entries. There is no compatibility launcher.
Run `ctest --test-dir build -R '^Sar_' --output-on-failure` for generated CPHD
smokes and headless visualization. The CPHD fixture is generated once by C++.
See [SAR input data](../../docs/planning/SAR/INPUT_DATA.md) for optional real
collections and [SDK development](../../SDK_DEVELOPMENT.md) for the pinned image.
