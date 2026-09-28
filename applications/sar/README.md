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

The former `applications/sar_formation/` path and `sar_formation` executable
were replaced by these five entries. There is no compatibility launcher.
Run `ctest --test-dir build -R '^Sar_' --output-on-failure` for generated CPHD
smokes and headless visualization. The CPHD fixture is generated once by C++.
See [SAR input data](../../docs/planning/SAR/INPUT_DATA.md) for optional real
collections and [SDK development](../../SDK_DEVELOPMENT.md) for the pinned image.
