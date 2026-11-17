# SAR data preparation and saved-image viewing

These Python command-line tools prepare **local CPHD files** and view saved
NumPy images for the six C++ SAR applications. They are not imported by an operator or application at run
time. CTest uses a C++ CPHD generator and does not install these dependencies
or download public collections.

## View saved images

Run from the repository root **on the host**:

```bash
uv run tools/sar_data/view_image.py path/to/image.npy
```

This writes `image.png` beside the saved array and prints its
absolute path. It needs no display and works over SSH. `uv` automatically
provides NumPy and Matplotlib in an isolated environment; you do not need the
CPHD preparation dependencies below. The application and original `.npy` file
are unchanged. Existing PNG output files are overwritten.

For other files and options:

```bash
uv run tools/sar_data/view_image.py image.npy --range-db 80 -o figures/image.png
uv run tools/sar_data/view_image.py clean.npy corrected.npy --compare
uv run tools/sar_data/view_image.py image.npy --show
```

The default is grayscale log magnitude over 60 dB, with the brightest sample
at 0 dB. Complex input is shown by magnitude; real input is treated as
amplitude, not power. Comparisons write `<first-stem>_comparison.png` and use
one peak reference and one color scale across all panels, so a dim image
stays dim. Arrays must have matching shapes; compare the same scene and grid,
since the viewer does no registration or georeferencing. Axes are array rows
and columns, with row zero at the top and no smoothing between samples.

`--show` additionally opens an interactive plot when your Python environment
has an interactive Matplotlib backend and a working display. The PNG is still
saved. With a noninteractive backend, the tool reports that the PNG is saved
but interactive display is unavailable. Missing, nonnumeric, non-finite,
empty, and all-zero images produce a clear error rather than a misleading plot.
Phase-vector diagnostics are 1-D arrays and are not accepted as images.

Saved 2-D floating-point or complex images within a repository checkout also
log a shell-quoted viewer command after `TensorDumpOp` confirms the file was
written. Run that command on the host from the repository root; its relative
path works across the CLI's host/container repository mount.

Offline viewer checks (no GPU or external collections needed):

```bash
uv run tools/sar_data/viewer_tests/test_view_image.py
```

## Data preparation setup

Use `uv run` for either preparation script; it installs the declared dependencies
in an isolated tool environment without changing the repository's `.venv`:

```bash
uv run tools/sar_data/pull_cphd.py --app autofocus --list
uv run tools/sar_data/subset_cphd.py --help
```

For direct `python3` invocation instead, install `tools/sar_data/requirements.txt`
in your chosen Python environment. The commands below use `uv run`.

Prepare data on the host; both the host CLI and development container can
then use the downloaded files. No Python dependency installation inside the
SDK container is needed for the C++ applications or generated-fixture tests.

## Host CLI data layout

Run these commands from the repository root **on the host**. Save downloads
under the ignored `data/sar/` directory. The shipped YAML presets use paths
such as `sar/imperial/...`, resolved under the build's `HOLOHUB_DATA_DIR`
(default: the repository `data/` directory). Both the host CLI and DevContainer
already mount this tree, so no additional bind mount is required. The paths
work even when the CLI runs the executable from its build directory.

For example, download the Imperial Valley collect and run continuous strip:

```bash
uv run tools/sar_data/pull_cphd.py --app continuous_strip \
  --dest data/sar/imperial
./holoscan-radar run continuous_strip
```

The pinned image and Holoscan CLI setup are in the
[developer guide](../../DEVELOPER.md#holoscan-cli). Each
[standalone app guide](../../applications/sar/README.md) has its own complete
download-and-run command block. If your CPHDs live elsewhere, use absolute
paths in a custom YAML or configure `-DHOLOHUB_DATA_DIR=/path/to/data` inside
the container, keeping the `sar/` layout beneath it. External host directories
must also be mounted into the container. Existing paths relative to the working
directory or YAML directory take precedence over the data-directory fallback.

## Public collection downloads

The selected YAML determines the required inputs. From the host repository root:

```bash
uv run tools/sar_data/pull_cphd.py --config applications/sar/autofocus/config/autofocus_imperial.yaml --list
uv run tools/sar_data/pull_cphd.py --config applications/sar/autofocus/config/autofocus_imperial.yaml
```

Use the same config you pass to the application. This handles every shipped SAR
preset, including the autofocus Atlanta, Imperial, and Houston variants. The
manifest must match every `cphd.path` entry, including its directory; unknown
paths fail before any download or output-directory creation. To use another data
root, pass `--data-root /path/to/data` or set `HOLOHUB_DATA_DIR`, preserving the
`sar/` layout and the application's matching data-root setting. Existing files
relative to the working directory or YAML directory retain application precedence;
if they point outside the registered layout, selection fails rather than downloading
another file. Custom datasets need a catalog entry in `pull_cphd.py`.

`--list` validates and prints the complete plan offline. Houston runs the existing
`houston_1m.yaml` subset recipe with HTTP range reads, avoiding the full raw
collection download. `uv run` supplies the downloader's YAML dependency and,
only when preparing Houston, the subset tool's separate dependencies.

`--app` selects an application's default YAML. Existing `--dest` overrides remain
available for these shortcuts; the application must then be configured to use that
location. With `--config`, use `--data-root` so configured paths stay authoritative.
`--app all` retains its historical eight-object selection (Kourou, Imperial, and
six independent images); Atlanta and Houston require explicit selection.

On the **host**, list a manifest without touching the network:

```bash
uv run tools/sar_data/pull_cphd.py --app tiled_strip --list
```

Choose the command for the preset you want to run. Destinations below are
relative to the repository root; omit `--dest` to use each downloader default.
The commands use isolated dependencies as described above.

| Application / preset | Collection | Download / preparation command | Destination |
| --- | --- | --- | --- |
| Autofocus default | Kourou Umbra, 2023-06-24 | `uv run tools/sar_data/pull_cphd.py --app autofocus` | `data/sar/` |
| Tiled strip | Imperial Valley Capella | `uv run tools/sar_data/pull_cphd.py --app tiled_strip` | `data/sar/imperial/` |
| Continuous strip / autofocus Imperial | Same Imperial Valley Capella | `uv run tools/sar_data/pull_cphd.py --app continuous_strip` | `data/sar/imperial/` |
| Independent images | Six Umbra collections | `uv run tools/sar_data/pull_cphd.py --app independent_images` | `data/sar/mode5/` |
| Spotlight / autofocus Atlanta | Atlanta Umbra | `uv run tools/sar_data/pull_cphd.py --app spotlight` | `data/sar/` |
| Video SAR / autofocus Houston | Houston FX subset | `uv run tools/sar_data/pull_cphd.py --app video_sar` | `data/sar/` |

For the default autofocus workflow, download Kourou and launch:

```bash
uv run tools/sar_data/pull_cphd.py --app autofocus --list
uv run tools/sar_data/pull_cphd.py --app autofocus
./holoscan-radar run autofocus --language cpp
```

This downloads about 4.19 GiB to
`data/sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd`, matching the default
[autofocus preset](../../applications/sar/autofocus/config/autofocus.yaml).
Do not use `--dest data/sar/imperial` for Kourou: that directory is for the
alternate Imperial Valley collection. An explicit `--dest` overrides the
single application's destination, so a different location also requires a
matching `cphd.path` in your YAML.

`--app all` downloads the collections supported by **this script**: Kourou
for autofocus, Imperial Valley for both strip apps, and the six independent-image
collections. It downloads the shared Imperial Valley object only once.
It does not download Atlanta or prepare the Houston subset; use the separate
instructions in the table for those.

```bash
uv run tools/sar_data/pull_cphd.py --app all --list
uv run tools/sar_data/pull_cphd.py --app all --dest data/sar
```

With `--app all`, `--dest` replaces the `data/sar/` root: Kourou goes directly
under it, and the other sets go under `imperial/` and `mode5/`.
Use `--dest data/sar`, not `--dest data/sar/imperial`; the script rejects the
known collection subdirectories as an `all` destination before downloading.
To fetch only Imperial Valley, use `--app continuous_strip` or `--app tiled_strip`.
Each transfer prints byte progress, rate, and an ETA on stderr. Existing readable
files are reused unless `--force` is set. The downloader checks that the six
independent-image collections have distinct image reference points. These are
multi-gigabyte downloads; inspect the manifest with `--list` before fetching.

## Houston video subset

The video preset expects `data/sar/mode02_houston_1p0m.cphd` in the
repository. On the **host**, create it under `data/sar/` without downloading
the full 110 GiB signal block:

```bash
mkdir -p data/sar
uv run tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
  --dry-run -o data/sar
uv run tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
  -o data/sar
```

The source is the public Houston HTTPS object named by the recipe. The tool
reads selected byte ranges and writes the central 16,753-sample FX subband
for the full dwell. Output is about 27 GiB. `--source` replaces the recipe's
remote source with a local CPHD; `--force` rebuilds an existing matching file.
The reader supports the tool's CF8 output. Keep the sample window and the
video preset's lattice and aperture settings together when changing GSD.

For an immediate offline check of any app, run its `Sar_<app>_smoke` CTest.
CTest generates its small CPHD fixture once in C++ and asserts image count
and dimensions. Passing that test does not validate real-collection focus.

## Generated local subset check

A small local subset can exercise the data tool without fetching public
data. First open the devcontainer or run `./holoscan-radar run-container`
**on the host**, then build the C++ fixture generator **inside the container**:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build --target generate_cphd_fixture
./build/tests/cpp/generate_cphd_fixture build/tests/cpp/generated_sar_smoke.cphd
```

The repository is mounted into the container, so the generated file is also
available **on the host**. With the optional Python dependencies installed,
run:

```bash
uv run tools/sar_data/subset_cphd.py \
  --source build/tests/cpp/generated_sar_smoke.cphd \
  --pulse-count 3 --sample-count 4 -o build/tests/cpp/subset_local.cphd
```

The generated file has three pulses and four samples. The C++ reader can
consume it through a fixture app YAML with `cphd.path` changed to
`build/tests/cpp/subset_local.cphd`. This check exercises parsing and data
flow; it does not replace validation against the Houston collection.

## Downloader checks

Run the offline CLI regressions from the repository root:

```bash
python3 -m unittest discover -s tools/sar_data -v
```

These checks cover the autofocus manifest, shared collection layout, invalid
`all` destinations, and HTTP failure reporting without downloading data.
