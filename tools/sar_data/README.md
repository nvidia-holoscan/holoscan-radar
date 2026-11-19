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
such as `sar/...`, resolved under the build's `HOLOHUB_DATA_DIR`
(default: the repository `data/` directory). Both the host CLI and DevContainer
already mount this tree, so no additional bind mount is required. The paths
work even when the CLI runs the executable from its build directory.

For example, download the Imperial Valley collect and run continuous strip:

```bash
uv run tools/sar_data/pull_cphd.py --app continuous_strip \
  --dest data/sar
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
| Tiled strip | Imperial Valley Capella | `uv run tools/sar_data/pull_cphd.py --app tiled_strip` | `data/sar/` |
| Continuous strip / autofocus Imperial | Same Imperial Valley Capella | `uv run tools/sar_data/pull_cphd.py --app continuous_strip` | `data/sar/` |
| Independent images | Six Umbra collections | `uv run tools/sar_data/pull_cphd.py --app independent_images` | `data/sar/` |
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
An explicit `--dest` overrides the application's destination, so a different
location also requires a matching `cphd.path` in your YAML.

`--app all` downloads the collections supported by **this script**: Kourou
for autofocus, Imperial Valley for both strip apps, and the six independent-image
collections. It downloads the shared Imperial Valley object only once.
It does not download Atlanta or prepare the Houston subset; use the separate
instructions in the table for those.

```bash
uv run tools/sar_data/pull_cphd.py --app all --list
uv run tools/sar_data/pull_cphd.py --app all --dest data/sar
```

All selected CPHD files go directly into `data/sar/`, without per-collection or
per-workflow subdirectories. With `--app all`, `--dest` replaces that directory
for every selected file; no subdirectory names receive special treatment.
To fetch only Imperial Valley, use `--app continuous_strip` or `--app tiled_strip`.
AWS, HTTPS, and subset preparation share one progress display on stderr:
completed/total bytes, percentage, rate, elapsed time, and ETA. Unknown totals
or estimates display `?` or `--:--`. Terminals update one line; redirected logs
receive a fresh line every two seconds, including while waiting for I/O.
Subset bytes count the selected signal payload (not the full source object);
resumed bytes count toward completion but not the current transfer rate.
Validation, skipped inputs, failures, and completion are labeled explicitly. Existing readable
files are reused unless `--force` is set. The downloader checks that the six
independent-image collections have distinct image reference points. These are
multi-gigabyte downloads; inspect the manifest with `--list` before fetching.

## Existing local data

This layout is a clean switch. Move existing CPHD inputs from collection
subdirectories directly into `data/sar/`, keeping their basenames. Check for
destination collisions before moving anything; never overwrite a different
file with the same name. Remove old directories only when empty. Rename any
previously prepared Houston subset to `houston_1p0m.cphd` (or `houston_0p5m.cphd`
for the optional finer subset), along with its matching `.subset.json` sidecar.
Apply the same path changes to custom YAML.
The applications and downloader do not search old locations or migrate data.
Full collection files retain their original vendor filenames and contents.

## Houston video subset

The video preset expects `data/sar/houston_1p0m.cphd` in the
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

Houston subset preparation retries connection failures, short responses, and
HTTP 408/429/500/502/503/504 up to five attempts per range. Retries use exponential
backoff with jitter and honor `Retry-After` up to 60 seconds. Permanent errors
(such as 403/404/412/416) and ignored or incorrect byte ranges fail immediately.
One bounded worker pool reuses its HTTP connections across the entire subset.

Subset preparation validates XML, PVP, support-array, and selected signal byte
spans against the source length and declared block sizes before fetching
payloads or creating partial-download state. Dimensions must be positive and
representable; PVP fields must fit their stride, and support-array element
sizes must match their format. A one-byte HTTP range probe establishes the
remote length, including for files shorter than the normal header probe.
Only the selected channel's PVP records are fetched, excluding block padding.
Signal rows wider than the staging target (64 MiB by default) are transferred in
sample-aligned pieces and checkpointed after each complete row. Interruption
inside a wide row refetches that row on resume; existing checkpoints remain
compatible. The staging target is not a total memory budget: valid XML, PVP,
and support-array representations can still consume proportional memory.

CPHD XML containing a `DOCTYPE` declaration is rejected, including declarations
without entities. Entity resolution and external DTD loading are disabled before
parsing, and parser network access is disabled. This policy also applies to dry
runs and forced replacement, before modifying an existing output or resume state.
Ordinary XML without a `DOCTYPE` keeps the existing validation and preparation path.

Recipe `name` must be a non-empty single filename component. Empty names,
`.` / `..`, path separators (including backslashes), and absolute or
drive-qualified paths are rejected rather than rewritten. Use `--output`
to select a directory or an explicit `.cphd` destination. Explicit output
selection does not bypass recipe-name validation.

Set `io_chunk_bytes` in a subset recipe, pass `--io-chunk-bytes 16777216`
to either preparation CLI, or call `subset_cphd(..., io_chunk_bytes=16777216)`
for a 16 MiB staging/transfer target. Values must be integers of at least
8 bytes, divisible by 8. CLI overrides take precedence over the recipe;
the default is 67108864 bytes. This controls signal transfer pieces and
staging chunks, not total memory or metadata sizes. It may be changed when
resuming: checkpoints still describe complete rows, and neither output bytes
nor source/window identity depends on chunk size. Plain object downloads
through `pull_cphd.py` do not use this subset-only setting.

Measured on GB10 with warm local NVMe/ext4 inputs, Python 3.13.13, NumPy 2.5.4,
and sarkit 1.12.0: nine alternating baseline/hardened process pairs, including
checkpoint flush/fsync, gave these median paired elapsed-time changes versus
`e1ae6390`. Output file checksums matched in every pair.

| Preparation case | Elapsed-time change | Median peak process memory |
| --- | --- | --- |
| Kourou local subset, 1024 pulses by 16000 samples | +7.8% | 181.7 to 181.6 MiB |
| Kourou loopback HTTP subset, 256 pulses by 8000 samples, 16 workers | -0.9% | 72.9 to 72.9 MiB |
| Synthetic two-row signal, 80 MiB per row | +12.5% | 286.6 to 126.6 MiB |

Positive means slower. The local case includes added per-row file-identity checks;
the wide-row case trades extra output-file checksum reads for about 56% less
peak memory. The HTTP difference is within observed variation, and loopback
does not characterize WAN behavior. These measure subset preparation, not
the CPHD writer operator or complete SAR Graph throughput.

Interrupted subset downloads now resume automatically when you rerun the same
`pull_cphd.py --config ...` or `subset_cphd.py ...` command. Keep both
`houston_1p0m.cphd.part` and `houston_1p0m.cphd.part.json`: the checkpoint advances
only after a complete chunk (about 64 MiB) is flushed to disk. At most the
incomplete/uncheckpointed chunk is fetched again. The final `.cphd` is replaced
only after all signal rows are complete; an existing completed file remains
usable during a replacement download. A small hidden destination lock file
prevents concurrent writers and is harmless after the process exits.

Resume validates the source identity, requested pulse/sample windows, encoded
CPHD metadata/layout, committed file length, and last chunk checksum. HTTP
sources need a strong ETag or Last-Modified validator; requests pin that version
while downloading. Changed requests/sources or damaged checkpoints are refused
with instructions to use `--force` for an explicit restart. The old downloader
deleted partial data on failure, so progress already lost by that version cannot
be recovered. The completed `.subset.json` provenance sidecar is unchanged.

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
uv run --no-project --with-requirements tools/sar_data/requirements.txt \
  python -m unittest discover -s tools/sar_data -v
```

These checks cover the catalog/config resolver plus transient and permanent
HTTP failures, range/source validation, bounded worker connections, interrupted
and process-terminated resume, corrupt/mismatched checkpoints, concurrent-writer
exclusion, publication recovery, and exact subset samples/PVPs. They use small
local files and a local HTTP server; no public collections are downloaded.
