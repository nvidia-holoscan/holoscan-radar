# Optional SAR data preparation

These Python command-line tools prepare **local CPHD files** for the five C++
SAR applications. They are not imported by an operator or application at run
time. CTest uses a C++ CPHD generator and does not install these dependencies
or download public collections.

On the **host**, install the optional dependencies in a Python environment
of your choice:

```bash
python3 -m pip install -r tools/sar_data/requirements.txt
```

If preparing data **inside a development container**, the pinned SDK image
lacks `ensurepip`; a local target also works:

```bash
python3 -m pip install --target .cache/sar-data-python -r tools/sar_data/requirements.txt
export PYTHONPATH="$PWD/.cache/sar-data-python${PYTHONPATH:+:$PYTHONPATH}"
```

## Host CLI data layout

Run these commands from the repository root **on the host**. Save downloads
under the ignored `data/sar/` directory. The shipped YAML presets use paths
such as `sar/imperial/...`, resolved under the build's `HOLOHUB_DATA_DIR`
(default: the repository `data/` directory). Both the host CLI and DevContainer
already mount this tree, so no additional bind mount is required. The paths
work even when the CLI runs the executable from its build directory.

For example, download the Imperial Valley collect and run continuous strip:

```bash
python3 tools/sar_data/pull_cphd.py --app continuous_strip \
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

On the **host**, list a manifest without touching the network:

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip --list
```

On the **host**, download the Capella Imperial Valley CPHD used by both strip
apps, or the six Umbra collections used by independent images. `--app all`
downloads both sets. With `--dest data/sar` it writes `data/sar/imperial/` and
`data/sar/mode5/`. Each transfer prints byte progress, rate, and an ETA on stderr.

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip --dest data/sar/imperial
python3 tools/sar_data/pull_cphd.py --app independent_images --dest data/sar/mode5
python3 tools/sar_data/pull_cphd.py --app all --dest data/sar
```

`--app continuous_strip` selects the same Capella key and default directory as
`tiled_strip`. Without `--dest`, the downloader writes to the host
`data/sar/imperial/` or `data/sar/mode5/`, matching the shipped presets.
Existing readable files are reused unless `--force` is set. The downloader
checks that independent-image collections have distinct image reference points. These are multi-gigabyte downloads;
inspect the manifest before running them. Spotlight uses a single Umbra file;
its [application guide](../../applications/sar/spotlight/README.md) gives a
direct `curl` command.

## Houston video subset

The video preset expects `data/sar/mode02_houston_1p0m.cphd` in the
repository. On the **host**, create it under `data/sar/` without downloading
the full 110 GiB signal block:

```bash
mkdir -p data/sar
python3 tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
  --dry-run -o data/sar
python3 tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
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
python3 tools/sar_data/subset_cphd.py \
  --source build/tests/cpp/generated_sar_smoke.cphd \
  --pulse-count 3 --sample-count 4 -o build/tests/cpp/subset_local.cphd
```

The generated file has three pulses and four samples. The C++ reader can
consume it through a fixture app YAML with `cphd.path` changed to
`build/tests/cpp/subset_local.cphd`. This check exercises parsing and data
flow; it does not replace validation against the Houston collection.
