# Optional SAR data preparation

These Python command-line tools prepare **local CPHD files** for the five C++
SAR applications. They are not imported by an operator or application at run
time. CTest uses a C++ CPHD generator and does not install these dependencies
or download public collections.

Install the optional dependencies in a Python environment of your choice:

```bash
python3 -m pip install -r tools/sar_data/requirements.txt
```

On the pinned SDK container, which lacks `ensurepip`, a local target also works:

```bash
python3 -m pip install --target .cache/sar-data-python -r tools/sar_data/requirements.txt
export PYTHONPATH="$PWD/.cache/sar-data-python${PYTHONPATH:+:$PYTHONPATH}"
```

## Host CLI data layout

Run these commands from the repository root. The shipped YAML presets refer to
`/tmp/sar` **inside the app container**. Save downloads under the ignored host
`data/sar/` directory, then pass
`--docker-opts="--mount=type=bind,src=$PWD/data/sar,dst=/tmp/sar,readonly"`
to `holoscan run`. The CLI already mounts the repository at `/workspace/radar`,
but it does not mount the host's `/tmp/sar` directory or symlink. `--add-volume`
mounts under `/workspace/volumes` and does not match the shipped YAML paths.

For example, download the Imperial Valley collect and run continuous strip:

```bash
python3 tools/sar_data/pull_cphd.py --app continuous_strip \
  --dest data/sar/imperial
holoscan run continuous_strip \
  --img holoscan-radar-dev:hsdk5 --no-docker-build \
  --configure-args=-DBUILD_ALL=OFF \
  --configure-args=-DHOLOSCAN_RADAR_BUILD_TESTING=OFF \
  --configure-args=-Dholoscan_DIR=/opt/nvidia/holoscan/lib/cmake/holoscan \
  --docker-opts="--mount=type=bind,src=$PWD/data/sar,dst=/tmp/sar,readonly"
```

The pinned image and Holoscan CLI setup are in the
[developer guide](../../DEVELOPER.md#holoscan-cli). Each
[standalone app guide](../../applications/sar/README.md) has its own complete
download-and-run command block. If your CPHDs live elsewhere, bind that host
folder to `/tmp/sar` with the same internal layout. The examples use a
read-only mount; if `dump: true` writes next to a CPHD, set a writable
`output.path` or use a writable mount.

## Public collection downloads

List a manifest without touching the network:

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip --list
```

Download the Capella Imperial Valley CPHD used by both strip apps, or the six
Umbra collections used by independent images. `--app all` downloads both sets.
With `--dest data/sar` it writes `data/sar/imperial/` and `data/sar/mode5/`.
Each transfer prints byte progress, rate, and an ETA on stderr.

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip --dest data/sar/imperial
python3 tools/sar_data/pull_cphd.py --app independent_images --dest data/sar/mode5
python3 tools/sar_data/pull_cphd.py --app all --dest data/sar
```

`--app continuous_strip` selects the same Capella key and default directory as
`tiled_strip`. Without `--dest`, the downloader writes to the host
`/tmp/sar/imperial/` or `/tmp/sar/mode5/`; those paths are not automatically
visible inside the CLI container. The explicit `--dest` values above match the
bind mount and keep the shipped YAML unchanged. If you choose another host
layout, mount its root at `/tmp/sar` or change the application YAML paths.
Existing readable files are reused unless `--force` is set. The downloader
checks that independent-image collections have distinct image reference points. These are multi-gigabyte downloads;
inspect the manifest before running them. Spotlight uses a single Umbra file;
its [application guide](../../applications/sar/spotlight/README.md) gives a
direct `curl` command.

## Houston video subset

The video preset expects `/tmp/sar/mode02_houston_1p0m.cphd` inside the
container. Create it under the host `data/sar/` directory without downloading
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

After building the C++ fixture generator in the pinned GPU image, a small
local subset can exercise the same CLI without fetching public data:

```bash
./build/tests/cpp/generate_cphd_fixture build/tests/cpp/generated_sar_smoke.cphd
python3 tools/sar_data/subset_cphd.py \
  --source build/tests/cpp/generated_sar_smoke.cphd \
  --pulse-count 3 --sample-count 4 -o build/tests/cpp/subset_local.cphd
```

The generated file has three pulses and four samples. The C++ reader can
consume it through a fixture app YAML with `cphd.path` changed to
`build/tests/cpp/subset_local.cphd`. This check exercises parsing and data
flow; it does not replace validation against the Houston collection.
