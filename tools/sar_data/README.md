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

## Public collection downloads

List a manifest without touching the network:

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip --list
```

Download the Capella Imperial Valley CPHD used by both strip apps, or the six
Umbra collections used by independent images:

```bash
python3 tools/sar_data/pull_cphd.py --app tiled_strip
python3 tools/sar_data/pull_cphd.py --app independent_images
```

`--app continuous_strip` selects the same Capella key and default directory as
`tiled_strip`. `--dest` changes the directory; if you use it, update the
application YAML's `cphd.path`. Existing readable files are reused unless
`--force` is set. The downloader checks that independent-image collections
have distinct image reference points. These are multi-gigabyte downloads;
inspect the manifest before running them. Spotlight uses a single Umbra file;
its [application guide](../../applications/sar/spotlight/README.md) gives a
direct `curl` command.

## Houston video subset

The video preset expects `/tmp/sar/mode02_houston_1p0m.cphd`. Create it
without downloading the full 110 GiB signal block:

```bash
python3 tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
  --dry-run -o /tmp/sar
python3 tools/sar_data/subset_cphd.py tools/sar_data/recipes/houston_1m.yaml \
  -o /tmp/sar
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
