# SAR collection simulator

Generates CPHD fixtures, and a ground-truth sidecar for each, from committed
recipes. This is the in-repo, public simulator the SAR data plan calls for
([`docs/planning/SAR/INPUT_DATA.md`](../../docs/planning/SAR/INPUT_DATA.md)):
users are expected to generate their own test input, so the generator cannot
be an external or internal-only dependency.

**No CPHD binaries live in git.** The recipe is the fixture. Every CI smoke
collection and every synthetic demo scene is produced by running this tool
over a YAML file in [`recipes/`](recipes/).

## Usage

```bash
pip install -r requirements.txt
python3 generate_cphd.py recipes/mode01_smoke.yaml -o /tmp/sar_fixtures
python3 subset_cphd.py recipes/mode02_houston.yaml -o /tmp/sar/mode2
```

`generate_cphd.py` writes two files into the output directory:

| File | What it is |
| --- | --- |
| `<name>.cphd` | CPHD 1.1.0 in the recipe's `storage.signal_format` — `CF8` by default, the same decode path as the pinned Atlanta collection |
| `<name>_truth.json` | Ground truth: target positions, resolutions, and the image lattice the fixture expects to be formed on |

Current recipes:

| Recipe | Size | Time | For |
| --- | --- | --- | --- |
| `mode01_smoke.yaml` | 1.1 MiB, 256 x 512 | < 1 s | CI; every convention the demo exercises, shrunk |
| `mode01.yaml` | 32 MiB, 2048 x 2048 | ~1 s | Mode 1 bring-up and the spotlight demo |
| `mode01_ci4.yaml` | 0.6 MiB, 256 x 512 | < 1 s | CI; `mode01_smoke` stored as `CI4` with `AmpSF` |
| `mode01_ci2.yaml` | 0.3 MiB, 256 x 512 | < 1 s | CI; the same, stored as `CI2` |

The public-collection **subsetter** is a sibling CLI. It does not synthesize: it
takes a local CPHD or an HTTPS URL and writes a new CPHD whose `SC0` / `FxC` /
`FxBW` are derived from the kept sample grid (`INPUT_DATA.md`). The kept window
is a recipe/CLI knob — not a constant in the tool. The live mode-2 YAML uses
the 1.0 m recipe (`mode02_houston_1p0m.yaml`); 0.5 m stays on disk. Omit an
axis to keep it whole.

```yaml
samples:
  count: 4096
  origin: center          # start | center | end
# or samples: [33389, 37485]
# or samples: {bandwidth_hz: 61.6e6, origin: center}
# pulses: {duration_s: 8.0, origin: center}
```

Houston is 208,753 Range requests at 128 keep-alive connections; pin that, do
not raise it.

```bash
# CI / local: slice a generated fixture (no network)
python3 subset_cphd.py --source /tmp/sar_fixtures/mode01_smoke.cphd \
    --pulse-count 40 --sample-count 128 -o /tmp/sar_fixtures/smoke_subset.cphd

# Mode 2: full-dwell Houston, window from the recipe (~6.84 GiB at count 4096)
python3 subset_cphd.py recipes/mode02_houston.yaml -o /tmp/sar/mode2
python3 subset_cphd.py recipes/mode02_houston.yaml --sample-count 2048 -o /tmp/sar/mode2

# 0.5 m / 1.0 m subbands from a local copy of the 110 GiB collect
python3 subset_cphd.py recipes/mode02_houston_0p5m.yaml \
    --source /tmp/sar/2024-10-30-16-10-33_UMBRA-06_CPHD.cphd -o /tmp/sar
python3 subset_cphd.py recipes/mode02_houston_1p0m.yaml \
    --source /tmp/sar/2024-10-30-16-10-33_UMBRA-06_CPHD.cphd -o /tmp/sar
```

The writer streams signal in 64 MiB chunks so a 0.5 m Houston subband (~52 GiB)
does not need to fit in RAM. Local files use one sequential reader; `connections`
applies to HTTP Range only.

| Recipe | Default window | For |
| --- | --- | --- |
| `mode02_houston.yaml` | centred 4096 of 70,875 samples, all pulses | Mode 2 video SAR at 4.09 m; change `samples.count` to try other bandwidths |
| `mode02_houston_0p5m.yaml` | centred 33505 samples, all pulses | ~0.5 m GSD, ~52 GiB, ~2.4 GB/s collect-time |
| `mode02_houston_1p0m.yaml` | centred 16753 samples, all pulses | ~1.0 m GSD, ~26 GiB, ~1.2 GB/s collect-time |

### The `storage` section

`storage.signal_format` picks how samples are written: `CF8` (big-endian
`complex<float>`, the default), `CI4` (two big-endian `int16`) or `CI2` (two
`int8`). The integer formats are **quantized per vector** — each pulse is
scaled so its peak reaches `storage.quantization_peak` of full scale, and the
reciprocal is written to that vector's `AmpSF` parameter, exactly as the
Capella collections do it. A reader that ignores `AmpSF` gets amplitudes that
are wrong per pulse rather than uniformly wrong, so it cannot be caught by
eyeballing an image.

That is the point of these two recipes: they let CI cover the integer decode
and scaling paths without a network fetch, and because they are quantized from
the same synthesis `mode01_smoke` uses, they can be asserted against it to
within an LSB. What they cannot do is validate the *convention* — a fixture we
scaled ourselves would agree with our own reading of `AmpSF` whether or not it
matches the standard. Real Capella files are what pins that; see the reader's
[`OPEN_ITEMS.md`](../../operators/io/cphd_reader_op/OPEN_ITEMS.md) item 9.

## Why the sidecar matters

A real collection can be imaged but not *checked* — there is no independent
statement of where a scatterer was. Here there is, and the sidecar records it
in the form a test consumes, including the `SampleGeometry` lattice and the
pixel index each target should focus to. Computing the lattice here rather
than duplicating it in the application YAML is deliberate: the predicted
indices are only valid for the lattice they were computed against, so the two
cannot be allowed to drift apart.

## Conventions, and the one that will bite

The signal is deramped phase history: for a scatterer at differential range
`dR` from the reference point, the stored phase at frequency `f` is
`-4 pi f dR / c`, paired with `Global/SGN = -1`. Transforming the frequency
axis to delay therefore takes the **positive**-exponent kernel, which numpy
spells `ifft`. Getting this backwards mirrors the image about the reference
point, and a mirrored image of a symmetric scene looks entirely plausible.

Two things guard it, and it is worth knowing what each does *not* cover:

- **The generate-time self-check** compresses one pulse and confirms each
  target lands at its predicted delay. It catches a disagreement between the
  phase model and the declared `SGN` — an internal inconsistency.
- **`tests/python/sar/test_sar_sim.py`** backprojects the fixture with
  [`tools/sar_image`](../sar_image/README.md), an independent implementation
  sharing no code with this one, and asserts the targets land on the
  sidecar's predicted pixels. That covers the whole chain: phase sign, `SGN`,
  image axes, and the ground-range projection.

Neither establishes agreement with the *outside world*, which no
self-consistent simulator can. That is what the phase 2 comparison against
the real Atlanta collection is for.

## Layout

```text
tools/sar_sim/
  generate_cphd.py     # CLI: recipe -> CPHD + sidecar
  subset_cphd.py       # CLI: local or HTTPS CPHD -> pulse/sample subset
  recipes/             # committed fixture and subset descriptions
  sar_sim/
    recipe.py          # recipe model + YAML loading
    geometry.py        # WGS84, ENU, platform track
    signal.py          # point-target phase-history synthesis
    cphd.py            # CPHD 1.1.0 XML + sarkit writing
    subset.py          # pulse window + frequency subband
    truth.py           # ground-truth sidecar
    generate.py        # recipe -> files, in one call
```

`sarpy` / `sarkit` are confined to this directory. No operator, test binary,
or application links against them.

## GPU

Phase-history synthesis runs on the GPU through cupy, which is what makes a
2048-pulse collection a sub-second job. The generate-time self-check stays on
the host deliberately: it costs milliseconds either way, and checking a GPU
result with a host implementation means a fault in one stack is not
reproduced by the other.

The fixtures are unchanged by this — synthesis accumulates in float64 and
casts to `complex64` once at the end, which rounds away the last-ulp
differences between host and device transcendentals, so a fixture generated
either way is bit-identical. `test_regeneration_is_deterministic` still
holds, and cross-platform stability remains the open question it already was
in [`INPUT_DATA.md`](../../docs/planning/SAR/INPUT_DATA.md).

## Not here yet

- **Elevated targets (layover).** `Target.up_m` must be 0, and a non-zero
  value is refused at recipe validation. The signal model handles height
  correctly — the phase history of an elevated target is right — but
  `truth.py` does not: it projects the target's ECEF offset onto the
  ground-plane lattice axes, which predicts the pixel *directly below* the
  target. What a ground-plane image actually shows is the pixel the target
  lays over onto, `h * tan(graze)` nearer the radar, and even that is
  first-order: an elevated point's range history matches no ground pixel
  exactly, so it lands slightly defocused rather than as a clean peak. A
  sidecar that predicts the wrong pixel is worse than one that refuses,
  because every assertion written against it passes for the wrong reason.
  Closing this means applying the layover shift in `truth.py`, deciding how
  much defocus the prediction is allowed to ignore, and adding an
  elevated-target recipe to cover it.
- **The image-derived scene mode**, for demos that need to look like
  something. Point targets are for metric assertions; a demo wants a scene
  with a reflector-count cap chosen against measured generation time.
- **Recipes for modes 2-5.** Mode 1 only for the *simulator*; Houston is a
  public-collection subset (`mode02_houston.yaml`), not a synthesized scene.
  Following the minimum-per-demo rule in
  [`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md).
