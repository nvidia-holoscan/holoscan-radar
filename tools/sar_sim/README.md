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
```

That writes two files into the output directory:

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
  generate_cphd.py     # CLI
  recipes/             # committed fixture descriptions
  sar_sim/
    recipe.py          # recipe model + YAML loading
    geometry.py        # WGS84, ENU, platform track
    signal.py          # point-target phase-history synthesis
    cphd.py            # CPHD 1.1.0 XML + sarkit writing
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

- **The image-derived scene mode**, for demos that need to look like
  something. Point targets are for metric assertions; a demo wants a scene
  with a reflector-count cap chosen against measured generation time.
- **Recipes for modes 2-5.** Mode 1 only, following the minimum-per-demo rule
  in [`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md).
- **The public-collection subsetter**, which will live beside this tool and
  share the `sarpy` dependency. It is a **mode 2** prerequisite: every
  application reads a local file, and Houston at 110 GiB cannot be pulled
  whole. Atlanta can — 28.59 GiB, read with `CPHDReaderOp.pulse_range` — so
  mode 1 needs no pull tooling.
