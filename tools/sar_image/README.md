# Reference image formation

A GPU reference backprojector and an offline PNG renderer, used to check SAR
fixtures and — once `BackprojectionOp` exists — to check the operator.

```bash
pip install -r requirements.txt
python3 form_image.py /tmp/sar_fixtures/mode01.cphd -o /tmp/mode01.png
python3 form_image.py --image dump.npy --sidecar scene_truth.json -o out.png
python3 form_image.py --image atlanta.npy -o atlanta.png
```

It reads the ground-truth sidecar beside the CPHD (`<stem>_truth.json`) for
the image lattice. Predicted target pixels stay in the sidecar — the PNG is
a 1:1 greyscale raster of the formed samples (no axes, no overlay). The
check that energy landed on those pixels is `tests/python/sar/test_sar_sim.py`,
not the picture.

## Why this is a separate tool

It shares no code with [`../sar_sim`](../sar_sim/README.md), and that is
load-bearing rather than incidental. No CPHD binaries live in git, so every
SAR test stands on a simulator-generated fixture and on the sidecar's claim
about where the targets are. Checking that claim with the simulator's own
geometry code would assert nothing. Two implementations, written separately,
have to agree.

The same reasoning keeps `backproject()` a literal transcription of the
algorithm — compress in range, then sum each pulse's contribution at the
delay each pixel implies. That delay is the two-leg path (`TxPos` and
`RcvPos`) versus the same path to the SRP, halved to one-way metres.
It is deliberately *not* the single-APC midpoint `BackprojectionOp`
currently forms against. If it adopted that approximation, agreement
between the two would stop being evidence.

What it cannot do is establish agreement with the outside world; no
self-consistent pair can. That is what the phase 2 comparison against the
real Atlanta collection and the MatX `sar_bp` golden is for.

## Performance

Running on the GPU costs none of the independence above, since what matters
is authorship and algorithm rather than array library. On an RTX PRO 6000,
against the `mode01` fixture (2048 pulses x 2048 samples onto a 693 x 525
lattice):

| Implementation | Time |
| --- | --- |
| numpy | 64.5 s |
| cupy | 0.8 s |

The two agree to 1.1e-16 relative. The speedup matters because bring-up means
forming the same scene repeatedly while chasing a convention bug, and the
real collections are far larger than this fixture.

## Layout

```text
tools/sar_image/
  form_image.py        # CLI: CPHD + sidecar -> PNG, or --image dump.npy
  sar_image/
    backproject.py     # the reference backprojector
    render.py          # 1:1 dB greyscale raster
```

## Not here yet

- **A diff mode** against a golden image, for the MatX comparison.
