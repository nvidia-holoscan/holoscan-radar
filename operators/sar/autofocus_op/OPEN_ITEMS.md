# AutofocusOp — open items

## Implementation follow-ups

1. **One full-aperture PGA estimate does not hold up over long
   apertures, and the behaviour is not monotonic in aperture
   length.** Measured on Kourou (UMBRA-05) at the `autofocus`
   preset's settings, 2048² product, 4 iterations, varying only
   `pulse_range` / `max_aperture_pulses`:

   | pulses | gain | first rejection |
   | --- | --- | --- |
   | 8000 | 1.096× | none; converges over all four iterations |
   | 12000 | 1.000× | iteration 0, sharpness 9.296e19 → 6.535e19 |
   | 16000 | 1.491× | iteration 1, 3.214e20 → 2.551e20 |
   | 20000 | 1.000× | iteration 0, 3.734e20 → 2.965e20 |
   | 23375 (full) | 1.000× | iteration 0, 6.159e20 → 4.066e20 |

   Where the gain is 1.000× the *first* estimate lowered sharpness
   and the guard refused it, so nothing was ever applied — the
   operator is behaving as specified, and the estimate itself is
   what is wrong. This is not `window_fraction`: at the full
   aperture, 256 / 1023 / 2047 / 4096 / 8181 samples all give
   1.000×. It is not precision either: at 8000 pulses `<float>`
   and `<double>` agree to four significant figures at every
   iteration (see `STATUS.md`, 2026-10-29).

   The shape of it suggests PGA wants a coarse-to-fine schedule
   over a long aperture rather than one estimate across the whole
   of it — plausibly sub-aperture estimates composed, or a
   per-iteration window that narrows. **Not** resolved here: this
   needs an algorithm decision, not a parameter change.

   Worth noting why it surfaced only now. The preset ships with the
   full 23375-pulse aperture, which declared 128.4 GiB of pools
   against 95.6 GiB of GPU and could never reach `start()`; the
   single-precision sample path (66.9 GiB) is what made the
   configuration runnable, so no one had seen its output before.

2. **`num_iterations` is bounded at 255 by the contract's `uint8`,
   which has not been exercised anywhere near that.** The preset
   uses 4 and the sweeps above use 4. Whether the guard-and-keep
   behaviour is still sensible over tens of iterations is unknown.
