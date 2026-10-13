# DetectionReduction semantic output validation

The user authorized this validator on 2026-10-13 instead of adding fixed-order
GPU accumulation. Production code, formulas, precision and tie rules remain
unchanged. Historical exact-comparison failures remain historical failures.
The campaign is paused; this implementation does not establish a new timing
baseline or accept an optimization.

## Policy and invocation

Use `detection-reduction-v1` for **both** `rep_idx` and `reduced_points`, for
both precisions. Supply a rationale; no adjustable numerical tolerance is
accepted. The existing `radar_perf.py compare-outputs` command accepts source-bound
exports with this policy (campaign baseline collection binds build fingerprints
in memory without modifying raw captures):

```json
{
  "rep_idx": {
    "float": {"mode": "detection-reduction-v1", "rationale": "Input-derived semantic validation"},
    "double": {"mode": "detection-reduction-v1", "rationale": "Input-derived semantic validation"}
  },
  "reduced_points": {
    "float": {"mode": "detection-reduction-v1", "rationale": "Input-derived semantic validation"},
    "double": {"mode": "detection-reduction-v1", "rationale": "Input-derived semantic validation"}
  }
}
```

Fresh `perf_detection_reduction` native captures (`--verify-output`) include
actual per-frame tensor bytes and effective static/live parameters in
`semantic_inputs`. Cached matches cover frames 0/1; ordinary matches cover
frame 1; frame 2 takes Params. Both captures must contain identical witnesses.
Existing exports without these fields block this policy; they remain usable
for exact comparison. Session policy hashes bind the mode and rationale as
before. A new confirmed session and baseline are required for production work.

## Independent bounds

This model requires IEEE round-to-nearest arithmetic and precise division/sqrt
(the current default CUDA build). Fast-math/approximate arithmetic requires a
separately declared model and fresh validation. It is not silently covered.

The oracle uses exact rational endpoints, IEEE binary32/binary64 unit roundoff
`u = 2^-24 / 2^-53`, and an absolute smallest-normal allowance for flush-to-zero.
For each elementary operation, its exact interval is widened by
`u * max(abs(endpoints)) + smallest_normal`; exact zero stays zero. This
conservatively covers separate multiply/add and fused arithmetic without
requiring their results to match bit for bit.

For an unordered sum of `n` terms, the additional bound is
`(n*u*sum(max(abs(term endpoints))) + n*smallest_normal) / (1-n*u)`.
Weight conversion, products, row-score arithmetic, accumulation, reciprocal,
final multiplication, index conversion and distance arithmetic are included.
Possible overflow, `n*u >= 1`, a denominator not provably positive, or row
weights not provably nonnegative block validation. Bounds are derived solely
from original input witnesses and precision, never observed output differences.
They are conservative enclosures, not claims that every enclosed value or
representative can occur in an actual CUDA execution.

For medoid, feature centroids and distances use feature-scaled input rows.
For weighted mean, row weights derive from empty/uniform weights,
`weighted_sum`, `weighted_l2` or `weighted_max`; feature and index means use the
same weight model. Square roots use integer-square-root dyadic rational
enclosures followed by the precision rounding allowance. Custom scores remain
explicitly blocked pending caller-specific reference models. Uniform weights
do not need a score reference.

A row is admissible only if its distance lower bound does not exceed the
smallest upper bound among cluster members. A separated winner is therefore
unique. Receipts record absolute rational distance bounds, the admissible rows
and the selected rows for every cluster in both captures.

## Exact protections and limitations

- Shapes, dtype, memory placement, emission coordinates/counts, wiring,
  diagnostics, termination and occupied cluster ordering remain exact.
- Representatives must be real members of their corresponding cluster; noise
  is excluded. All preserved index columns participate in weighted selection.
- `first` selects the lowest input row. `peak` retains exact pair comparison
  and its existing analytic/custom-score contract fixtures.
- Gathered feature bytes must match the chosen input member; duplicate input
  indices require a matching admissible member/feature pair. Weighted feature
  means must independently lie within their input-derived bounds.
- Proven order-independent exact arithmetic and identical selection locations
  enforce lowest-row computed ties. Interval overlap is not an exact tie.
  Arbitrary computed ties cannot be reconstructed from output tensors alone:
  existing GPU tie fixtures remain required and the oracle does not claim to
  check every internal tie in every execution.
- Ownership, unsupported-mode errors, temporal joining and missing-live
  fallback still require the existing native/C++ fixtures. Output bytes alone
  cannot establish those properties.
- Alternative representatives can change `SnapshotGather` samples and later
  angle estimates materially. This policy does **not** relax any consumer
  policy. Affected consumer sessions remain blocked unless their own declared
  comparison rules pass.

## Validation evidence

Validation results and artifact manifests are recorded under
`build-optimization/operator-perf/detection-reduction-semantic/`.
Only fresh captures carrying input witnesses exercise this policy; no historical
artifact is rewritten or upgraded.

Fresh validation on the pinned HSDK 5 CUDA image passed **60/60** case/precision
combinations (30 recipes, both precisions, three frames per capture, two
captures: 360 native fixture frames). Exact comparison of the same captures
still fails for **13/60** combinations. This is semantic admissibility evidence,
not bitwise determinism, downstream equivalence, or timing qualification.

**82 host tests** pass, including 15 new semantic-validator tests; **40 native
DetectionReduction contract tests**, **one custom-score test** and **three
performance smokes** pass. The final rebuild was checked against an archive
of 27 resolved runtime libraries. Pinned changed-file and all-file checks pass.
The full non-performance C++ suite was not rerun; production sources are
unchanged. RootMusic's separate correctness blocker remains unresolved here.

The final native capture pair took about 5.2 seconds. Complete rational semantic
replay took about 357.2 seconds on the host; this untimed correctness work is
separate from performance sampling and agent working time.

### Casewise capture differences

Counts are changed emitted representative rows across three frames. Feature
values are maximum absolute differences between captures, **not tolerances**.
Every listed combination passed semantic validation.

| Case | Float changed representatives | Float max feature difference | Double changed representatives | Double max feature difference |
| --- | --- | --- | --- | --- |
| baseline-small | 0 | 0 | 0 | 0 |
| baseline-medium | 0 | 0 | 0 | 0 |
| baseline-large | 0 | 0 | 0 | 0 |
| first-medium | 0 | 0 | 0 | 0 |
| medoid-medium | 11 | 0.149999976 | 0 | 0 |
| weighted_mean-medium | 2 | 2.98023224e-07 | 0 | 4.4408921e-16 |
| l2-large | 0 | 0 | 0 | 0 |
| representative-small | 2 | 0.0900000334 | 0 | 0 |
| representative-medium | 4 | 0.0900000334 | 0 | 0 |
| representative-large | 1 | 0.0450000167 | 0 | 0 |
| score_method-small | 0 | 0 | 0 | 0 |
| score_method-medium | 0 | 0 | 0 | 0 |
| score_method-large | 0 | 0 | 0 | 0 |
| feature_weights-small | 0 | 0 | 0 | 0 |
| feature_weights-medium | 0 | 0 | 0 | 0 |
| feature_weights-large | 0 | 0 | 0 | 0 |
| weighted-weights-medium | 0 | 2.38418579e-07 | 0 | 3.33066907e-16 |
| representative-small-cached | 6 | 0.199999988 | 0 | 0 |
| representative-medium-cached | 9 | 0.13499999 | 0 | 0 |
| representative-large-cached | 2 | 0.0450000167 | 0 | 0 |
| score_method-small-cached | 0 | 0 | 0 | 0 |
| score_method-medium-cached | 0 | 0 | 0 | 0 |
| score_method-large-cached | 0 | 0 | 0 | 0 |
| feature_weights-small-cached | 0 | 0 | 0 | 0 |
| feature_weights-medium-cached | 0 | 0 | 0 | 0 |
| feature_weights-large-cached | 0 | 0 | 0 | 0 |
| weighted-weights-medium-cached | 0 | 2.98023224e-07 | 0 | 3.33066907e-16 |
| optional-minimal-small | 0 | 0 | 0 | 0 |
| optional-minimal-medium | 0 | 0 | 0 | 0 |
| optional-minimal-large | 0 | 0 | 0 | 0 |

### Example of admitted representative variation

`medoid-medium` / float, zero-based frame 0, cluster label 14: selected input rows 462 and 526. The input-derived admissible rows are `[462, 526]`. Their squared-distance intervals are:

| Input row | Distance lower bound | Distance upper bound |
| --- | --- | --- |
| 462 | 0.00719982368363 | 0.00720017086905 |
| 526 | 0.0071998301343 | 0.00720017045333 |

These overlapping intervals justify admitting either row under this policy;
they do not prove a computed exact tie. Each gathered feature row was separately
checked against the corresponding selected input row.
