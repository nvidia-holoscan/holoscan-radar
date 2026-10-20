# Per-frame Distribution Analysis

Use this statistical definition with the approved
[screening and integrated confirmation workflow](confirmation.md). The maintained
archive analysis tools remain exploratory. Version-4 confirmation implements
this estimand and fresh held-out acceptance; see the
[command guide](../../../performance/optimization/CONFIRMATION.md). Host validation
is not deployment qualification. Preserve historical policies and decisions.

## Population And Statistic

- Declare case, precision, deployment context and measured processing horizon.
  Keep the first 50 measured frames and the complete declared horizon separate.
  A finite burst and sustained operation are different populations when timing
  varies with frame position. All original warmup boundaries remain explicit.
- Within each A or B side of a complete confirmation block, form the equal-weight
  mixture of its two isolated run empirical frame distributions for each window.
  Equal-length runs can be concatenated to compute this mixture's quantiles.
  This weighting does not make frames independent replicates.
- Use p50, p90 and p99 of that mixture as the primary distribution summaries.
  They expose central and tail behavior without claiming to characterize every
  possible distribution difference. Retain maxima, stall counts and chronology.
  Other quantiles, whole-CDF distances and bootstrap analyses are exploratory
  unless separately declared and validated. Do not select an estimator because
  it produces a narrower interval or more favorable result.
- For confirmation block `i`, quantile `q` and window `w`, compute
  `d_i(q,w) = log(Q_after_i(q,w) / Q_before_i(q,w))`.
  Estimate `100 * expm1(median(d_i))` with exact sign/order-statistic intervals
  for the population median of paired log ratios. Positive means slower.
  This is a typical-block mixture-quantile effect, not the quantile of all blocks
  pooled together, a mean-frame effect, or a median of repeat-run quantiles.
- The inference unit is the complete four-run block. Assumptions
  include independence between blocks and a stable population median of their
  contrasts; isolation alone does not establish either assumption. Never use
  IID-frame tests or frame resampling to manufacture independent observations.

## Decisions And Multiplicity

The user-selected protection semantics remain: block a statistically
demonstrated slowdown; an inconclusive protected result is allowed. For a
finite simultaneous interval `[L, U]` in the positive-is-slower convention:

| Evidence | Interpretation |
| --- | --- |
| `L > 0` | Demonstrated slowdown; blocks a protected metric. |
| `L <= 0 <= U` | Inconclusive; permitted for protection, but not proof of equivalence or bounded harm. |
| `U < 0` | Demonstrated speedup. |
| `U < -2%` | Meets the campaign's more-than-2% objective, if this is a predeclared target. |
| Missing/unbounded interval | Insufficient interval resolution; blocks validation rather than counting as an ordinary inconclusive result. |

All predeclared targets must meet their objective, and every protected
case/precision/window/quantile is checked. Account for the complete campaign,
affected consumers, unique metrics/windows, candidate attempts and interim looks in the familywise
budget. The existing exact Bonferroni intervals permit dependence between
comparisons; they still require valid block-level assumptions. Finite bounds
and +/-1% diagnostic width do not establish power. Do not replace directional
protection with an equivalence or non-inferiority requirement without approval.

New-policy A/A and B/B controls share the block's runs with A/B. Preserve this
joint dependence; control contrasts are not extra independent samples. Their
role is to diagnose measurement interpretation, not to pass fixed P90-delta/IQR
limits. A non-significant control is not proof of stationarity or adequate power.
Unsupported evidence or unresolved confounding can block acceptance; ordinary
scatter does not automatically do so. Do not subtract a noise estimate or turn
it into an allowed regression margin.

Version-2/version-3 production commands still use aggregated repeat metrics and
legacy A/A admission. Use version 4 for this workflow. See
[compatibility](tools.md#workflow-and-tool-compatibility); do not reinterpret a
historical rejection or relabel exploratory output as new acceptance evidence.

## Diagnose Before Enlarging Samples

Use `performance/optimization/distributions.py` with hash-bound version-3 raw
pairs. See [tool documentation](../../../performance/optimization/FRAME_COUNT.md).
Compare declared equal-length chronological windows against the first window
and adjacent windows. Temporal comparisons match the same runs, pool both A/A
sides within each original pair, and retain that pair as the inference unit.
Declare all window looks in the exploratory family, including earlier analyses
when interpreting them together. Overlapping comparisons remain dependent.

Separate within-run quantile estimation from between-process repeatability.
More frames may help tails while changing the measured timing regime; more
independent pairs address a different uncertainty. A prefix of a long run is
not a fresh short run. Do not infer missing samples or pair counts.

Use exploration to select a horizon/protocol, then freeze it and validate on
fresh disjoint processes. Do not claim stationarity because adjacent intervals
include zero, select an unvalidated steady cutoff, discard early frames, or
increase warmup silently. Chronological changes do not establish a hardware,
SDK or kernel cause without attribution evidence.

Existing archive tools below retain their original pair-based schemas; they do
not collect four-run blocks or implement integrated acceptance. Keep historical
analysis labels, orientation and identities intact. New tooling must explicitly
encode the block grouping and randomized order, with fresh held-out confirmation.
