# Measurement approach review — workload-dependent uncertainty

Historical rationale for the formal version-4 design. Its full statistical
process is now opt-in; [proportionate acceptance](acceptance.md) is the default
for routine profiling and optimization. Preserve the historical reasoning and
artifacts rather than applying it as a universal campaign requirement.

Review dated 2026-10-19; workflow adopted and implemented on 2026-10-20.
Use the [approved integrated confirmation protocol](confirmation.md) and explicit
version-4 [commands](../../../performance/optimization/CONFIRMATION.md). The core
change is direct A/B screening followed by fresh two-A/two-B confirmation, with
no standalone A/A admission or fixed dispersion veto. Statistical refinements
below explain the design; they do not authorize unimplemented methods, old
evidence migration or a new GPU budget. Preserve historical decisions under
their original rules. Host validation is not deployment qualification.

## What the latest session establishes

The [MultiSourceExpansion session](../../../docs/performance/optimizations/assessment-2026-10-13/multi-source-expansion-session-2026-10-19.md)
passed its executed correctness checks but stopped at inherited A/A dispersion
limits. Small float p99 was roughly 131–153 µs; four of six paired differences
exceeded 5%. The fixed 30-pair rule could no longer pass. In the same session,
15–22 ms processing cases had P90 absolute paired differences of 0.028–1.629%
for median/p90/p99. These are preliminary observations, not newly qualified
cases or evidence that a candidate improved.

The stop correctly followed the confirmed rule. The rule itself conflates
repeatability with whether inference about a candidate is possible. Merely
adding distribution analyses downstream did not repair that admission policy.

## Interpret variation before choosing a gate

Distinguish three questions:

| Question | Evidence and consequence |
| --- | --- |
| Are these valid measurements of the declared workload? | Check timing boundaries, completed frames, provenance, output readiness, devices, instrumentation, process ownership and environmental context. Missing or mismatched evidence can invalidate a comparison. |
| What variation does this workload exhibit? | Characterize within-run dependence, between-process variation, quantile estimation, order effects and changes across time windows. High variation alone does not establish invalidity or an operator bug. |
| What change can this experiment demonstrate? | Estimate effects and uncertainty at the actual independent experimental unit. Report limited sensitivity or inconclusive evidence instead of treating a repeatability percentage as a universal requirement. |

Observed variation includes real deployment behavior, not just timer error.
Scheduling, memory/cache state and other effects may contribute; timings alone
do not identify their causes. Retain stalls and chronology. Do not subtract an
assumed overhead, select only fast frames, or relabel real tail behavior as
measurement error.

There is no demonstrated intrinsic noise floor for this platform. A/A estimates
repeatability for a particular case, precision, horizon and context. Report it
in microseconds and percent, with uncertainty. Do not extrapolate it to another
operator or infer it solely from nominal runtime or kernel count.

## Recommended replacement

Remove fixed relative-dispersion **vetoes** from admission. Retain their observed
statistics as diagnostics and preserve legacy replay. Do not replace 5% with
another universal percentage, fixed microsecond allowance or a margin equal
to the observed A/A spread.

Use integrated A/A and B/B controls to assess the experiment: investigate
order/side confounding,
nonstationarity, forecast uncertainty and assess sensitivity to a declared
candidate effect. Large dispersion should usually widen intervals or require
more independent observations; it should not automatically forbid an A/B
experiment with a much larger plausible effect. Persistent confounding or an
unidentified changing target population still requires investigation or an
explicitly scoped conclusion. A non-significant drift test does not prove
stationarity.

Separate validity, statistical resolution and engineering usefulness. A large
change may be demonstrable on a noisy workload. A small change may remain
unresolved on a seemingly stable workload. Noise is not an allowable regression
margin; any engineering tolerance must be defined separately from data scatter.

## Estimand and experimental design

Use the [paired distribution definition](distributions.md): equal-weight mixtures
of repeat-run frame distributions; p50/p90/p99 per original pair and declared
horizon; median paired log-quantile ratios, positive meaning slower. This
replaces the old median-of-repeat-quantiles estimator only after adoption.

These are selected distribution summaries, not proof of equality of entire
latency distributions. Keep first-use/setup, cold, live transitions and finite
bursts separate. Include total measured time, achieved throughput, maxima and
stall frequencies in reports. Changing the median objective to mean time or
sustained throughput is a separate explicit policy decision.

Keep the complete paired execution block as the inferential unit. Frames in a
run are dependent observations, not independent experiments. Isolated processes
do not establish independence between blocks. Record temporal dependence and
choose blocks to address observed correlation. Additional runs from one session
do not automatically establish cross-session reproducibility.

Randomize and balance execution order within short paired blocks, with the
seed and assignment recorded before execution. Mere deterministic alternation
can remain confounded with periodic behavior. Declare the grouping of repeats
and randomization unit together; do not reshuffle individual frame labels.
Randomization helps separate treatment from time/order effects but does not
remove carryover or establish a particular quantile-effect model by itself.

Retain exact paired sign/order-statistic intervals as a conservative first
implementation for the declared median contrast, subject to their pair-level
assumptions. A cluster bootstrap or randomized paired test may be investigated,
but must specify its null, estimand and assumptions. A sharp-null permutation
test does not automatically provide a valid confidence interval for an arbitrary
population quantile effect. Do not promote an ordinary IID-frame KS test,
Mann–Whitney test or bootstrap as a shortcut around clustering or sparse tails.

## Decisions

Preserve the user's directional protection semantics and more-than-2% target
objective unless changed explicitly. For a valid simultaneous effect interval
`[L, U]`, with positive meaning slower:

| Result | Decision |
| --- | --- |
| A predeclared target has `U < -2%` | Demonstrated target improvement; every declared target must meet its objective. |
| A protected metric has `L > 0` | Demonstrated slowdown; reject. |
| A protected interval spans zero | Inconclusive and permitted under directional protection; report uncertainty. This is not proof of equivalence or bounded harm. |
| The estimate, interval or required evidence is invalid/missing/unbounded | Unresolved; do not silently convert it into protected success. Additional sampling or an explicitly confirmed scope change may be required. |

An inconclusive target cannot justify retaining an optimization. A noisy
protected case remains in the matrix and in the report. It does not disappear
because another case improved. Correctness, ownership, errors, descriptors,
ordering and authorized equivalence checks remain unchanged and mandatory.

Review multiplicity as one declared design: campaign operators, affected
consumers, unique endpoints, candidate attempts and interim looks. Deduplicate
literally identical endpoints, such as first-50 and full windows when both are
50 frames, while reporting both labels. Account for adaptive candidate selection
and stopping; independent validation does not by itself excuse unlimited looks.
For a concrete initial policy, reserve a campaign-wide 5% familywise error
budget, allocate it across the declared primary operator sessions and their
maximum candidate attempts before results, then divide each candidate's budget
across its actual unique endpoints, including consumers. A simple allocation is
`0.05 / (operator_count * (max_candidates_per_operator + 1))` per candidate, followed
by Bonferroni across that candidate's unique endpoints. The extra slot reserves
a final cumulative look; the implementation also splits affected-operator
comparisons conservatively. Derive the required
pair count from this budget; do not reuse the old family count or assume 30
pairs are sufficient. Fixed endpoint testing consumes one look; additional
looks require their own allocation. Unused budget may be reassigned only by a
predeclared spending scheme. This is a statistical error budget, separate from
wall-clock authorization.

Keep Bonferroni as the reference implementation. Holm testing is an option for
valid marginal tests; it is not automatically a replacement confidence-interval
procedure. Resampled maximum-statistic methods need their own strong-control
assumptions. Do not silently replace familywise protection with FDR.

## Sampling should resolve a decision

Use existing captures, paired A/B screens and their within-build diagnostics
to estimate detectable effects or detection probabilities for
specified alternatives, with uncertainty and assumptions. A single null dataset
cannot establish power against every candidate mechanism. Predeclare the
smallest target improvement of interest, protected semantics, maximum spend
and stopping design. Choose affordable sampling to answer that decision;
otherwise report it as unresolved. Do not demand that every operator attain
an arbitrary percentage of inter-run consistency before any optimization.

Use more frames when within-run distribution estimation is the limitation and
more independent blocks when between-run uncertainty dominates. Longer runs may
change cache/thermal/timing regimes; declare the horizon and never silently
increase warmup or discard early frames. Avoid an exhaustive frame/repeat grid
when existing captures answer the question.

At 50 measured frames, the expected number above a population p99 is only 0.5
under a stable common marginal distribution. Pooling three equal-length repeats
provides 150 observations but does not make them independent or establish tail
precision. Distinguish an empirical finite-window p99 from a precise estimate
of a deployment population p99. Use quantile uncertainty and observed dependence
to assess tail information; do not prescribe a universal tail sample count.

Use fixed planned endpoints for the first implementation. A later sequential
version must predeclare valid interim rules, such as alpha spending or suitable
confidence sequences; repeatedly inspecting ordinary 95% intervals until they
pass is invalid. A deterministic rejection certificate remains different from
stopping on a favorable significance result.

## Campaign workflow and implementation

1. Continue broad discovery/screening across all requested operators. Rank
   avoidable work using profiles and source mechanisms; the largest runtime is
   not automatically the best opportunity. Screening is not proof that an
   operator has no remaining optimization opportunity.
2. For a concrete candidate, declare the complete affected matrix and outcomes.
   Use workload-specific noise/sensitivity evidence to forecast a useful A/B
   experiment. Do not require an unrelated noisy case to pass a universal
   dispersion limit as a prerequisite to preparing that experiment.
3. Implement the replacement as a new policy version, including estimator,
   pairing/randomization, sample resolution and decision semantics. Test
   collection, replay, provenance, interruption and reports together. Keep old
   reporting/replay unchanged; do not upgrade historical evidence in place.
4. Validate offline first with existing captures and independent synthetic
   experiments. Include A/A nulls, additive and multiplicative changes,
   run-level shocks, autocorrelation, skewed/heavy tails, rare stalls,
   chronological drift, order confounding, discrete timings and mixed outcomes
   across protected endpoints. Scaling units alone must not change decisions.
   Check false detections, interval coverage and sensitivity, reporting Monte
   Carlo uncertainty. Resampling the same archive repeatedly is not independent
   held-out evidence. Artificial controls validate tools, not a real speedup.
5. Freeze the design before fresh held-out GPU validation under an explicitly
   approved budget. Include disparate workload families and fast/long cases;
   representative validation does not qualify unmeasured case matrices. Confirm
   concrete production-session policies before any fresh candidate edits.

The core workflow is implemented and host-tested. Next, use explicit caller
budgets for deployment assessment/validation and complete real native behavioral
coverage before a production candidate. Do not relaunch old A/A qualification or
claim host synthetic tests establish an operator gain.

## Statistical references

- [NIST randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm): blocking and randomization address nuisance variation.
- [NIST autocorrelated measurement uncertainty](https://www.nist.gov/publications/calculation-uncertainty-mean-autocorrelated-measurements): treating correlated observations as independent misstates uncertainty.
- [SciPy paired permutation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html): paired sample exchanges, null hypotheses and exchangeability assumptions.
- [R multiple-comparison documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html): Holm and Bonferroni provide familywise error control with valid marginal tests; FDR is a different objective.
