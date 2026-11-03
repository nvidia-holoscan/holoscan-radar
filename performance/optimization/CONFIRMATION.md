# Version-4 integrated candidate confirmation

This guide documents the optional automated **formal statistical** workflow.
Routine profiling and optimization default to the smaller
[engineering acceptance process](../../skills/operator-perf/references/acceptance.md).
That reviewed process is not implemented by this CLI's acceptance state machine;
do not turn its small-sample evidence into a passing version-4 gate.

Use paired A/B screening, then fresh four-run confirmation blocks with integrated
A/A and B/B controls. Version 4 removes standalone A/A admission and fixed
repeatability thresholds. Version 2/3 collectors, calibration and historical
replay remain unchanged. This interface is host-tested with synthetic subprocesses;
new GPU/deployment qualification is not claimed by those tests.

## Configuration and preparation

Select `measurement_policy_version: 4` explicitly. Keep existing output policies,
verification cases, timing precisions, editable scope and candidate limits. Use:

```json
{
  "measurement_policy_version": 4,
  "pairs": 32,
  "screen_pairs": 2,
  "min_effect_pct": 2.0,
  "wall_time_s": 3600,
  "sampling": {
    "scope": "thorough",
    "sizes": "all",
    "cases": "baseline-small,baseline-medium,baseline-large",
    "warmup": 5,
    "frames": 50,
    "repeats": 2,
    "early_frames": 50,
    "raw_frame_times": true,
    "process_mode": "isolated",
    "aggregation": "mixture_quantiles",
    "pairing_order": "randomized_four_run",
    "random_seed": 20261020
  }
}
```

This is a fragment, not a complete operator configuration or time authorization.
`pairs` counts confirmation **blocks**, each with two A and two B run processes
per case (and separate precision subprocesses when using `paired_precision.py`).
`screen_pairs` counts two-run A/B screening blocks. Example counts/budgets are
illustrative. Obtain an explicit caller budget; forecast all phases, and use
`radar_perf.py budget` on a complete baseline preview to derive block resolution.
The proposal includes campaign/consumer multiplicity, maximum candidate attempts
and one cumulative-validation look; finite bounds are not a power guarantee.

Version 4 rejects legacy `max_aa_delta_pct`, `max_baseline_spread_pct` and
per-metric dispersion policies rather than silently applying or ignoring them.
It currently supports latency objectives; throughput remains a reported
harness diagnostic. Confirm each consumer's `consumer_sampling` when its case
IDs differ. Do not migrate existing sessions or rewrite their evidence.

Campaign discovery and `qualify` resolve the version-4 design without launching
standalone A/A. `campaign baseline` needs an explicit positive
`baseline_budget_seconds`; it collects native correctness repeats and one clean
baseline screen per case. Coverage receipts need hashed nonempty fixture evidence
and the existing ownership/boundary/analytic/error/live coverage declarations.
`campaign prepare-sessions --confirmed` validates native evidence and prepares
version-4 sessions without A/A calibration. Baseline screening is not a claim
that an operator has no optimization opportunities.

## Candidate commands

Prepare a candidate plan and verified source checkpoint before editing. Use the
campaign-generated session configuration for all commands, not an earlier leaf
configuration lacking its admission reference. Generic checkpoint/restore,
contract, coverage, tests and exact native-output gates still apply.

```bash
python3 performance/optimization/radar_perf.py plan \
  --session /absolute/session.json --root /absolute/checkout --record /absolute/plan.json
python3 performance/optimization/radar_perf.py measure \
  --before /absolute/original-build.json --after /absolute/candidate-build.json \
  --config /absolute/session-config.json --session /absolute/session.json \
  --plan /absolute/plan.json --phase screen --out /absolute/screen
python3 performance/optimization/radar_perf.py measure \
  --before /absolute/original-build.json --after /absolute/candidate-build.json \
  --config /absolute/session-config.json --session /absolute/session.json \
  --plan /absolute/plan.json --phase confirmation \
  --screening /absolute/screen/blocks.json --out /absolute/confirmation
python3 performance/optimization/radar_perf.py inspect-confirmation \
  --blocks /absolute/confirmation/blocks.json --config /absolute/session-config.json \
  --plan /absolute/plan.json --out /absolute/analysis.json
```

Freeze candidate manifests and settings before confirmation. The collector seals
order, seed, plan, screening identities and freeze time before measured launches.
It checks artifact/runtime hashes, fresh process receipts, context and raw frames,
serializes per-device jobs, and cleans up owned processes/containers. Resume
uses the same output directory, verified completed units and the original freeze;
unsealed outputs require quarantine, not invented receipts. An explicit total
session budget bounds both screening and confirmation.

`inspect-confirmation` reconstructs all observations and produces A/B effect
intervals plus dependent A/A and B/B diagnostics. Positive effect means slower.
Its `passed` remains false until an explicit evidence-backed control review is
supplied. The review is an agent assessment, not an extra user approval or a
fixed variability test. For example:

```json
{
  "index": {"path": "/absolute/confirmation/blocks.json", "sha256": "ACTUAL_SHA256"},
  "candidate_plan_sha256": "ACTUAL_PLAN_SHA256",
  "interpretable": true,
  "rationale": "Explain the observed controls, chronology and deployment evidence; identify any confounding or limitations.",
  "evidence": [{"path": "/absolute/analysis.json", "sha256": "ACTUAL_ANALYSIS_SHA256"}]
}
```

Use actual hashes; this is a schema illustration. Review control differences in
microseconds/percent, the complete A/B sequence and deployment/phase context.
A non-significant control is not proof of reliability. High control dispersion
alone is not a failure or a regression allowance. Set `interpretable: false`
when unresolved confounding prevents attribution. A favorable review cannot
rescue failed statistical or correctness gates. Preserve all unfavorable data;
do not keep sampling or rewriting reviews until a candidate passes.

```bash
python3 performance/optimization/radar_perf.py compare-timing \
  --blocks /absolute/confirmation/blocks.json --config /absolute/session-config.json \
  --plan /absolute/plan.json --context /absolute/context.json \
  --control-review /absolute/control-review.json --out /absolute/speed.json
```

No `--calibration` is supplied. Speed acceptance replays the raw blocks, screening
separation, control-review references, plan/context, complete metric/precision
coverage and statistical decision. Each consumer needs corresponding evidence;
use the existing aggregate gate workflow. A protected demonstrated slowdown
rejects; a finite interval spanning zero is permitted and reported. Missing or
unbounded intervals remain unresolved. Every declared target must demonstrate
more than its confirmed gain threshold. A/A and B/B observations are not extra
independent A/B samples.

Record the five usual candidate gates. For final original-versus-retained
validation, use direct integrated confirmation; an existing fresh first-candidate
confirmation is sufficient if it already binds original and final retained
identities. Pass its `--control-review`
to `finish --cumulative-timing`; version 4 does not require original A/A
calibration. Reports label the positive-is-slower orientation and retain control
diagnostics, absolute values, effect intervals and coverage gaps. No cumulative
gain is inferred from multiplying accepted intermediate percentages.
