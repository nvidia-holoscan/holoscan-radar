# Measurement Qualification

Qualification separates measurement design from candidate results. Use the
maintained tools in `performance/optimization/`; do not recreate a one-off
pilot or treat historical version-2 reports as raw version-3 evidence.

1. Inventory existing cases, precision and native verification coverage. Record
   deployment preflight, image/SDK identity, runtime libraries and source hashes.
2. Obtain an explicit positive qualification time budget before measurement.
   Forecast runtime and storage, with assumptions and unknowns visible. Begin
   exploration with 50/500/2,000 frames and 3/7 repeats; these are configurable
   exploration levels, not universal benchmark defaults. Evaluate affordable
   levels within the budget. Exhaustion leaves unresolved scopes.
3. Collect adjacent alternating A/A runs in isolated case/repeat processes.
   Capture chronological raw frames and `[0, frames]` repeat boundaries. Keep
   five warmup frames and existing graph/process warmup by default. Never
   silently add warmup or discard early observations. Declare any changed
   measurement policy and validate it afresh.
4. Compare aggregation on calibration data. Median is the default; mean is an
   optional exploratory comparison of repeat summaries. Record selection rules
   and freeze the protocol, case overrides, family size and pair budgets before
   held-out validation. Validation uses a separate artifact namespace and fresh
   processes; no calibration observations can enter validation.
5. Inspect full-window and first-50 median/p90/p99, setup, cold and applicable
   live transitions. Full-case calibration still precedes candidate acceptance.
   Add early metrics and campaign/operator/consumer multiplicity to the existing
   familywise calculation. Twenty pairs is a starting proposal, not a power
   guarantee. Confidence width and intervals within +/-1% are diagnostics only.
6. Keep failures separate: correctness, directional bias, insufficient interval
   resolution, dispersion and unresolved timing regimes. Report absolute A1/A2
   milliseconds, P90 paired delta and IQR against their concrete limits,
   familywise intervals, maxima, stalls, half-window drift and early bias for
   every affected case and precision. Descriptive drift does not establish a
   stationarity test or automatically authorize a regime change.

The existing steady P90-delta/IQR proposals remain 5%/10%; setup 15%/25%, cold
20%/30%, override 25%/40%. Freeze confirmed policies before candidate results.
No blocked calibration authorizes relaxed equivalence or acceptance gates.

`qualification.qualify()` seals selection and validation receipts and a
cumulative budget ledger. Completed units resume only after child hashes,
protocol identity, source/build/runtime identity and clock configuration match.
Interrupted unsealed outputs require explicit quarantine; do not count or
rehash them. Per-device advisory locks serialize cooperative tools. External
GPU activity remains a deployment observation, not something to stop silently.
Owned process groups and named containers are cleaned up on interruption.
Unknown clocks, power, affinity or profiling capabilities stay unknown. Never
change clock/power/affinity settings or privileged profiling setup implicitly.
