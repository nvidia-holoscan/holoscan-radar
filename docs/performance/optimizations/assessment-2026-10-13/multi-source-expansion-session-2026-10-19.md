# MultiSourceExpansion — blocked at timing admission

2026-10-19 · `deustice/optimization-skills` · approved four-hour session. **Stopped after 15.6 minutes of preparation/collection because unchanged calibration limits could no longer pass. No production session opened, candidates attempted, operator changes retained or gains claimed.** Report/check closure followed the measurement stop.

## Decision and next action

The serial-planning opportunity remains strong, but small-case float p99 failed the confirmed timing-consistency requirement. This is a measurement-admission failure, not evidence that the operator became slower or that its outputs are incorrect. All executed correctness checks passed.

Do not rerun another bulk baseline under the same conditions merely to seek a pass. The next decision is an explicit review of the timing-admission policy or a separately confirmed, reduced timing scope with excluded cases kept visibly blocked. Existing correctness requirements remain mandatory. No replacement gate, threshold relaxation, new budget or additional production scope was adopted. Overall campaign scope remains 26 harness-backed operators; this is one blocked session, not completion of the campaign.

## Exact blocker evidence

Each A1/A2 value below is the median of three independently launched repeat-run **per-frame p99** values; each repeat contains five warmups, the existing two process-warmup frames and 50 measured frames. These are p99 latency summaries, not medians of the raw frame distribution. A1 and A2 execute identical code and archived libraries. Pair order alternates adjacently by repeat.

| Pair | A1 p99 µs | A2 p99 µs | Absolute paired difference | Exceeds 5% |
| --- | --- | --- | --- | --- |
| 1 | 136.786 | 136.659 | 0.093% | no |
| 2 | 146.284 | 134.097 | 8.331% | yes |
| 3 | 144.544 | 136.952 | 5.252% | yes |
| 4 | 151.161 | 153.230 | 1.368% | no |
| 5 | 131.195 | 143.980 | 9.744% | yes |
| 6 | 131.297 | 147.705 | 12.496% | yes |

The six-pair descriptive P90 absolute difference is **11.120%**. More decisively, fill the remaining 24 planned pairs with zero difference: the maintained interpolated P90 of all 30 values is still **5.560%**, above **5%**. Four observed differences exceed 5%; the last two relevant order statistics interpolate to this lower bound. Future absolute differences cannot lower it. This is a deterministic rejection-only certificate against the original 30-pair rule, not an optional-stopping significance claim or a reduced acceptance budget.

Both `p99_ms` and `early_p99_ms` are blocked. With 50 measured frames, the first-50 and complete windows coincide; they are not two independent failures. No frames were dropped, warmup increased or aggregation changed. Other partial dispersion statistics are descriptive, not additional certified rejection claims.

## Casewise absolute timings

These are descriptive medians across the six completed pairs, in microseconds. A1/A2 always mean original versus identical original. All eight primary cases and both precisions are represented in each complete pair. Twenty-four additional case/repeat containers from the unfinished seventh matrix remain preserved but do not enter this table.

| Case | Precision | Median A1 / A2 µs | p90 A1 / A2 µs | p99 A1 / A2 µs |
| --- | --- | --- | --- | --- |
| baseline-small | float | 124.028 / 123.921 | 127.296 / 126.762 | 140.665 / 140.466 |
| baseline-small | double | 125.960 / 126.298 | 129.528 / 129.041 | 141.988 / 147.748 |
| baseline-medium | float | 467.121 / 466.933 | 471.480 / 471.933 | 487.498 / 489.173 |
| baseline-medium | double | 578.707 / 577.822 | 582.962 / 582.383 | 604.400 / 596.029 |
| baseline-large | float | 2859.063 / 2859.241 | 2864.468 / 2865.047 | 2880.553 / 2883.274 |
| baseline-large | double | 3744.651 / 3745.079 | 3753.065 / 3752.704 | 3769.894 / 3773.943 |
| sparse-medium | float | 281.050 / 281.837 | 285.343 / 285.383 | 303.509 / 304.966 |
| sparse-medium | double | 330.739 / 330.714 | 334.964 / 335.324 | 349.632 / 355.792 |
| dense-medium | float | 775.058 / 774.744 | 779.023 / 779.715 | 793.999 / 800.492 |
| dense-medium | double | 997.252 / 997.316 | 1005.869 / 1001.344 | 1022.361 / 1012.579 |
| four-sources-medium | float | 615.369 / 615.590 | 622.358 / 621.353 | 637.589 / 644.739 |
| four-sources-medium | double | 825.697 / 824.985 | 829.678 / 831.036 | 845.499 / 850.602 |
| wide-indices-medium | float | 467.087 / 467.293 | 472.225 / 471.283 | 488.287 / 484.378 |
| wide-indices-medium | double | 578.225 / 578.482 | 584.224 / 582.234 | 602.760 / 601.479 |
| full-selected-large | float | 15005.369 / 15005.143 | 15021.606 / 15016.536 | 15042.728 / 15034.925 |
| full-selected-large | double | 21534.100 / 21535.996 | 21569.828 / 21572.036 | 21579.541 / 21589.474 |

Familywise intervals are **unbounded** for these partial data: six completed pairs versus a minimum of 19 for finite interval endpoints across 6,656 protected comparisons. Thirty remained the confirmed budget. No candidate improvement, protected-regression verdict or proof of equivalence is inferred from the partial timing data. The [complete 128-row metric appendix](multi-source-expansion-aa-2026-10-19.md) includes setup, cold, first-50 and full-window metrics, absolute timings, percentage changes, thresholds, observed IQR and interval status. Maxima, raw chronological frames and stall/drift diagnostics remain in the original hashed reports.

## Correctness, provenance and coverage

- Passed 28 expansion stage tests across float/double, 30 FeatureAssembly tests, one selective-AoA integration fixture and 37 simulator tests. The integration graph itself is float; both operator precisions have native capture coverage.
- Two independent native captures matched exactly for all eight expansion and three consumer cases in both precisions: 22 case/precision combinations, three frames each. Independent formula checks also passed for those 22 combinations, including row ordering, exact indices, output shapes and feature gathers. Formula-based angle checks accommodate the input recipe literal representation; the approved output-equivalence policy remains exact.
- Four fresh harness executables were byte-identical to their preserved originals. All 26 dynamically resolved dependencies per executable came from the separate runtime archive. The pinned SDK image and production archives were unchanged, and clock configuration and the whole-checkout source fingerprint remained fixed throughout measurements.
- Consumer score/AoA packs vary map size but use 64 detections. They do not validate full expanded-large application throughput. Their repeated native outputs passed; consumer A/A timing was not launched after dependent primary admission failed.
- Retained device-loan ownership and additional native operator boundary/error checks remain pending. Stage error-precedence and empty-input tests passed; snapshot copies alone do not establish retained-output lifetime. These gaps were not waived.

## Changes and outcome

| Item | Disposition |
| --- | --- |
| Production optimization | Not attempted: timing admission blocked before edits. |
| Retained/rejected candidates | None / none; zero of 20 attempts consumed. |
| Cumulative versus original | No retained change or validated gain; A/A observations are not optimization results. |
| Measurement tooling | Consumer-specific case collection/replay was exercised. A closure fix preserves historical normalized hashes when its optional map is absent; 105 host tests pass, including that regression case. |
| Unresolved suspected production bugs | None newly identified. Ownership coverage and timing dispersion remain unresolved limitations. |

## Durations and cleanup

The four-hour allocation was a cap. Measurement stopped 15.55 minutes after authorization. Fresh build/archive preparation took 54.2 seconds; native repeated-capture phases took 6.3 seconds; the A/A phase took 12.89 minutes including hashing, orchestration and report handling. The 312 isolated A/A container jobs account for 6.91 minutes of summed process time. That job time is **not** a measure of agent working time. Overall report/check completion duration is recorded separately in the publication receipt.

All 337 owned process receipts were closed with known cleanup, and the device lock was released. No unrelated containers were removed. Original builds, user edits, historical reports and `.agents/` remain preserved. No commits, pushes, external issues or clock/power changes were made.

Artifacts: `build-optimization/operator-perf/assessment-2026-10-19/multi-source-admission/`. Key files: `authorization.json`, `config.json`, `build-receipt.json`, `correctness-tests.json`, `native-reference-review.json`, `baseline/*/native-review.json`, `baseline/multi_source_expansion/aa/pairs.json`, `irreversible-dispersion-review.json`, `partial-diagnostics.json`, `phase-duration-review.json`, `cleanup-review.json`. The production session was deliberately not initialized because version-3 admission failed.
