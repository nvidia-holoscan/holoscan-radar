# MultiSourceExpansion session admission — 2026-10-19

Branch: `deustice/optimization-skills`. Status: preparation complete; GPU admission and production session unconfirmed. No production files changed, no additional GPU workloads executed, and no optimization gain claimed in this preparation.

## Finding and bounded mechanism

The [completed attribution](attribution-results-2026-10-19.md) measured fully selected large-case medians of 15,006.8 µs (float) and 21,535.2 µs (double). Serial expansion planning occupied approximately 99% of measured kernel time. The proposed candidate parallelizes integer/validation planning and preserves output order, formulas, precision and error precedence. A large possible removable fraction is evidence of opportunity, not a measured candidate gain.

Production scope is limited to `multi_source_expansion_op.cu`, `multi_source_expansion_stage.hpp` and directly required scratch/launch plumbing. The stage and selective-AoA integration tests may change. Public interfaces, recipes, shared helpers and comparison policies are outside scope.

## Concrete proposed session

| Item | Proposal |
| --- | --- |
| Primary timing and verification | All eight existing cases: baseline-small, baseline-medium, baseline-large, sparse-medium, dense-medium, four-sources-medium, wide-indices-medium, full-selected-large; float and double. |
| Objective eligibility | More than 2% median improvement on full-selected-large in both precisions, using the existing familywise confidence gate. Declare candidate targets before editing. |
| Output policy | Exact detection_idx_out and aoa_estimates_out, including shapes, row order, emission coordinates and ownership; preserve validation errors and state behavior. |
| Consumer | Adaptive-AoA connects both expansion outputs to FeatureAssembly. Protect points exactly in float and double on score-aoa-pack-small, score-aoa-pack-medium and score-aoa-pack-large. Check application/report wiring and selective-AoA integration behavior. Other FeatureAssembly modes remain outside this consumer timing scope. |
| Protected timings | Complete-window and first-50 median/p90/p99, setup and cold for every selected case/precision; live transitions where applicable. Expansion has no live ports or optional outputs. Keep maxima and stall diagnostics. |
| Sampling | Five warmups, existing two-frame process warmup, 50 measured frames, three isolated repeats, 30 alternating A/A or A/B pairs, adjacent repeat pairing, chronological raw frames, median aggregation. No silent early-frame exclusion. |
| Calibration limits | Steady P90 paired delta 5% and baseline IQR 10%; setup 15%/25%, cold 20%/30%, applicable overrides 25%/40%. Missing or unbounded intervals block admission. Ordinary inconclusive protected comparisons remain allowed. |
| Multiplicity | 26 campaign operator slots and two affected operators. Each comparison counts its actual case/precision/metric rows; verify the pair proposal with a complete preview. Thirty pairs is not a power guarantee. |
| Limits | At most 20 candidates, three consecutive unsuccessful candidates, or exhausted actionable queue; blocked attempts consume attempts without increasing that streak. Proposed four-hour total cap, including admission and reporting. Stop at the cap; no automatic extensions. |
| Workflow | Single checkout, immutable original/current-best/candidate builds and runtime libraries, verified checkpoints/restores, fresh source and runtime bindings, exact/native/analytic checks and paired gates. Retain only fully passing changes. |

The machine-readable proposal is `build-optimization/operator-perf/assessment-2026-10-19/multi-source-admission/session-proposal.json`. It is explicitly unconfirmed. Existing attribution captures are diagnostics, not fresh full-matrix calibration evidence.

## Preparation completed

- Expanded the stage fixture from six single-precision-specific tests to 28 instantiated tests across float and double. Added empty inputs, mixed source capacities, non-prefix large selections, negative counts, unused-angle behavior and cross-row/same-row error precedence. The fixture compiled and linked against the pinned HSDK 5 image; listing verified all 28 tests. **GPU execution is pending.**
- Added optional version-3 consumer-specific sampling. Collection and replay now use the consumer's own declared case IDs. Confirmation hashes bind the mapping; version-2 evidence cannot opt in. Defaults preserve historical behavior and all existing gates.
- All 104 host-tool tests passed, including real synthetic subprocess collection/replay, consumer case routing, stale-protocol rejection, confirmation hashing and legacy compatibility. Synthetic host reports are tool tests, not GPU performance evidence.

## Admission work still required

Run the new stage fixture, relevant analytic/reference and integration tests, repeated native exact captures for both operators, and retained-output lifetime checks. The existing integration graph is float; isolated operator verification covers both precisions. Snapshot copies alone do not prove device-output lifetime. Candidate tests must exercise the actual new parallel launcher rather than relying solely on the existing serial stage launch.

Then collect complete, fresh A/A matrices for the eight primary and three consumer cases. Existing calibration blockers remain unresolved; this preparation does not relax or reclassify them. If any scope fails correctness or calibration, preserve evidence and block its candidate acceptance rather than extrapolating from the favorable large-case profile.

One complete matrix entails 3,960 isolated case/repeat invocations: 11 cases × two precisions × 30 pairs × two sides × three repeats. Forecast roughly 50–75 minutes per matrix, based on recent diagnostic process costs, with substantial startup and large-case uncertainty. Admission A/A, one candidate A/B and a retained-best refresh can therefore approach three to four hours including builds/checks. The four-hour proposal is a cap, not a required wait or a guarantee of completion. Stop early on an admission blocker; do not spend the remainder on generic protocol research. Storage depends on native capture sizes; record a concrete forecast after the first complete capture, before bulk collection.

## Next action

Confirm the concrete session and a positive time budget. Within that authorization, complete correctness and full-matrix calibration first; make production edits only if admission succeeds. Rejected candidates restore through verified checkpoints. Publish the operator disposition and reassess the remaining 25-operator queue. Overall campaign scope remains all 26 harness-backed operators.

No new suspected production correctness bug was identified. Unresolved ownership/analytic coverage and calibration are explicit admission gaps. Compilation and host tests do not resolve them.

Artifacts and receipts: `build-optimization/operator-perf/assessment-2026-10-19/multi-source-admission/`. Historical reports and immutable original builds remain preserved.
