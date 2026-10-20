# Attribution findings: first ranked batch

2026-10-19 · `deustice/optimization-skills` · NVIDIA Thor / existing HSDK 5 deployment.

**All 24 approved jobs completed: 12 clean runs and 12 Nsight captures.** The strongest measured opportunity is MultiSourceExpansion's serial expansion-plan kernel. No production edit, candidate acceptance or retained gain is claimed.

## Opportunity ranking after measurement

| Priority | Finding | Evidence and next decision |
| --- | --- | --- |
| 1 | MultiSourceExpansion serial planning | For 16,384 fully selected rows, clean frame medians are 15.007 ms float / 21.535 ms double. The single-thread planning kernel accounts for about 99% of measured operator kernel work. Prepare a bounded parallel-planning candidate with unchanged ordering/errors. |
| 2 | DetectionReduction allocation/free | Baseline large performs six malloc/free pairs per measured compute; medoid medium performs eight. Free API time is substantial but can include waits for required GPU work. Explore internal scratch reuse after resolving acceptance prerequisites; do not treat all free time as removable. |
| 3 | ClusterDBSCAN allocation/free | Nine malloc/free pairs recur per measured compute. Combined allocation/free call time is roughly 32–33 µs; clustering kernels dominate medium workloads. Small workloads may benefit, but this is a smaller opportunity than source inspection alone suggested. |

The ranking now uses observed recurring work, affected workload size and mechanism confidence. It does not rank the slowest operator first automatically. The other 23 operators remain in the [campaign queue](breadth-first-review-2026-10-16.md); this batch does not certify them or even complete the three selected operators' matrices.

## Clean diagnostic latency

Each row is one isolated process, one repeat, five warmup and 50 measured frames, with the existing two-frame process warmup. These are snapshots, not qualified baselines, confidence intervals or repeated-run regression decisions. Positive/negative changes versus historical runs are not calculated.

| Operator / case | Precision | Median µs | p90 µs | p99 µs |
| --- | --- | --- | --- | --- |
| cluster_dbscan / baseline-small | float | 657.415 | 662.131 | 705.705 |
| cluster_dbscan / baseline-small | double | 1095.530 | 1103.300 | 1149.000 |
| cluster_dbscan / baseline-medium | float | 2747.010 | 2763.630 | 2774.960 |
| cluster_dbscan / baseline-medium | double | 5685.700 | 5705.580 | 5734.230 |
| detection_reduction / baseline-large | float | 293.749 | 301.693 | 326.949 |
| detection_reduction / baseline-large | double | 293.194 | 300.216 | 343.647 |
| detection_reduction / medoid-medium | float | 250.559 | 262.879 | 284.331 |
| detection_reduction / medoid-medium | double | 254.313 | 288.451 | 311.310 |
| multi_source_expansion / baseline-small | float | 116.842 | 124.595 | 125.213 |
| multi_source_expansion / baseline-small | double | 126.523 | 127.957 | 131.927 |
| multi_source_expansion / full-selected-large | float | 15006.800 | 15012.200 | 15021.400 |
| multi_source_expansion / full-selected-large | double | 21535.200 | 21572.100 | 21578.000 |

## Scoped profile evidence

Every capture contained exactly 55 operator compute ranges inside the explicit case/repeat NVTX session: five declared warmups and 50 measured ranges. The table uses only those 50 measured ranges. Full traces and all-range summaries remain preserved. API durations are host-side times and may include waits; GPU durations come from kernels correlated with launches in each measured compute range, including kernels that finish after the CPU range ends.

| Case / precision | Profiled compute median µs | malloc / free calls per frame | malloc median µs | free median µs | GPU kernel work median µs | Planning share, median per frame |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_dbscan / baseline-small / float | 692.762 | 9 / 9 | 16.477 | 16.495 | 471.935 | 0.00% |
| cluster_dbscan / baseline-small / double | 1131.975 | 9 / 9 | 16.890 | 16.389 | 907.565 | 0.00% |
| cluster_dbscan / baseline-medium / float | 2772.405 | 9 / 9 | 16.060 | 15.750 | 2554.602 | 0.00% |
| cluster_dbscan / baseline-medium / double | 5730.259 | 9 / 9 | 17.784 | 16.513 | 5500.147 | 0.00% |
| detection_reduction / baseline-large / float | 303.824 | 6 / 6 | 11.814 | 151.635 | 129.648 | 0.00% |
| detection_reduction / baseline-large / double | 302.986 | 6 / 6 | 11.954 | 151.089 | 130.368 | 0.00% |
| detection_reduction / medoid-medium / float | 267.824 | 8 / 8 | 14.621 | 94.828 | 72.896 | 0.00% |
| detection_reduction / medoid-medium / double | 266.425 | 8 / 8 | 14.820 | 94.838 | 74.576 | 0.00% |
| multi_source_expansion / baseline-small / float | 87.241 | 0 / 0 | 0.000 | 0.000 | 12.384 | 50.44% |
| multi_source_expansion / baseline-small / double | 89.990 | 0 / 0 | 0.000 | 0.000 | 14.144 | 56.56% |
| multi_source_expansion / full-selected-large / float | 14862.585 | 0 / 0 | 0.000 | 0.000 | 14906.477 | 99.10% |
| multi_source_expansion / full-selected-large / double | 21378.283 | 0 / 0 | 0.000 | 0.000 | 21435.757 | 99.32% |

Planning share is a paired within-frame ratio of planning-kernel duration to the sum of correlated kernel durations. It is not a fraction of clean end-to-end latency or a predicted optimization gain. API, compute and kernel durations overlap; category medians are not additive. Profiling overhead and asynchronous output work prevent subtracting these columns to predict final latency.

### MultiSourceExpansion

The fully selected float planning kernel takes about 14.771 ms per measured frame, with about 14.906 ms total correlated kernel work; double takes about 21.288 ms out of 21.436 ms. Source inspection confirms a `<<<1,1>>>` launch and serial loops over input and selected rows. This is direct evidence of avoidable serialization to investigate, not proof that a particular replacement is equivalent or faster.

In contrast, the small case has only about 6.2/8.0 µs of planning kernel time. A rewrite can improve the large case while hurting small cases through extra launches or scratch work; both precisions and the complete case matrix must be protected. There are **zero malloc/free calls in the 50 measured compute ranges** for these expansion cases: the workspace growth seen in warmup must not be misclassified as recurring allocation.

### DetectionReduction

The profile confirms recurring temporary allocation in the implementation `.cuh`, which a translation-unit-only scan misses. Baseline-large allocation and free medians are about 12 µs and 151 µs; medoid-medium about 15 µs and 95 µs. Free time can absorb pending device work, so 151 µs is not a demonstrated removable cost. Preserve kernel reduction order and the existing input-derived semantic policy; no deterministic-accumulation rewrite is proposed.

### ClusterDBSCAN

The nine observed allocation/free pairs match scratch creation and compaction paths. Medium kernels dominate the tick; removing scratch allocation alone is unlikely to transform those workloads. Small-case allocator overhead is a more plausible target. Convergence/count synchronization serves an algorithm dependency; it was not removed or reclassified as an accidental stall.

## Tool-chain assessment

- Fresh diagnostic adapters were compiled against unchanged archived production static libraries; all six executables are byte-identical to their preserved originals. Archived runtime dependencies, resolved libraries and matching recipe identities are recorded separately. Old source manifests were not patched to bypass unrelated DopplerNotch harness changes.
- All 24 clean/profile reports passed raw-frame/precision/case validation. All 12 captures exported CUDA API, kernel and memory summaries and SQLite data; explicit case/repeat attribution succeeded. Raw data, process receipts and artifact hashes are retained.
- Each API cost is paired with recurrence across the exact measured window. All-range summaries alone could misleadingly highlight warmup-only allocation in MultiSourceExpansion. Zero-filled per-frame accounting and recorded boundaries avoid that attribution error.
- The tools now support concrete opportunity selection on this deployment. Candidate equivalence, acceptance/rejection, rollback and final original/current-best comparison have not yet been demonstrated by a production optimization in this campaign.

## Proposed next operator session: MultiSourceExpansion

The machine-readable, schema-validated handoff is `multi-source-session-proposal.json` in the ignored session directory. It is **not confirmed or admitted** and creates no production permission.

| Configuration | Proposal |
| --- | --- |
| Mechanism / scope | Parallel expansion planning and directly required private scratch/launch plumbing; leave ports, formulas, output scatter semantics and shared helpers unchanged. |
| Target | `full-selected-large`, median latency, both precisions; more than 2% improvement by the existing familywise gate. |
| Protected cases | All eight static cases: baseline-small/medium/large, sparse-medium, dense-medium, four-sources-medium, wide-indices-medium and full-selected-large. |
| Output policies | Exact `detection_idx_out` and `aoa_estimates_out` in float/double; exact shapes, row order, timestamp, ownership, errors and state behavior. |
| Sampling proposal | Five warmups, 50 measured frames, three isolated repeats, 30 alternating A/B pairs; preserve first50, full, setup/cold and applicable metrics. Final budget requires full-matrix/consumer review. |
| Limits | At most 20 candidates, stop after three consecutive unsuccessful candidates or exhausted actionable queue. Blocked attempts count toward the attempt cap without increasing the unsuccessful streak. |
| Editable production files | `multi_source_expansion_op.cu` and `multi_source_expansion_stage.hpp`; relevant stage/integration tests may be extended. No recipe, shared-helper, API or equivalence-policy edits. |
| Consumers / fixtures | Adaptive-AoA graph, selective-AoA integration fixture and expansion stage/native fixtures. Complete transitive correctness/timing review before admission. |
| Admission state | Pending exact full-matrix coverage, unchanged-gate A/A calibration, consumer closure and explicit configuration/budget confirmation. Existing calibration blockers remain; profiling does not waive them. |

Integer row counts, stable output order, finite-angle validation, malformed-input error behavior, output limits and empty cases must survive any parallel rewrite. Do not widen acceptance rules to accommodate it. The preliminary primary-matrix family has 8 cases × 2 precisions × 8 applicable metrics × 26 operator slots = 3,328 comparisons; affected consumer additions must be included before confirming a final pair budget.

## Duration, context and limits

Fresh build phase: 136.3 seconds. Serial measurement phase: 43.2 seconds. Trace export: 3.7 seconds. Overall elapsed time through publication/checks is recorded in the final receipt and remains bounded by the approved 45 minutes. Process/job time includes tool launch and does not measure agent working time or GPU busy time.

The 25–40 minute forecast was conservative; compilation, diagnosis and publication dominate this small batch, not the 24 execution jobs. No extra workloads are added to consume unused budget. Capture/report storage is about 10.6 MiB, excluding the separate copied runtime/build archive.

GPU tracing succeeded with Nsight Systems 2026.4.1 on the recorded immutable HSDK 5 image. Clock/governor configuration remained unchanged; no affinity, power, clock or profiler privilege settings were changed. Installed SDK source revision remains unproved. No Nsight Compute or CUDA sanitizer capability is claimed. Pre-existing containers were recorded and not stopped.

All owned process receipts completed with no unknown cleanup state. Measurement sources were unchanged until the session was sealed; this report is published afterward with fresh provenance. No production changes, commits or pushes occurred. No new suspected correctness bugs were found; prior RootMusic review gaps and unrelated missing coverage remain outside this attribution result.

## Next decision

Review the MultiSourceExpansion session proposal and resolve its concrete correctness/calibration/consumer admission requirements before authorizing edits. This is a real operator-specific optimization opportunity, not another generic frame-count research round. Continue independent assessment of the remaining operator queue while blocked scopes stay visible.

Raw session: `build-optimization/operator-perf/assessment-2026-10-19/attribution-plan/`. Key artifacts: `authorization.json`, `ready.json`, `build-receipt.json`, `execution.json`, `exports.json`, `attribution-v2.json`, `measurement-completion.json`, `multi-source-session-proposal.json` and final publication receipt.
