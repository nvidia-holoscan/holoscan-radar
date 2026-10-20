# Ranked attribution plan

2026-10-19 · `deustice/optimization-skills` · assessment only.
**Prepared; no GPU workloads launched and no budget authorized.**

## Findings and priorities

All 26 harness-backed operators remain in the campaign. The current source
inventory now includes implementation `.cuh` headers as well as `.cu` and
`.hpp` files: 187 syntactic allocation/readback/synchronization sites across
operator-owned files. This is a search inventory, not 187 defects or a
completed transitive implementation audit. In particular, DetectionReduction's
13-line translation unit delegates its implementation to a `.cuh` header;
a translation-unit-only scan misses its recurring allocations.

| Rank | Operator / evidence | Decision to resolve | Constraints |
| --- | --- | --- | --- |
| 1 | ClusterDBSCAN: four scratch tensors created by `prepare_dbscan_workspace()` on each non-empty tick; additional compaction scratch and a host convergence loop | Does recurring allocation/free contribute enough end-to-end cost to justify persistent internal scratch? | Preserve kernel formulas, initialization, shape changes, deterministic labels, errors, optional sizes and output lifetime. |
| 2 | DetectionReduction: per-call scalar/selection tensors in `detection_reduction_op_impl.cuh`, including mode-specific medoid scratch | Which temporary tensors recur in baseline and medoid paths, and what fraction of the tick do allocation and count readbacks occupy? | Existing timing remains blocked; attribution is permitted, production acceptance is not. Retain authorized semantic validation and coupled output rules. |
| 3 | MultiSourceExpansion: plan kernel launched as `<<<1,1>>>`, with serial loops over items and selected rows | Does serial planning dominate the larger fully selected case, or do scatter/scan/readback costs dominate? | A parallel rewrite is more complex than scratch reuse. Preserve ordering, validation errors, output count and empty cases. No rewrite proposed yet. |
| 4 | ClutterFilter: MatX mean expression may materialize a temporary | Attribute allocation frequency and traffic if the first batch leaves this actionable | No established allocation trace or removable fraction; not part of this batch. |

No percentage gain or removable latency fraction is established for these leads.
The order reflects mechanism evidence and validation readiness, not absolute
runtime. ClusterDBSCAN small/medium have historical qualified timing scopes;
DetectionReduction and MultiSourceExpansion still need appropriate calibration
before any performance candidate could be accepted.

False leads checked: Beamscan's main scratch is constructor-allocated; MVDR
retains its covariance/inverse storage, and its device synchronization helper
is teardown. RankOneResidualScreen uploads positions only when geometry changes.
SnapshotGather grows its batch workspace by capacity. Sparse gather/assembly
operators use host readback to report validation errors; merely deleting those
synchronizations would change observable behavior. Cached DDM/TDM tables are
not presumed to rebuild each frame. Indirect MatX/CUB allocations still need
runtime attribution; absence of explicit allocation calls proves little.

## Concrete batch

Every row runs float and double separately, with one clean diagnostic process
and one profiled process per precision. Six cases therefore yield **12 clean
runs + 12 Nsight captures**, 24 execution jobs total.

| Operator | Cases | Why these cases |
| --- | --- | --- |
| ClusterDBSCAN | `baseline-small`, `baseline-medium` | Existing qualified scopes; compare 256 versus 1,024 points and relative scratch/convergence cost. |
| DetectionReduction | `baseline-large`, `medoid-medium` | Compare baseline peak selection and mode-specific medoid scratch without changing formulas. |
| MultiSourceExpansion | `baseline-small`, `full-selected-large` | Compare 32 rows with partial selection versus 16,384 fully selected rows; attribute serial planning, not just total latency. |

Use the exact canonical recipes whose hashes match the archived build manifests.
Both precision executables were hash-verified while preparing the commands.
The existing driver resolves these recipes from the checkout by walking upward
from the executable; archived executable location alone does not freeze YAML.
Recheck the recipe hashes before execution. Preserve existing timed/drained
outputs and fixtures. No new recipe, live mode or production edit is included.

Sampling: five warmup frames, existing two-frame process warmup, **50 measured
frames and one repeat**. This is a diagnostic snapshot, not A/A calibration,
reproducibility evidence or an optimization baseline. Clean and profiled runs
have distinct output paths. Profiler latency never supplies acceptance timing.
Keep full traces, setup/cold behavior and early observations; use the explicit
case/repeat NVTX range to separate process warmup. Verify frame boundaries
before interpreting any warmup-filtered phase summaries; do not guess ranges
from launch order or discard early measured frames.

Nsight Systems 2026.4.1 is present in the previously recorded immutable HSDK 5
image. CLI help confirms CUDA/NVTX tracing, memory tracking, disabled CPU
sampling/context switching and existing CUDA summary reports. Report-list help
returns code 1 while printing its available reports; the text was saved as
capability evidence. **Successful GPU tracing is still unknown.** No counter,
clock, affinity, power, privileged profiling or deployment changes are included.

## Measurement and artifact plan

1. Produce a fresh source/runtime reuse binding or rebuild if verification
   requires it. Check dynamic-library resolution, image identity, recipe case
   lists, precision labels and device context. Original builds stay immutable.
   Broad old provenance includes unrelated changes; do not bypass a mismatch.
2. Serialize jobs through the existing device lock. Record owned process groups
   and named containers, clean up only owned processes, and reject existing
   output paths. Preserve completed and interrupted receipts separately.
3. For each case/precision, collect the clean diagnostic JSON and a fresh
   `nsys profile` capture with CUDA/NVTX and memory usage tracing. CPU sampling
   and context-switch collection are disabled. Exact argv arrays, original
   manifests, executable/recipe hashes and future output paths are in `jobs.json`.
4. Export scoped CUDA API, kernel and memory summaries plus SQLite/NVTX evidence.
   Use supported report names `cuda_api_sum`, `cuda_gpu_kern_sum`,
   `cuda_gpu_mem_time_sum` and `cuda_gpu_mem_size_sum`. Keep summary creation
   times/hashes and validate with the existing profile-artifact tools.
5. Inspect same-thread, case/repeat-scoped compute ranges. Separate recurring
   steady scratch calls from setup/cache misses and output-pool allocation.
   Report CPU API union, kernel/memory behavior, waits and launch gaps without
   adding overlapping durations or treating CPU API time as GPU execution time.
6. Publish absolute clean latency, source/case context, attributable recurring
   work, uncertainty, coverage gaps and the next decision. Missing NVTX schemas
   or unavailable CUDA tracing remain explicit blockers; do not invent phase
   assignments or elevate privileges to force a capture.

## Consumers and candidate boundaries

ClusterDBSCAN and DetectionReduction are directly instantiated in the SISO,
SIMO, MIMO-TDM, MIMO-DDM and multi-waveform examples. DetectionReduction is also
a downstream consumer of cluster labels. MultiSourceExpansion is instantiated
in adaptive-AoA. `jobs.json` binds these direct application source references.
This is a direct-consumer map, not proof that all transitive consumers are
covered. Shared-helper edits would expand the review and verification closure.

Relevant fixtures include `test_cluster_dbscan.cpp`,
`test_detection_reduction.cpp`, `test_detection_reduction_custom.cu` and
`test_multi_source_expansion.cu`; application pipeline fixtures must be included
where a later candidate can affect them. Before edits, verify full float/double,
live, optional-output, ownership, error and shape-change coverage against the
confirmed candidate scope. Historical static qualification does not certify
these other modes.

The first possible edit scope is ClusterDBSCAN **internal scratch reuse only**,
with kernels, output allocation, synchronization, labels/sizes ordering and
numerical formulas preserved. This is a hypothesis, not editing permission.
If allocation is insignificant, deprioritize it and move to the next lead.
Do not bundle convergence or algorithm changes. Any retained candidate still
requires confirmed targets/outputs/consumers and exact correctness plus all
existing timing gates, including the greater-than-2% objective.

## Time, stop conditions and next decision

Recommend a **45-minute total attribution budget**, not an hours-long baseline
rerun. Expected elapsed time is **25–40 minutes**, with substantial uncertainty
because profiler/export overhead has not yet been measured on this deployment.
Historical twelve-executable pair jobs suggest roughly 1.3–1.6 seconds per clean
executable for these families; this is only a rough process-cost forecast.
Variant and tracing costs may differ.

| Reserved component | Maximum allocation |
| --- | --- |
| Fresh preflight/binding, or bounded preparation | 8 minutes |
| 24 execution jobs, at most 60 seconds each | 24 minutes |
| Trace export and initial attribution | 8 minutes |
| Publication/cleanup reserve | 5 minutes |
| Total cap | 45 minutes |

Stop rather than extend if provenance repair/rebuild, profiling, exports or
storage cannot fit. Do not start a job unless its capped cost plus the
five-minute closeout reserve fits. Never accept partial jobs as complete.
Trace size is unmeasured: check free disk space, track cumulative bytes, and
stop before exhausting storage. No native tensor exports or production tests
are proposed in this attribution-only batch. Existing correctness evidence
remains contextual, not fresh candidate validation.

The batch asks whether the observed mechanisms consume enough recurring work
to justify bounded candidate sessions. Its deliverable is an attributed
shortlist and concrete editing-session configurations, or evidence to
reject/deprioritize the leads. If a profile is blocked, continue independent
jobs and report the gap; no automatic replacement cases or budget extension.
All other operators remain in the [26-operator inventory](breadth-first-review-2026-10-16.md).
No new suspected correctness bugs were identified. No GPU workload, production
optimization, commit or push occurred while preparing this plan.

Artifacts: `build-optimization/operator-perf/assessment-2026-10-19/attribution-plan/`
contains `jobs.json`, `source-sites.json`, capability text and publication
receipts. **Next action: confirm the 45-minute attribution budget**, then run
only the declared jobs and publish the findings before proposing production edits.
