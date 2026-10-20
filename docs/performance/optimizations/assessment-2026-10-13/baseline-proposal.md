# Baseline scope and budget proposal

Status: review draft, assessment only; no new GPU measurements authorized or started. Branch: `deustice/optimization-skills`. The [structured proposal](baseline-proposal.json) records all 26 operators, exact existing case IDs, input axes, fixtures, output wiring and prospective additions. It is a review document, not an executable campaign configuration or frozen protocol.

## Execution order

1. Refresh deployment/provenance, complete missing builds, validate recipes and executable inventories, and audit native fixture coverage. Reuse preserved artifacts only through source/runtime/measurement-context receipts. Reports changing the checkout do not by themselves invalidate production builds.
2. Screen three existing static cases for every operator, both precisions, round-robin by profile rather than completing one operator first. Add the 15 explicit cube scenarios only after constructor bounds, waveform consistency, memory estimates and native verification support are checked. Run two A/A pairs per case; report uncertainty without treating this as qualification.
3. Spend the remaining qualification budget on ten-pair calibration of promising workload scopes, rotating across FFT, reduction, eigensolver, irregular and marshaling families. Existing tiny cases stay visible as diagnostics even when larger cases qualify. Freeze selected sampling and familywise budgets before disjoint held-out validation (at least 30 pairs, recomputed for the actual matrix).
4. Publish partial static baseline/disposition reports. Only cases with correctness, coverage and timing gates complete are admitted; incomplete or unstable cases remain blocked. Stop at the confirmed caps, retain complete hash-verified units, and forecast the next round.
5. A later separately budgeted round adds the remaining 682 existing variant, live, cached/unseen and optional-output cases. Every new case requires casewise evidence. Production optimization still requires the consolidated baseline-milestone review.

The proposed initial matrix is **93 cases / 186 precision-case combinations**: 78 existing static cases plus 15 added cube scenarios. The complete proposed matrix is **775 cases / 1,550 precision-case combinations**. No blanket scaling of every operator is proposed. Added scenarios are not yet recipes or qualified cases.

## Workload definitions

Retain existing recipe sizes as precisely labeled synthetic workloads. There is no fixed deployment or frame deadline. Interpret each operator using its actual work dimensions; a shared small/medium/large name does not imply comparable cost. Case-specific fixtures and optional wiring remain as declared in the structured proposal.

| Operator | Axes in profile order | Existing small / medium / large | Proposed extra scenarios |
| --- | --- | --- | --- |
| beamscan_estimator | channels, items, looks, az, el | 4×64×1×21×11 / 8×256×3×41×21 / 16×1024×9×61×31 | None |
| esprit_estimator | channels, detections, snapshots | 4×64×16 / 8×128×32 / 16×256×48 | None |
| music_estimator | channels, detections, snapshots, az, el | 4×64×16×21×11 / 8×128×32×41×21 / 16×256×48×61×31 | None |
| mvdr_estimator | channels, detections, snapshots, az, el | 4×64×16×21×11 / 8×128×32×41×21 / 16×256×48×61×31 | None |
| phase_shift_beamformer | channels, az, el, doppler, range | 4×17×1×16×32 / 8×31×1×32×64 / 16×41×11×32×128 | None |
| rank_one_residual_screen | items, channels, looks | 32×8×3 / 1024×32×9 / 4096×64×9 | None |
| root_music_estimator | channels, detections, snapshots | 4×64×16 / 8×128×32 / 16×256×48 | None |
| snapshot_gather | channels, doppler, range, detections | 4×64×256×64 / 16×256×512×1024 / 32×512×1024×4096 | None |
| subarray_selection | items, channels, snapshots | 64×4×2 / 256×8×4 / 1024×16×8 | None |
| cluster_dbscan | points, features, clusters | 256×2×8 / 1024×3×32 / 2048×5×64 | None |
| cfar_detector | doppler, range | 64×256 / 128×512 / 256×512 | None |
| detection_reduction | points, clusters, features | 256×32×2 / 1024×64×2 / 4096×128×2 | None |
| feature_assembly | doppler, range | 64×256 / 256×512 / 1024×2048 | None |
| multi_source_expansion | rows | 32 / 2048 / 16384 | None |
| clutter_filter | chirps, range_bins | 16×128 / 64×512 / 128×1024 | Compact, larger, stress |
| ddm_dense_demultiplex | tx, rx, doppler, range | 2×4×32×128 / 4×16×128×512 / 8×32×256×1024 | None |
| ddm_power_fold | tx, rx, doppler, range | 2×4×64×256 / 4×8×256×512 / 4×16×512×1024 | None |
| ddm_sparse_demultiplex | tx, rx, doppler, range, detections | 2×4×64×256×16 / 4×8×256×512×256 / 4×16×512×1024×1024 | None |
| doppler_notch | doppler, range | 32×128 / 128×512 / 256×2048 | Compact, larger, stress |
| fmcw_doppler_processor | chirps, range_bins | 16×128 / 64×512 / 128×1024 | Compact, larger, stress |
| fmcw_range_doppler_processor | chirps, samples | 16×128 / 64×512 / 128×1024 | Compact, larger, stress |
| fmcw_range_processor | chirps, samples | 16×128 / 64×512 / 128×1024 | Compact, larger, stress |
| mimo_deinterleave | tx, rx, chirps_per_tx, samples | 2×4×8×64 / 4×16×32×256 / 8×32×64×512 | None |
| tdm_dense_compensation | tx, rx, doppler, range | 2×4×32×128 / 4×16×128×512 / 8×32×256×1024 | None |
| tdm_sparse_compensation | tx, rx, doppler, range, detections | 2×4×64×256×16 / 4×8×256×512×256 / 4×16×512×1024×1024 | None |
| virtual_array_assembly | tx, rx, doppler, range | 2×4×32×128 / 4×16×128×512 / 8×32×256×1024 | None |

For ClutterFilter, RangeProcessor, DopplerProcessor, fused RangeDopplerProcessor and DopplerNotch, add explicit `(channels, chirps or Doppler bins, samples or range bins)` scenarios: compact `(4,128,512)`, larger `(16,256,1024)`, stress `(64,512,2048)`. DopplerNotch uses its `rx` fixture, with `tx=1`; the other four use `channels`. These are individual operator inputs, not a claim of one end-to-end radar configuration. Range/fused waveform duration must match ADC samples at 12.8 MSPS, with PRI greater than duration. Do not automatically scale FFT padding or physical live values from legacy cases.

Sparse and angle operators already vary channels, snapshots/looks, detected-item counts and scan grids. Preserve those axes initially. DBSCAN retains 256/1,024/2,048 points with 2/3/5 features and 8/32/64 generated clusters; density and neighborhood variants remain in round two. FeatureAssembly explicitly uses 64/1,024/16,384 detections in its baseline cases, not its default fixture alone. CFAR rank-four suppression and FeatureAssembly score/AoA packing remain visible in the full existing matrix and require verification.

## Proposed measurement and consistency policy

Use version 3, **50 measured frames × 3 isolated repeats**, five warmup frames, existing two-frame process warmup, chronological raw frames and median aggregation. First-50 and full-window median/p90/p99, setup, cold and applicable live-transition metrics remain protected. At 50 frames the early/full values duplicate observations; preserve declared rows in the existing family calculation. Include 26 operators and reviewed consumers when recomputing family size; do not copy the pilot’s 416 as the full campaign family size.

| Metric class | P90 paired A/A difference limit | Baseline IQR limit |
| --- | --- | --- |
| Steady and early windows | 5% | 10% |
| Setup | 15% | 25% |
| Cold | 20% | 30% |
| Applicable live transition | 25% | 40% |

**Recommendation: adopt no numeric absolute allowance yet.** Report empirical absolute bands per case, precision and metric alongside these limits. Existing sub-microsecond RangeProcessor median variability and the two-pair copy control do not establish a universal scheduling floor or reliable p99 allowance. A guessed 1 µs median or 5 µs tail floor could conceal material changes in small operators. If this round supports an allowance, present concrete values prospectively and collect fresh disjoint validation before use; never retroactively pass failed evidence.

Use per-run ECDFs, matched-run distribution distances, quantile-shift intervals, drift and stalls as diagnostics. Frames within a run are not independent replicates. Confidence width and ±1% precision remain diagnostics. Keep dispersion, bias, insufficient interval resolution, correctness and timing regimes separate. Qualification noise allowances never automatically become acceptable candidate regressions. Fewer than 50 frames would change the protected early-window semantics and are outside this proposal.

## Runtime and storage forecast

Completed pilot phases cost approximately **11–78 seconds per A/A pair**, including both precisions, three repeats and phase checks. Raw evidence: `build-optimization/operator-perf/assessment-2026-10-13/distribution-pilot/budget.json`. Use 15/30/75 seconds per pair as sensitivity examples; unmeasured eigensolver, dense-angle or irregular cases can exceed them. These are not guaranteed bounds.

| Timing scope | Pairs per case | At 15 s/pair | At 30 s/pair | At 75 s/pair |
| --- | --- | --- | --- | --- |
| round1: screen (93 cases) | 2 | 0.8 h | 1.6 h | 3.9 h |
| round1: calibration (93 cases) | 10 | 3.9 h | 7.8 h | 19.4 h |
| round1: heldout_floor (93 cases) | 30 | 11.6 h | 23.2 h | 58.1 h |
| round1: all_timing_phases (93 cases) | 42 | 16.3 h | 32.5 h | 81.4 h |
| expanded_full_matrix: all_timing_phases (775 cases) | 42 | 135.6 h | 271.2 h | 678.1 h |

The 42-pair rows assume two screening, ten calibration and 30 held-out pairs for every case. Blocked cases stop before held-out validation; escalation, larger familywise requirements and retries increase cost. A pair launches six wrapper jobs (two sides × three repeats), each running two separate precision executables. Do not multiply the already combined empirical pair costs by precision again. Full-matrix timing alone could occupy days to weeks; a scripted runner removes manual work, not measurement cost.

**Recommended next authorization:** an **8-hour qualification cap**, plus a separate **4-hour preparation/correctness cap**. Finish the broad screen first if affordable, then rotate calibration/held-out work across families until the cap. This is a bounded partial milestone, not a promise that all 93 cases qualify. Count charged phase durations and enforce affordability before launching units. Expensive cases that threaten breadth receive a deferred-cost disposition rather than monopolizing the queue. No automatic extension.

Native verification storage is the largest uncertainty. Stress complex outputs alone total about 1.5 GiB per float-plus-double frame; two captures × three frames require at least **9 GiB** before JSON/text expansion or extra ports. Check free space, predict output sizes and serialization costs before starting; stream/compare only if the maintained verification path preserves all required values and witnesses. Otherwise block that coverage and report why. Do not substitute tiny-case correctness for stress-case verification.

For all 775 cases at 42 pairs, chronological timing samples alone total 19,530,000 numbers, about **447 MiB** at an assumed 24 serialized bytes each. Repeated child/merged reports, inventories, logs and receipts add overhead; native captures and immutable executables/runtime libraries are additional. Keep a storage ledger and revise after the broad screen rather than claiming a fixed disk bound. The ignored machine-readable sensitivity table is `build-optimization/operator-perf/assessment-2026-10-13/baseline-proposal/forecast.json`.

## Admission, reuse and unresolved findings

Preserve exact outputs and behavior for both precisions except the already authorized DetectionReduction semantic policy with fresh native input witnesses and coupled outputs. Assessments have empty production editable scopes and no candidate targets. Future shared-source changes require consumer review and invalidation, including ClutterFilter’s shared fused RangeDoppler stage.

The prior larger/stress RangeProcessor and stress ClutterFilter timing evidence remains historical qualified evidence. Reuse in a new campaign requires verified identity, unchanged context and reviewed receipts; new recipes, source or policies require fresh evidence. No timing qualification substitutes for missing new-scenario native captures.

No new suspected correctness bugs were found during this planning step. Existing RootMusic algebraic-rank interpretation and borrowed-output retention remain explicit review gaps. Missing sanitizer, Python bindings, Nsight Compute and installed SDK revision proof remain limits. The six harnessless SAR/I/O operators remain excluded. No optimization, deployment-setting changes, worktrees, commits or pushes are included.

## Review boundary

The JSON budget fields remain null until the user confirms positive time caps. After approval, create fresh maintained-tool campaign/operator configs, validate each effective sampling/recipe/output policy and provenance closure, seal protocol identities and launch bounded work. This proposal does not silently revise the completed pilot or authorize another full baseline sweep.
