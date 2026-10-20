# Operator benchmark campaign

Phase: qualification_complete. Mode: assess.

Assessment timings do not establish adequate performance for omitted or blocked cases.

Baseline/final columns show ranges of casewise compute medians in milliseconds, not pooled run statistics.

| Operator | Family | Disposition | Baseline medians ms | Final medians ms | Retained changes | Coverage gaps |
| --- | --- | --- | --- | --- | --- | --- |
| beamscan_estimator | irregular | discovered | not collected | not collected | 0 | None recorded |
| esprit_estimator | eigensolver | discovered | not collected | not collected | 0 | None recorded |
| music_estimator | eigensolver | blocked | not collected | not collected | 0 | report commit does not match build manifest |
| mvdr_estimator | eigensolver | discovered | not collected | not collected | 0 | None recorded |
| phase_shift_beamformer | irregular | discovered | not collected | not collected | 0 | None recorded |
| rank_one_residual_screen | irregular | discovered | not collected | not collected | 0 | None recorded |
| root_music_estimator | eigensolver | discovered | not collected | not collected | 0 | None recorded |
| snapshot_gather | irregular | discovered | not collected | not collected | 0 | None recorded |
| subarray_selection | irregular | discovered | not collected | not collected | 0 | None recorded |
| cluster_dbscan | irregular | blocked | not collected | not collected | 0 | report commit does not match build manifest |
| cfar_detector | irregular | discovered | not collected | not collected | 0 | None recorded |
| detection_reduction | irregular | discovered | not collected | not collected | 0 | None recorded |
| feature_assembly | irregular | discovered | not collected | not collected | 0 | None recorded |
| multi_source_expansion | irregular | discovered | not collected | not collected | 0 | None recorded |
| clutter_filter | reduction | blocked | not collected | not collected | 0 | report commit does not match build manifest |
| ddm_dense_demultiplex | fft | discovered | not collected | not collected | 0 | None recorded |
| ddm_power_fold | reduction | discovered | not collected | not collected | 0 | None recorded |
| ddm_sparse_demultiplex | irregular | discovered | not collected | not collected | 0 | None recorded |
| doppler_notch | irregular | discovered | not collected | not collected | 0 | None recorded |
| fmcw_doppler_processor | fft | discovered | not collected | not collected | 0 | None recorded |
| fmcw_range_doppler_processor | fft | discovered | not collected | not collected | 0 | None recorded |
| fmcw_range_processor | fft | blocked | not collected | not collected | 0 | report commit does not match build manifest |
| mimo_deinterleave | irregular | discovered | not collected | not collected | 0 | None recorded |
| tdm_dense_compensation | irregular | discovered | not collected | not collected | 0 | None recorded |
| tdm_sparse_compensation | irregular | discovered | not collected | not collected | 0 | None recorded |
| virtual_array_assembly | irregular | discovered | not collected | not collected | 0 | None recorded |

## Phase durations

Job durations include process launch and orchestration, and do not measure agent working time.

- clutter_filter / qualification: 8.7 seconds (blocked).
- fmcw_range_processor / qualification: 8.7 seconds (blocked).
- music_estimator / qualification: 9.3 seconds (blocked).
- cluster_dbscan / qualification: 8.5 seconds (blocked).

## Optimization handoff

Agent applies operator-perf candidate declarations and gates. No production edits are performed by this tool.

Opportunity ranking requires attributed recurring work, removable fraction, affected cases and mechanism confidence; absolute latency alone is insufficient.

Final gains require direct paired original/current-best measurements; intermediate percentages are never multiplied.
