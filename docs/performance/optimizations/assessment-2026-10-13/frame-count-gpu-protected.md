# Fresh frame-count exploration: protected metrics

Companion to [frame-count exploration](frame-count-gpu.md). These are the existing median-of-repeat timing metrics, not the empirical-mixture distribution quantiles in the main report. Calibration has ten pairs; independent replication has thirty. All original early/full/setup/cold metrics are shown for both precisions. Limits and acceptance gates are unchanged; this table is diagnostic, not a new qualified baseline. CI family is the existing 156 scope-family multiplier times all protected metric/precision rows (usually 2,496); main distribution intervals protect a separate conservative family of 9,360. Intervals are sign-flipped for display so positive means A2 slower, matching the main report. The underlying review and gates remain unchanged.

| Phase / operator / case / frames | Precision / metric | A1 µs | A2 µs | P90 delta % / limit | IQR % / limit | Familywise A2-change CI % |
| --- | --- | --- | --- | --- | --- | --- |
| calibration / fmcw_range_processor / baseline-small / 50 | double / cold_ms | 268.075 | 270.247 | 2.563 / 20.0 | 2.133 / 30.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / early_median_ms | 44.933 | 45.051 | 1.080 / 5.0 | 0.626 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / early_p90_ms | 46.131 | 46.344 | 4.431 / 5.0 | 1.447 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / early_p99_ms | 48.790 | 51.763 | 19.722 / 5.0 | 12.409 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / median_ms | 44.933 | 45.051 | 1.080 / 5.0 | 0.626 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / p90_ms | 46.131 | 46.344 | 4.431 / 5.0 | 1.447 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / p99_ms | 48.790 | 51.763 | 19.722 / 5.0 | 12.409 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | double / setup_ms | 4020.475 | 4035.980 | 2.719 / 15.0 | 1.175 / 25.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / cold_ms | 264.659 | 258.223 | 3.426 / 20.0 | 2.744 / 30.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / early_median_ms | 26.544 | 26.504 | 0.874 / 5.0 | 0.637 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / early_p90_ms | 27.299 | 27.141 | 2.599 / 5.0 | 1.002 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / early_p99_ms | 27.884 | 27.943 | 30.759 / 5.0 | 6.609 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / median_ms | 26.544 | 26.504 | 0.874 / 5.0 | 0.637 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / p90_ms | 27.299 | 27.141 | 2.599 / 5.0 | 1.002 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / p99_ms | 27.884 | 27.943 | 30.759 / 5.0 | 6.609 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 50 | float / setup_ms | 4012.015 | 4004.305 | 3.662 / 15.0 | 1.445 / 25.0 | unresolved |
| replication / fmcw_range_processor / baseline-small / 50 | double / cold_ms | 269.533 | 269.103 | 2.272 / 20.0 | 1.678 / 30.0 | [-2.239, +1.842] |
| replication / fmcw_range_processor / baseline-small / 50 | double / early_median_ms | 44.778 | 44.843 | 1.549 / 5.0 | 0.791 / 10.0 | [-1.000, +1.095] |
| replication / fmcw_range_processor / baseline-small / 50 | double / early_p90_ms | 46.078 | 46.094 | 4.114 / 5.0 | 1.755 / 10.0 | [-2.133, +3.661] |
| replication / fmcw_range_processor / baseline-small / 50 | double / early_p99_ms | 49.518 | 51.048 | 17.519 / 5.0 | 12.966 / 10.0 | [-11.341, +14.869] |
| replication / fmcw_range_processor / baseline-small / 50 | double / median_ms | 44.778 | 44.843 | 1.549 / 5.0 | 0.791 / 10.0 | [-1.000, +1.095] |
| replication / fmcw_range_processor / baseline-small / 50 | double / p90_ms | 46.078 | 46.094 | 4.114 / 5.0 | 1.755 / 10.0 | [-2.133, +3.661] |
| replication / fmcw_range_processor / baseline-small / 50 | double / p99_ms | 49.518 | 51.048 | 17.519 / 5.0 | 12.966 / 10.0 | [-11.341, +14.869] |
| replication / fmcw_range_processor / baseline-small / 50 | double / setup_ms | 4010.690 | 4010.960 | 3.288 / 15.0 | 1.864 / 25.0 | [-2.881, +2.187] |
| replication / fmcw_range_processor / baseline-small / 50 | float / cold_ms | 259.116 | 258.440 | 3.138 / 20.0 | 1.913 / 30.0 | [-2.589, +2.135] |
| replication / fmcw_range_processor / baseline-small / 50 | float / early_median_ms | 26.470 | 26.470 | 1.649 / 5.0 | 0.760 / 10.0 | [-1.411, +0.753] |
| replication / fmcw_range_processor / baseline-small / 50 | float / early_p90_ms | 27.164 | 27.174 | 1.891 / 5.0 | 1.048 / 10.0 | [-1.626, +1.369] |
| replication / fmcw_range_processor / baseline-small / 50 | float / early_p99_ms | 27.943 | 28.050 | 29.511 / 5.0 | 11.739 / 10.0 | [-19.142, +21.787] |
| replication / fmcw_range_processor / baseline-small / 50 | float / median_ms | 26.470 | 26.470 | 1.649 / 5.0 | 0.760 / 10.0 | [-1.411, +0.753] |
| replication / fmcw_range_processor / baseline-small / 50 | float / p90_ms | 27.164 | 27.174 | 1.891 / 5.0 | 1.048 / 10.0 | [-1.626, +1.369] |
| replication / fmcw_range_processor / baseline-small / 50 | float / p99_ms | 27.943 | 28.050 | 29.511 / 5.0 | 11.739 / 10.0 | [-19.142, +21.787] |
| replication / fmcw_range_processor / baseline-small / 50 | float / setup_ms | 4008.975 | 4009.285 | 3.105 / 15.0 | 1.134 / 25.0 | [-2.263, +2.605] |
| calibration / detection_reduction / baseline-large / 50 | double / cold_ms | 548.667 | 543.394 | 6.106 / 20.0 | 1.378 / 30.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / early_median_ms | 294.582 | 294.463 | 1.814 / 5.0 | 0.462 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / early_p90_ms | 305.698 | 303.972 | 3.915 / 5.0 | 1.746 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / early_p99_ms | 333.073 | 332.649 | 13.478 / 5.0 | 2.264 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / median_ms | 294.582 | 294.463 | 1.814 / 5.0 | 0.462 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / p90_ms | 305.698 | 303.972 | 3.915 / 5.0 | 1.746 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / p99_ms | 333.073 | 332.649 | 13.478 / 5.0 | 2.264 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | double / setup_ms | 3838.555 | 3815.305 | 4.923 / 15.0 | 3.034 / 25.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / cold_ms | 549.982 | 544.686 | 5.629 / 20.0 | 2.949 / 30.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / early_median_ms | 294.031 | 294.253 | 0.920 / 5.0 | 0.369 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / early_p90_ms | 302.235 | 303.560 | 4.695 / 5.0 | 1.322 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / early_p99_ms | 332.803 | 334.094 | 4.981 / 5.0 | 2.459 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / median_ms | 294.031 | 294.253 | 0.920 / 5.0 | 0.369 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / p90_ms | 302.235 | 303.560 | 4.695 / 5.0 | 1.322 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / p99_ms | 332.803 | 334.094 | 4.981 / 5.0 | 2.459 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 50 | float / setup_ms | 3893.000 | 3994.165 | 6.142 / 15.0 | 4.631 / 25.0 | unresolved |
| replication / detection_reduction / baseline-large / 50 | double / cold_ms | 548.321 | 545.566 | 3.149 / 20.0 | 1.580 / 30.0 | [-3.139, +1.978] |
| replication / detection_reduction / baseline-large / 50 | double / early_median_ms | 294.195 | 293.975 | 1.137 / 5.0 | 0.434 / 10.0 | [-1.106, +0.605] |
| replication / detection_reduction / baseline-large / 50 | double / early_p90_ms | 304.857 | 303.246 | 3.821 / 5.0 | 1.775 / 10.0 | [-2.981, +1.794] |
| replication / detection_reduction / baseline-large / 50 | double / early_p99_ms | 333.978 | 335.346 | 8.361 / 5.0 | 3.741 / 10.0 | [-6.025, +3.390] |
| replication / detection_reduction / baseline-large / 50 | double / median_ms | 294.195 | 293.975 | 1.137 / 5.0 | 0.434 / 10.0 | [-1.106, +0.605] |
| replication / detection_reduction / baseline-large / 50 | double / p90_ms | 304.857 | 303.246 | 3.821 / 5.0 | 1.775 / 10.0 | [-2.981, +1.794] |
| replication / detection_reduction / baseline-large / 50 | double / p99_ms | 333.978 | 335.346 | 8.361 / 5.0 | 3.741 / 10.0 | [-6.025, +3.390] |
| replication / detection_reduction / baseline-large / 50 | double / setup_ms | 3913.005 | 3875.655 | 7.136 / 15.0 | 5.102 / 25.0 | [-6.369, +5.550] |
| replication / detection_reduction / baseline-large / 50 | float / cold_ms | 540.968 | 540.228 | 3.448 / 20.0 | 2.199 / 30.0 | [-3.443, +2.781] |
| replication / detection_reduction / baseline-large / 50 | float / early_median_ms | 294.114 | 294.192 | 0.913 / 5.0 | 0.436 / 10.0 | [-0.513, +0.681] |
| replication / detection_reduction / baseline-large / 50 | float / early_p90_ms | 304.002 | 304.063 | 2.831 / 5.0 | 1.561 / 10.0 | [-1.641, +1.999] |
| replication / detection_reduction / baseline-large / 50 | float / early_p99_ms | 334.587 | 330.943 | 6.549 / 5.0 | 3.375 / 10.0 | [-6.518, +2.527] |
| replication / detection_reduction / baseline-large / 50 | float / median_ms | 294.114 | 294.192 | 0.913 / 5.0 | 0.436 / 10.0 | [-0.513, +0.681] |
| replication / detection_reduction / baseline-large / 50 | float / p90_ms | 304.002 | 304.063 | 2.831 / 5.0 | 1.561 / 10.0 | [-1.641, +1.999] |
| replication / detection_reduction / baseline-large / 50 | float / p99_ms | 334.587 | 330.943 | 6.549 / 5.0 | 3.375 / 10.0 | [-6.518, +2.527] |
| replication / detection_reduction / baseline-large / 50 | float / setup_ms | 3825.525 | 3840.140 | 6.564 / 15.0 | 3.965 / 25.0 | [-3.829, +3.833] |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / cold_ms | 270.534 | 269.565 | 4.548 / 20.0 | 1.705 / 30.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / early_median_ms | 44.815 | 45.142 | 3.017 / 5.0 | 1.316 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / early_p90_ms | 46.199 | 46.560 | 4.639 / 5.0 | 2.065 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / early_p99_ms | 50.764 | 52.182 | 19.338 / 5.0 | 16.297 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / median_ms | 43.889 | 44.038 | 2.888 / 5.0 | 0.778 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / p90_ms | 45.193 | 45.477 | 2.732 / 5.0 | 0.992 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / p99_ms | 55.588 | 55.323 | 3.615 / 5.0 | 0.744 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | double / setup_ms | 4020.050 | 4054.750 | 4.039 / 15.0 | 1.785 / 25.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / cold_ms | 258.158 | 258.610 | 3.879 / 20.0 | 1.268 / 30.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / early_median_ms | 26.611 | 26.542 | 1.022 / 5.0 | 0.838 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / early_p90_ms | 27.160 | 27.240 | 2.340 / 5.0 | 1.534 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / early_p99_ms | 28.055 | 28.306 | 42.455 / 5.0 | 5.348 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / median_ms | 25.634 | 25.435 | 2.263 / 5.0 | 1.403 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / p90_ms | 26.686 | 26.491 | 1.171 / 5.0 | 0.970 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / p99_ms | 36.189 | 37.676 | 28.937 / 5.0 | 22.190 / 10.0 | unresolved |
| calibration / fmcw_range_processor / baseline-small / 2000 | float / setup_ms | 4001.580 | 4004.525 | 4.955 / 15.0 | 2.061 / 25.0 | unresolved |
| replication / fmcw_range_processor / baseline-small / 2000 | double / cold_ms | 270.166 | 270.429 | 3.281 / 20.0 | 1.857 / 30.0 | [-1.868, +2.458] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / early_median_ms | 44.834 | 44.908 | 1.581 / 5.0 | 0.785 / 10.0 | [-0.972, +1.558] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / early_p90_ms | 46.136 | 46.174 | 5.111 / 5.0 | 2.131 / 10.0 | [-1.671, +4.987] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / early_p99_ms | 51.667 | 52.677 | 18.507 / 5.0 | 15.826 / 10.0 | [-12.680, +18.372] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / median_ms | 43.882 | 43.843 | 1.531 / 5.0 | 0.865 / 10.0 | [-1.111, +1.333] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / p90_ms | 45.311 | 45.269 | 1.880 / 5.0 | 0.727 / 10.0 | [-1.502, +0.888] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / p99_ms | 55.825 | 55.726 | 3.451 / 5.0 | 1.636 / 10.0 | [-2.795, +2.240] |
| replication / fmcw_range_processor / baseline-small / 2000 | double / setup_ms | 4010.935 | 4022.250 | 3.653 / 15.0 | 1.535 / 25.0 | [-3.541, +1.689] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / cold_ms | 259.490 | 258.312 | 2.600 / 20.0 | 1.217 / 30.0 | [-2.530, +0.975] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / early_median_ms | 26.500 | 26.472 | 1.444 / 5.0 | 0.836 / 10.0 | [-1.356, +0.894] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / early_p90_ms | 27.232 | 27.190 | 1.435 / 5.0 | 1.341 / 10.0 | [-1.332, +1.327] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / early_p99_ms | 27.814 | 27.943 | 31.027 / 5.0 | 14.777 / 10.0 | [-21.196, +13.662] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / median_ms | 25.466 | 25.549 | 2.288 / 5.0 | 1.055 / 10.0 | [-1.554, +1.908] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / p90_ms | 26.500 | 26.556 | 1.421 / 5.0 | 0.767 / 10.0 | [-0.492, +1.409] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / p99_ms | 40.784 | 43.035 | 34.850 / 5.0 | 17.868 / 10.0 | [-13.071, +29.373] |
| replication / fmcw_range_processor / baseline-small / 2000 | float / setup_ms | 4025.220 | 4015.265 | 3.963 / 15.0 | 1.972 / 25.0 | [-1.807, +3.159] |
| calibration / detection_reduction / baseline-large / 2000 | double / cold_ms | 549.814 | 545.707 | 3.823 / 20.0 | 1.857 / 30.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / early_median_ms | 294.087 | 294.460 | 11.229 / 5.0 | 0.490 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / early_p90_ms | 305.018 | 305.052 | 9.699 / 5.0 | 1.750 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / early_p99_ms | 336.322 | 336.504 | 6.307 / 5.0 | 2.600 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / median_ms | 291.900 | 291.854 | 0.657 / 5.0 | 0.443 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / p90_ms | 302.708 | 302.559 | 8.160 / 5.0 | 1.419 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / p99_ms | 339.090 | 339.300 | 5.077 / 5.0 | 3.028 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | double / setup_ms | 3963.150 | 3941.095 | 6.471 / 15.0 | 5.010 / 25.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / cold_ms | 543.592 | 539.893 | 3.385 / 20.0 | 2.246 / 30.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / early_median_ms | 294.178 | 293.893 | 0.934 / 5.0 | 0.503 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / early_p90_ms | 304.174 | 304.140 | 1.947 / 5.0 | 1.353 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / early_p99_ms | 330.263 | 332.137 | 3.808 / 5.0 | 1.606 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / median_ms | 290.331 | 290.266 | 0.516 / 5.0 | 0.442 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / p90_ms | 301.333 | 302.045 | 0.975 / 5.0 | 0.575 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / p99_ms | 334.680 | 339.289 | 2.552 / 5.0 | 1.602 / 10.0 | unresolved |
| calibration / detection_reduction / baseline-large / 2000 | float / setup_ms | 3939.130 | 3847.135 | 3.318 / 15.0 | 6.071 / 25.0 | unresolved |
| replication / detection_reduction / baseline-large / 2000 | double / cold_ms | 547.253 | 548.154 | 3.517 / 20.0 | 2.001 / 30.0 | [-2.773, +3.226] |
| replication / detection_reduction / baseline-large / 2000 | double / early_median_ms | 294.263 | 294.367 | 0.900 / 5.0 | 0.427 / 10.0 | [-0.405, +0.859] |
| replication / detection_reduction / baseline-large / 2000 | double / early_p90_ms | 302.858 | 302.468 | 2.758 / 5.0 | 1.292 / 10.0 | [-1.286, +2.156] |
| replication / detection_reduction / baseline-large / 2000 | double / early_p99_ms | 330.389 | 334.549 | 5.699 / 5.0 | 2.704 / 10.0 | [-2.863, +4.636] |
| replication / detection_reduction / baseline-large / 2000 | double / median_ms | 291.763 | 291.964 | 0.711 / 5.0 | 0.416 / 10.0 | [-0.519, +0.631] |
| replication / detection_reduction / baseline-large / 2000 | double / p90_ms | 302.750 | 303.221 | 1.643 / 5.0 | 0.705 / 10.0 | [-0.555, +1.617] |
| replication / detection_reduction / baseline-large / 2000 | double / p99_ms | 337.109 | 338.344 | 3.164 / 5.0 | 1.402 / 10.0 | [-1.386, +2.878] |
| replication / detection_reduction / baseline-large / 2000 | double / setup_ms | 3910.200 | 4005.065 | 7.190 / 15.0 | 5.441 / 25.0 | [-3.907, +5.318] |
| replication / detection_reduction / baseline-large / 2000 | float / cold_ms | 539.567 | 542.271 | 5.207 / 20.0 | 2.092 / 30.0 | [-4.540, +3.387] |
| replication / detection_reduction / baseline-large / 2000 | float / early_median_ms | 294.211 | 293.934 | 0.997 / 5.0 | 0.325 / 10.0 | [-0.639, +0.728] |
| replication / detection_reduction / baseline-large / 2000 | float / early_p90_ms | 303.811 | 304.569 | 3.862 / 5.0 | 1.711 / 10.0 | [-2.349, +2.469] |
| replication / detection_reduction / baseline-large / 2000 | float / early_p99_ms | 332.041 | 333.450 | 7.000 / 5.0 | 3.069 / 10.0 | [-6.131, +4.313] |
| replication / detection_reduction / baseline-large / 2000 | float / median_ms | 290.673 | 290.539 | 0.655 / 5.0 | 0.523 / 10.0 | [-0.596, +0.489] |
| replication / detection_reduction / baseline-large / 2000 | float / p90_ms | 302.085 | 301.937 | 1.774 / 5.0 | 0.785 / 10.0 | [-0.926, +0.935] |
| replication / detection_reduction / baseline-large / 2000 | float / p99_ms | 337.467 | 337.792 | 4.885 / 5.0 | 2.062 / 10.0 | [-3.414, +3.342] |
| replication / detection_reduction / baseline-large / 2000 | float / setup_ms | 3843.975 | 3824.145 | 5.925 / 15.0 | 2.126 / 25.0 | [-5.147, +4.092] |
| calibration / music_estimator / baseline-small / 500 | double / cold_ms | 4860.175 | 4425.940 | 15.784 / 20.0 | 15.647 / 30.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / early_median_ms | 4221.783 | 4173.458 | 9.946 / 5.0 | 4.756 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / early_p90_ms | 5013.159 | 5070.614 | 11.604 / 5.0 | 2.612 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / early_p99_ms | 5381.436 | 5503.682 | 10.893 / 5.0 | 7.342 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / median_ms | 4152.271 | 4019.100 | 8.661 / 5.0 | 5.366 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / p90_ms | 4958.759 | 4625.163 | 13.363 / 5.0 | 9.950 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / p99_ms | 5542.122 | 5520.905 | 13.443 / 5.0 | 10.270 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | double / setup_ms | 5918.700 | 6010.820 | 4.225 / 15.0 | 2.001 / 25.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / cold_ms | 3666.320 | 3875.325 | 26.359 / 20.0 | 13.785 / 30.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / early_median_ms | 3458.244 | 3418.312 | 15.778 / 5.0 | 5.426 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / early_p90_ms | 4092.846 | 4195.777 | 13.377 / 5.0 | 6.654 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / early_p99_ms | 4329.461 | 4330.217 | 6.913 / 5.0 | 4.788 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / median_ms | 2865.202 | 2866.590 | 9.224 / 5.0 | 8.817 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / p90_ms | 3614.172 | 3636.145 | 8.301 / 5.0 | 8.948 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / p99_ms | 4248.428 | 4309.938 | 12.313 / 5.0 | 4.388 / 10.0 | unresolved |
| calibration / music_estimator / baseline-small / 500 | float / setup_ms | 6174.905 | 6107.985 | 3.449 / 15.0 | 2.103 / 25.0 | unresolved |
| replication / music_estimator / baseline-small / 500 | double / cold_ms | 4334.215 | 4247.570 | 19.191 / 20.0 | 14.802 / 30.0 | [-13.478, +16.555] |
| replication / music_estimator / baseline-small / 500 | double / early_median_ms | 4154.337 | 3974.378 | 15.264 / 5.0 | 8.174 / 10.0 | [-11.441, +6.421] |
| replication / music_estimator / baseline-small / 500 | double / early_p90_ms | 4979.414 | 4895.980 | 10.572 / 5.0 | 8.733 / 10.0 | [-10.211, +8.264] |
| replication / music_estimator / baseline-small / 500 | double / early_p99_ms | 5286.624 | 5317.438 | 11.699 / 5.0 | 4.852 / 10.0 | [-4.761, +6.190] |
| replication / music_estimator / baseline-small / 500 | double / median_ms | 4014.642 | 4002.080 | 7.761 / 5.0 | 6.368 / 10.0 | [-7.147, +6.159] |
| replication / music_estimator / baseline-small / 500 | double / p90_ms | 4601.412 | 4576.053 | 10.443 / 5.0 | 8.197 / 10.0 | [-7.748, +10.264] |
| replication / music_estimator / baseline-small / 500 | double / p99_ms | 5314.653 | 5315.598 | 10.620 / 5.0 | 3.833 / 10.0 | [-7.828, +8.864] |
| replication / music_estimator / baseline-small / 500 | double / setup_ms | 5956.475 | 5992.925 | 4.932 / 15.0 | 2.515 / 25.0 | [-3.735, +4.491] |
| replication / music_estimator / baseline-small / 500 | float / cold_ms | 3630.040 | 3624.690 | 18.212 / 20.0 | 9.014 / 30.0 | [-14.472, +7.466] |
| replication / music_estimator / baseline-small / 500 | float / early_median_ms | 3305.737 | 3307.618 | 18.654 / 5.0 | 17.339 / 10.0 | [-15.204, +14.227] |
| replication / music_estimator / baseline-small / 500 | float / early_p90_ms | 4064.805 | 4067.446 | 15.147 / 5.0 | 11.414 / 10.0 | [-12.521, +11.489] |
| replication / music_estimator / baseline-small / 500 | float / early_p99_ms | 4302.005 | 4165.030 | 16.472 / 5.0 | 12.989 / 10.0 | [-13.060, +12.024] |
| replication / music_estimator / baseline-small / 500 | float / median_ms | 2870.267 | 2861.889 | 16.896 / 5.0 | 7.372 / 10.0 | [-11.222, +14.980] |
| replication / music_estimator / baseline-small / 500 | float / p90_ms | 3638.428 | 3649.001 | 13.103 / 5.0 | 7.112 / 10.0 | [-8.371, +12.237] |
| replication / music_estimator / baseline-small / 500 | float / p99_ms | 4118.805 | 4162.431 | 12.510 / 5.0 | 7.084 / 10.0 | [-10.486, +5.850] |
| replication / music_estimator / baseline-small / 500 | float / setup_ms | 6046.880 | 6081.580 | 4.450 / 15.0 | 2.417 / 25.0 | [-1.763, +2.798] |

Raw observations, maxima, stalls, source cadence and every child reference remain in the ignored immutable session. Missing or unstable metrics prevent qualification; a favorable full-window result does not cancel early, setup or cold failures. No production changes.
