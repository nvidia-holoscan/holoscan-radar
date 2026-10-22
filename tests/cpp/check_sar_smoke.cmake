# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(COMMAND "${SAR_EXECUTABLE}" "${SAR_CONFIG}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "SAR app failed (${result}): ${out}${err}")
endif()
# SarImageReportOp INFO-logs at most once per second, and again when the formed
# size changes. Fast fixture runs therefore emit fewer "image N formed" lines
# than images. ImageCount is the "stopped after N image(s)" line.
set(log "${out}${err}")
if(NOT log MATCHES "stopped after ${SAR_COUNT} image\\(s\\)")
    message(FATAL_ERROR "Expected stopped after ${SAR_COUNT} image(s): ${log}")
endif()
string(REGEX MATCHALL "image [0-9]+ formed, [0-9]+x[0-9]+" all_matches "${log}")
string(REGEX MATCHALL "image [0-9]+ formed, ${SAR_DIMS}" expected_matches "${log}")
list(LENGTH all_matches actual)
list(LENGTH expected_matches expected_dims)
if(actual LESS 1 OR NOT expected_dims EQUAL actual)
    message(FATAL_ERROR "Expected at least one ${SAR_DIMS} formed line and no other sizes, "
        "got ${actual} formed lines, ${expected_dims} matching: ${log}")
endif()
