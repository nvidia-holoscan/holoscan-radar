# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(COMMAND "${SAR_EXECUTABLE}" "${SAR_CONFIG}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "SAR app failed (${result}): ${out}${err}")
endif()
string(REGEX MATCHALL "image [0-9]+ formed, [0-9]+x[0-9]+" all_matches "${out}${err}")
string(REGEX MATCHALL "image [0-9]+ formed, ${SAR_DIMS}" expected_matches "${out}${err}")
list(LENGTH all_matches actual)
list(LENGTH expected_matches expected_dims)
if(NOT actual EQUAL SAR_COUNT OR NOT expected_dims EQUAL SAR_COUNT)
    message(FATAL_ERROR "Expected ${SAR_COUNT} images of ${SAR_DIMS}, got ${actual} images, "
        "${expected_dims} with correct dimensions: ${out}${err}")
endif()
