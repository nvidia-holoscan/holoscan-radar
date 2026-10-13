# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(COMMAND "${SAR_EXECUTABLE}" "${SAR_CONFIG}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "SAR varying-size visualization failed (${result}): ${out}${err}")
endif()
string(REGEX MATCHALL "image [0-9]+ formed, [0-9]+x[0-9]+" images "${out}${err}")
list(LENGTH images count)
if(NOT count EQUAL 2 OR
        NOT "${out}${err}" MATCHES "image 1 formed, 1x1" OR
        NOT "${out}${err}" MATCHES "image 2 formed, 8x8")
    message(FATAL_ERROR "Expected 1x1 then 8x8 images, got: ${out}${err}")
endif()
