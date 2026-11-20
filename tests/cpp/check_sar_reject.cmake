# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(COMMAND "${SAR_EXECUTABLE}" "${SAR_CONFIG}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 30)
if(result EQUAL 0 OR NOT "${out}${err}" MATCHES "configuration selects another SAR topology")
    message(FATAL_ERROR "Expected topology rejection, got (${result}): ${out}${err}")
endif()
