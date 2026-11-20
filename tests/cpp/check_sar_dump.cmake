# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

file(REMOVE "${SAR_DUMP_PATH}")
execute_process(COMMAND "${SAR_EXECUTABLE}" "${SAR_CONFIG}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 120)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "SAR dump smoke failed (${result}): ${out}${err}")
endif()
if(NOT EXISTS "${SAR_DUMP_PATH}")
    message(FATAL_ERROR "SAR dump did not write ${SAR_DUMP_PATH}")
endif()
file(READ "${SAR_DUMP_PATH}" magic LIMIT 6 HEX)
if(NOT magic STREQUAL "934e554d5059")
    message(FATAL_ERROR "SAR dump has no NumPy magic: ${magic}")
endif()
file(SIZE "${SAR_DUMP_PATH}" size)
if(size LESS 100)
    message(FATAL_ERROR "SAR dump is unexpectedly small: ${size} bytes")
endif()
file(REMOVE "${SAR_DUMP_PATH}")
