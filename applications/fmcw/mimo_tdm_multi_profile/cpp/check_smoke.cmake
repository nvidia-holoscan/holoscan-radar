# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(
    COMMAND "${APP}" "${CONFIG}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
set(log "${stdout}${stderr}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "multi-profile smoke failed (${result}):\n${log}")
endif()

set(near "multi-profile axes: frame=0 profile=near range_bins=256 velocity_bins=64 effective_pri_ns=50000")
set(far "multi-profile axes: frame=1 profile=far range_bins=512 velocity_bins=64 effective_pri_ns=90000")
string(FIND "${log}" "${near}" near_position)
string(FIND "${log}" "${far}" far_position)
string(FIND "${log}" "key: c2c:nfft=512:len=512" far_range_position)
string(FIND "${log}"
    "sparse TDM correction table configuration; building it now and caching for reuse (key: 2|64|7.7e+10|9e-05|slots:0:1)"
    far_compensation_position)
if(near_position LESS 0 OR far_position LESS 0 OR
   far_range_position LESS 0 OR far_compensation_position LESS 0 OR
   NOT near_position LESS far_position OR
   NOT far_position LESS far_range_position)
    message(FATAL_ERROR
        "multi-profile smoke did not observe ordered Exact axes:\n${log}")
endif()
