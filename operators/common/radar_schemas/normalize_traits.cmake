# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED TRAITS_FILE)
    message(FATAL_ERROR "TRAITS_FILE is required")
endif()

file(READ "${TRAITS_FILE}" _traits)
set(_replacement "#include \"radar_schemas/bridge_utils.hpp\"\n")
string(FIND "${_traits}" "${_replacement}" _already_normalized)
if(NOT _already_normalized EQUAL -1)
    return()
endif()

set(_marker
    "// FlatBuffers 25.12 emits an unqualified CreateTensor call")
string(FIND "${_traits}" "${_marker}" _begin)
if(_begin EQUAL -1)
    message(FATAL_ERROR
        "Tensor forwarding block not found in ${TRAITS_FILE}")
endif()

string(SUBSTRING "${_traits}" ${_begin} -1 _tail)
string(FIND "${_tail}" "#include \"" _relative_include)
if(_relative_include EQUAL -1)
    message(FATAL_ERROR
        "Generated root include not found in ${TRAITS_FILE}")
endif()
math(EXPR _next_include "${_begin} + ${_relative_include}")

string(SUBSTRING "${_traits}" 0 ${_begin} _prefix)
string(SUBSTRING "${_traits}" ${_next_include} -1 _suffix)
file(WRITE "${TRAITS_FILE}" "${_prefix}${_replacement}\n${_suffix}")
