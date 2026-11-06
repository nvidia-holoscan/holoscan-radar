# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compiles every installed Radar header in its own translation unit against
# the installed package only, so a header that includes an uninstalled file
# fails here instead of in a downstream build.
cmake_minimum_required(VERSION 4.0)
project(RadarInstalledHeaders LANGUAGES CXX CUDA)

find_package(holoscan-radar CONFIG REQUIRED)
get_property(_radar_imported DIRECTORY PROPERTY IMPORTED_TARGETS)
list(FILTER _radar_imported INCLUDE REGEX "^holoscan::")

file(GLOB_RECURSE _radar_headers RELATIVE "${RADAR_INCLUDE_ROOT}"
    "${RADAR_INCLUDE_ROOT}/*.hpp" "${RADAR_INCLUDE_ROOT}/*.cuh")
if(NOT _radar_headers)
    message(FATAL_ERROR "No installed headers under ${RADAR_INCLUDE_ROOT}")
endif()
foreach(_header IN LISTS _radar_headers)
    string(MAKE_C_IDENTIFIER "${_header}" _id)
    set(_source "${CMAKE_CURRENT_BINARY_DIR}/tu/${_id}.cu")
    file(WRITE "${_source}" "#include <holoscan/radar/${_header}>\n")
    add_library(header_${_id} OBJECT "${_source}")
    target_compile_features(header_${_id} PRIVATE cxx_std_20 cuda_std_20)
    target_link_libraries(header_${_id} PRIVATE ${_radar_imported})
endforeach()
