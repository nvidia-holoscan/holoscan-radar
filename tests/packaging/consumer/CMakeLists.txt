# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(RadarPackageConsumer LANGUAGES CXX CUDA)

find_package(holoscan-radar ${RADAR_VERSION} CONFIG REQUIRED ${RADAR_COMPONENT_ARGS})
# A second lookup must not redefine imported targets.
find_package(holoscan-radar CONFIG REQUIRED)
foreach(_target IN LISTS RADAR_EXPECT_TARGETS)
    if(NOT TARGET holoscan::${_target})
        message(FATAL_ERROR "Missing installed target: ${_target}")
    endif()
endforeach()
foreach(_target IN LISTS RADAR_ABSENT_TARGETS)
    if(TARGET holoscan::${_target})
        message(FATAL_ERROR "Unexpected installed target: ${_target}")
    endif()
endforeach()
add_executable(radar_package_consumer main.cpp)
target_compile_features(radar_package_consumer PRIVATE cxx_std_20)
target_link_libraries(radar_package_consumer PRIVATE holoscan::fmcw_range_processor_op)

# Link representative static dependency chains when present.
foreach(_target IN ITEMS fmcw_range_doppler_processor_op
        tdm_dense_compensation_op cphd_reader_op)
    if(TARGET holoscan::${_target})
        target_link_libraries(radar_package_consumer PRIVATE holoscan::${_target})
    endif()
endforeach()
