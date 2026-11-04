# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(RadarPackageConsumer LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED COMPONENTS core params_io)
find_package(CUDAToolkit REQUIRED)
find_package(matx REQUIRED)

# Radar currently exports individual targets rather than an umbrella Config file.
foreach(_export IN ITEMS radar_common radar_schemas fmcw_range_processor_op)
    include("${RADAR_PREFIX}/lib/cmake/holoscan_radar/holoscan_${_export}_targets.cmake")
endforeach()
add_executable(radar_package_consumer main.cpp)
target_compile_features(radar_package_consumer PRIVATE cxx_std_20)
target_link_libraries(radar_package_consumer PRIVATE holoscan::fmcw_range_processor_op)
