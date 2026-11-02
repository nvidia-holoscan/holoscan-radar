# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoviz_device_repro LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED COMPONENTS core visualization)

# Only the workaround needs renderer headers/linkage. The baseline uses the public operator API.
option(REPRO_INTERNAL_DISCOVERY "Build the unsupported renderer-device discovery workaround" OFF)
add_executable(holoviz_device main.cpp)
target_compile_features(holoviz_device PRIVATE cxx_std_20)
target_link_libraries(holoviz_device PRIVATE
    holoscan::core holoscan::visualization::op_holoviz CUDA::cudart)
if(REPRO_INTERNAL_DISCOVERY)
    target_compile_definitions(holoviz_device PRIVATE REPRO_INTERNAL_DISCOVERY=1)
    target_link_libraries(holoviz_device PRIVATE holoscan::visualization::holoviz)
endif()
