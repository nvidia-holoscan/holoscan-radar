# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoviz_device_repro LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED COMPONENTS core visualization)

add_executable(holoviz_broken broken.cpp)
add_executable(holoviz_workaround workaround.cpp)
foreach(repro IN ITEMS holoviz_broken holoviz_workaround)
    target_compile_features(${repro} PRIVATE cxx_std_20)
    target_link_libraries(${repro} PRIVATE
        holoscan::core holoscan::visualization::op_holoviz CUDA::cudart)
endforeach()
# Only the workaround requires the unsupported renderer API and its linkage.
target_link_libraries(holoviz_workaround PRIVATE holoscan::visualization::holoviz)
