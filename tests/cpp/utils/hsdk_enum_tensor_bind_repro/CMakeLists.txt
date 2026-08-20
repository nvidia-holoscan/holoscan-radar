# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Drop-in build for Holoscan maintainers. Copy this directory only.
#   cmake -S . -B build
#   cmake --build build
#   ./build/enum_tensor_bind_repro
# Optional: -Dholoscan_DIR=/path/to/holoscan/lib/cmake/holoscan

cmake_minimum_required(VERSION 3.20)
project(enum_tensor_bind_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(enum_tensor_bind_repro enum_tensor_bind_repro.cpp)
target_link_libraries(enum_tensor_bind_repro PRIVATE holoscan::core CUDA::cudart)
