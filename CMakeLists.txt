# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Default the CUDA architectures before project() enables the CUDA language,
# following the SDK's MatX examples: `native` targets the GPU present at
# build time; numeric values below 70 are floored (MatX requires compute
# capability >= 7.0). Override with -DCMAKE_CUDA_ARCHITECTURES for CI or
# cross-target builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$"
       AND CMAKE_CUDA_ARCHITECTURES LESS 70)
    message(STATUS "MatX requires CUDA compute capability >= 7.0; using 70")
    set(CMAKE_CUDA_ARCHITECTURES "70")
endif()

project(holoscan_radar
    VERSION 0.2.0
    DESCRIPTION "Accelerated radar signal processing operators for the Holoscan ecosystem"
    LANGUAGES CXX CUDA
)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# HSDK 4.x re-exported MatX from find_package(holoscan) via find_dependency.
# HSDK 5 keeps MatX in the SDK install tree but does not pull it in
# transitively — find it explicitly.

find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED COMPONENTS core params_io testing)
find_package(matx REQUIRED)


# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
# BUILD_ALL toggles every operator/application/package on by default. It
# matches the HoloHub-monorepo convention: each subproject is a CMake option
# (e.g. APP_<name>, OP_<name>, PKG_<name>) defined via add_holohub_application
# / add_holohub_operator / add_holohub_package, defaulting to ${BUILD_ALL}.
# To build a single subproject, configure with -DBUILD_ALL=OFF -DAPP_<name>=ON.
#
# Default to ON only when this is the top-level CMake project (standalone
# build). When nested under another project (e.g. consumed as a HoloHub
# external dependency via add_subdirectory), defer to whatever the parent
# build has decided — otherwise this module's BUILD_ALL=ON would leak into
# the parent's variable scope and force-enable unrelated subprojects there.

if(NOT DEFINED BUILD_ALL)
    if(PROJECT_IS_TOP_LEVEL)
        set(BUILD_ALL ON CACHE BOOL "Build all operators, applications, and packages by default")
    else()
        set(BUILD_ALL OFF)
    endif()
endif()

# Module tests live behind a module-namespaced flag rather than the standard
# CMake BUILD_TESTING variable. This lets a parent project (HoloHub, another
# module) keep its own BUILD_TESTING setting independent of whether *this*
# module's tests are built. Defaults ON when standalone, OFF when nested.
if(NOT DEFINED HOLOSCAN_RADAR_BUILD_TESTING)
    if(PROJECT_IS_TOP_LEVEL)
        set(HOLOSCAN_RADAR_BUILD_TESTING ON
            CACHE BOOL "Build the radar module's CTest and pytest suites")
    else()
        set(HOLOSCAN_RADAR_BUILD_TESTING OFF
            CACHE BOOL "Build the radar module's CTest and pytest suites")
    endif()
endif()

# Performance characterization suite (performance/). Off by default; it builds
# standalone benchmark executables (not part of the operator libraries or the
# CTest unit/statistical suites) that measure setup, steady-state compute, and
# live-override latency. Enable with -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON.
if(NOT DEFINED HOLOSCAN_RADAR_BUILD_PERFORMANCE)
    set(HOLOSCAN_RADAR_BUILD_PERFORMANCE OFF
        CACHE BOOL "Build the radar module's performance characterization suite")
endif()

# ---------------------------------------------------------------------------
# Python package root in the build tree. Add this directory to PYTHONPATH for
# canonical ``holoscan_radar`` imports during development.
# ---------------------------------------------------------------------------

set(HOLOSCAN_RADAR_PYTHON_ROOT
    ${CMAKE_BINARY_DIR}/python/lib)

if(NOT DEFINED HOLOSCAN_INSTALL_LIB_DIR)
    set(HOLOSCAN_INSTALL_LIB_DIR lib)
endif()

# ---------------------------------------------------------------------------
# CMake helpers
# ---------------------------------------------------------------------------
# HoloHubConfigHelpers.cmake provides add_holohub_application/operator/package
# which gate each subproject on its own option (APP_/OP_/PKG_<name>). The
# helpers originated in HoloHub and are copied into this module's cmake/
# directory by the cookiecutter post-gen hook so the project also builds
# without a HoloHub clone.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HoloHubConfigHelpers)

# Optional SDK features (Holoviz, …) — see HoloscanRadarOptionalFeatures.cmake.
include(HoloscanRadarOptionalFeatures)

# The provider links every public FMCW operator even when BUILD_ALL is OFF.
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    foreach(_radar_op IN LISTS HOLOSCAN_RADAR_OPERATORS)
        set(OP_${_radar_op} ON CACHE BOOL "Required by the Radar Python provider" FORCE)
    endforeach()
    unset(_radar_op)
endif()

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
# pkg/ must come *before* operators/ and applications/: add_holohub_package()
# inside pkg/ uses set(... CACHE BOOL "" FORCE) to flip OP_<x>=ON / APP_<x>=ON
# when its PKG_<name> option is ON, and the operator/application macros that
# read those values run their `option(OP_<x> ... ${BUILD_ALL})` calls when
# their subdirectories are processed — those bind the cache value the FIRST
# time option() is encountered, so the FORCE has to happen before then.

add_subdirectory(pkg)
add_subdirectory(applications)
add_subdirectory(operators)

if(HOLOSCAN_RADAR_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HOLOSCAN_RADAR_BUILD_PERFORMANCE)
    add_subdirectory(performance)
endif()

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
# Debian / wheel packaging lives under pkg/. pkg/CMakeLists.txt calls
# add_holohub_package(holoscan-radar OPERATORS <operators> APPLICATIONS <apps>),
# which defines the PKG_holoscan_radar option, FORCEs the OP_/APP_ deps to ON
# when the package is enabled, and enters pkg/holoscan-radar/CMakeLists.txt for
# the holohub_configure_deb() call.
