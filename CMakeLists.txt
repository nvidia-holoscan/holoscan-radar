# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

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
    VERSION 0.1.0
    DESCRIPTION "Accelerated radar signal processing operators for the Holoscan ecosystem"
    LANGUAGES CXX CUDA
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# The Holoscan SDK install tree bundles MatX and holoscan-config.cmake chains
# find_dependency(MATX ...), so find_package(holoscan) also provides the
# matx::matx target used by the operator kernels.

find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED COMPONENTS core)


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
# Python package root in the build tree.
# PYTHONPATH=${HOLOSCAN_RADAR_PYTHON_ROOT} lets Python find
# holoscan.radar alongside the installed Holoscan SDK package.
# ---------------------------------------------------------------------------

set(HOLOSCAN_RADAR_PYTHON_ROOT
    ${CMAKE_BINARY_DIR}/python/lib)
set(HOLOSCAN_RADAR_PYTHON_PKG_DIR
    ${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan/radar)

# Variables consumed by pybind11_add_holohub_module:
#   HOLOHUB_PYTHON_MODULE_OUT_DIR — where per-operator subpackages land
#   HOLOSCAN_INSTALL_LIB_DIR     — relative lib dir, used for rpath resolution
set(HOLOHUB_PYTHON_MODULE_OUT_DIR
    ${HOLOSCAN_RADAR_PYTHON_PKG_DIR})
file(MAKE_DIRECTORY ${HOLOHUB_PYTHON_MODULE_OUT_DIR})
if(NOT DEFINED HOLOSCAN_INSTALL_LIB_DIR)
    set(HOLOSCAN_INSTALL_LIB_DIR lib)
endif()

# ---------------------------------------------------------------------------
# CMake helpers
# ---------------------------------------------------------------------------
# HoloHubConfigHelpers.cmake provides add_holohub_application/operator/package
# which gate each subproject on its own option (APP_/OP_/PKG_<name>). The
# helpers ship with the HoloHub CLI and are copied into this module's cmake/
# directory by the cookiecutter post-gen hook so the project also builds
# without a HoloHub clone.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HoloHubConfigHelpers)

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
# Stage dev-mode hook files in the build tree
# ---------------------------------------------------------------------------
# `./holohub install --dev` copies these into the user's
# Python user-site so `import holoscan.radar` works in
# any shell — without a wheel install. We only *stage* them here; the install
# itself is opt-in so a plain `./holohub build` never
# touches the host's Python environment.

set(_DEV_HELPER_FILE "${CMAKE_BINARY_DIR}/holoscan_radar_dev.py")
set(_DEV_PTH_FILE    "${CMAKE_BINARY_DIR}/holoscan-radar-dev.pth")
set(_DEV_TARGET_PATH "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan")

file(WRITE ${_DEV_HELPER_FILE}
"# Auto-generated by ./holohub build — do not edit.
# Extends holoscan.__path__ to include this module's build tree, so that
# `import holoscan.radar` resolves to the live build
# output without a wheel install. Installed to user-site by
# `./holohub install --dev`.
import os

_BUILD_PATH = r\"${_DEV_TARGET_PATH}\"

try:
    import holoscan
    if os.path.isdir(_BUILD_PATH) and _BUILD_PATH not in holoscan.__path__:
        holoscan.__path__.insert(0, _BUILD_PATH)
except Exception:
    # Stay silent — a stale dev hook must never abort Python startup.
    pass
")

file(WRITE ${_DEV_PTH_FILE}
"import holoscan_radar_dev
")

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
# Debian / wheel packaging lives under pkg/. pkg/CMakeLists.txt calls
# add_holohub_package(holoscan-radar OPERATORS <the fmcw operators>
# APPLICATIONS fmcw_siso), which defines the
# PKG_holoscan_radar option, FORCEs the OP_/APP_ deps to ON when the
# package is enabled, and enters pkg/holoscan-radar/CMakeLists.txt
# for the holohub_configure_deb() call.
