# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared library for the supporting data structures and helpers used across
# operator groups (see README.md in this directory). Host-side C++ only in
# `radar_common`: MatX-facing helpers are header-only
# (radar_common/tensor/matx_utils.hpp) and are compiled into each operator's .cu
# translation units, so that target does not link MatX or CUDA.
# CUDA shared kernels (group-by-label reduce) live in `radar_common_reduce`.

add_library(radar_common STATIC
    radar_common/tensor/tensor_utils.cpp
    radar_common/dsp/window.cpp
    radar_common/dsp/ddm_code_generators.cpp
    radar_common/dsp/angle_convention.cpp
    radar_common/dsp/steering_weights.cpp
    radar_common/dsp/spatial_smoothing.cpp
    radar_common/dsp/shift_invariant_subarray.cpp
    radar_common/dsp/planar_aperture.cpp
    radar_common/dsp/subarray_selection.cpp
    radar_common/dsp/source_count.cpp
    radar_common/dsp/virtual_array.cpp
    radar_common/descriptors/tx_schedule.cpp
    radar_common/descriptors/ddm_code_descriptor.cpp
)
add_library(holoscan::radar_common ALIAS radar_common)

# Headers install under ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar (see the
# install(DIRECTORY) below), and code includes them as "radar_common/...",
# so that directory is the exported include root.
include(GNUInstallDirs)
target_include_directories(radar_common
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar>
)

# Installed radar_common/sdk YAML helpers and io/cphd_reader_op/collection_paths.hpp
# include yaml-cpp headers; HSDK 5 no longer re-exports yaml-cpp.
find_package(yaml-cpp CONFIG REQUIRED)
target_link_libraries(radar_common
    PUBLIC
        holoscan::core
        yaml-cpp::yaml-cpp
)

target_compile_features(radar_common PUBLIC cxx_std_20)
set_property(TARGET radar_common PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common
    EXPORT holoscan_radar_targets
)

install(DIRECTORY radar_common
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar
    FILES_MATCHING PATTERN "*.hpp"
)

# CUDA TDM/DDM coherent-compensation stages shared by the dense MIMO operator,
# sparse TDM operator, tests, and application planning helpers. The host-only
# radar_common target remains CUDA-free; this target owns the kernels and
# per-instance cuFFT scratch.
add_library(radar_common_mimo_compensation STATIC
    radar_common/dsp/mimo_coherent_compensation_stage.cu)
add_library(holoscan::radar_common_mimo_compensation ALIAS
    radar_common_mimo_compensation)

target_include_directories(radar_common_mimo_compensation
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar>
)

target_link_libraries(radar_common_mimo_compensation
    PUBLIC
        holoscan::radar_common
    PRIVATE
        CUDA::cufft
)

set_property(TARGET radar_common_mimo_compensation
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_mimo_compensation
    EXPORT holoscan_radar_targets
)

# Generated wire schemas and descriptor bridges depend on radar_common.
add_subdirectory(radar_schemas)

# CUDA group-by-label reduction primitives (public library surface for the
# DetectionReductionOp contract and any future consumer). Host-only
# radar_common stays CUDA-free; this target owns the .cu objects so operators
# link once instead of recompiling kernels (ODR).
add_library(radar_common_reduce STATIC
    radar_common/reduce/group_by_label.cu)
add_library(holoscan::radar_common_reduce ALIAS radar_common_reduce)

target_include_directories(radar_common_reduce
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(radar_common_reduce
    PUBLIC
        holoscan::radar_common
)

set_property(TARGET radar_common_reduce PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_reduce
    EXPORT holoscan_radar_targets
)

# CUDA device steering-weight builder (shared by the angle operators). Keeps
# the O(num_beams*num_channels) phase evaluation off the serial host path;
# radar_common stays CUDA-free, so the kernel owns its own .cu target (ODR).
add_library(radar_common_steering STATIC
    radar_common/dsp/steering_weights_device.cu)
add_library(holoscan::radar_common_steering ALIAS radar_common_steering)

target_include_directories(radar_common_steering
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(radar_common_steering
    PUBLIC
        holoscan::radar_common
)

set_property(TARGET radar_common_steering PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_steering
    EXPORT holoscan_radar_targets
)

# GPU-accelerated point-target simulator (MatX). The device counterpart of the
# STL simulate() in radar_common/sim/scene.hpp; links MatX privately.
add_library(radar_common_simulator STATIC
    radar_common/sim/simulator.cu)
add_library(holoscan::radar_common_simulator ALIAS radar_common_simulator)

target_include_directories(radar_common_simulator
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(radar_common_simulator
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx
)

set_property(TARGET radar_common_simulator PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_simulator
    EXPORT holoscan_radar_targets
)

# Python bindings for shared descriptors / enums (holoscan.radar.radar_common).

# CUDA spatial-smoothing gather (shared by the covariance angle estimators).
# Materializes sub-array "looks" from the host-planned index tables; the host
# planning (lattice detection, sub-array enumeration) stays in CUDA-free
# radar_common, so this target owns only the .cu (ODR).
add_library(radar_common_spatial_smoothing STATIC
    radar_common/dsp/spatial_smoothing_device.cu)
add_library(holoscan::radar_common_spatial_smoothing ALIAS
    radar_common_spatial_smoothing)

target_include_directories(radar_common_spatial_smoothing
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(radar_common_spatial_smoothing
    PUBLIC
        holoscan::radar_common
)

set_property(TARGET radar_common_spatial_smoothing
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_spatial_smoothing
    EXPORT holoscan_radar_targets
)

# CUDA source-count (EFT) kernel (shared by the subspace angle estimators:
# MUSIC / Root-MUSIC / ESPRIT). Batched per-item model-order decision from the
# eigenvalues matx::eig already produced on the GPU, with no host round trip;
# radar_common stays CUDA-free, so this target owns the .cu (ODR).
add_library(radar_common_source_count STATIC
    radar_common/dsp/source_count_device.cu)
add_library(holoscan::radar_common_source_count ALIAS
    radar_common_source_count)

target_include_directories(radar_common_source_count
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(radar_common_source_count
    PUBLIC
        holoscan::radar_common
)

set_property(TARGET radar_common_source_count
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common_source_count
    EXPORT holoscan_radar_targets
)
