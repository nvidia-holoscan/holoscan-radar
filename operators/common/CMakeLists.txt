# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
    radar_common/dsp/angle_convention.cpp
    radar_common/dsp/steering_weights.cpp
    radar_common/dsp/spatial_smoothing.cpp
    radar_common/dsp/subarray_selection.cpp
    radar_common/dsp/source_count.cpp
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

target_link_libraries(radar_common
    PUBLIC
        holoscan::core
)

target_compile_features(radar_common PUBLIC cxx_std_17)
set_property(TARGET radar_common PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common
    EXPORT holoscan_radar_common_targets
)

install(EXPORT holoscan_radar_common_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

install(DIRECTORY radar_common
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar
    FILES_MATCHING PATTERN "*.hpp"
)

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
    EXPORT holoscan_radar_common_reduce_targets
)

install(EXPORT holoscan_radar_common_reduce_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
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
    EXPORT holoscan_radar_common_steering_targets
)

install(EXPORT holoscan_radar_common_steering_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
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
    EXPORT holoscan_radar_common_simulator_targets
)

install(EXPORT holoscan_radar_common_simulator_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

# Python bindings for shared descriptors / enums (holoscan.radar.radar_common).
add_subdirectory(python)

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
    EXPORT holoscan_radar_common_spatial_smoothing_targets
)

install(EXPORT holoscan_radar_common_spatial_smoothing_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
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
    EXPORT holoscan_radar_common_source_count_targets
)

install(EXPORT holoscan_radar_common_source_count_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)
