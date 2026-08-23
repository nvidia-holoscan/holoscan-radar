# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

set(RADAR_SCHEMAS_GENERATED_ROOT
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(RADAR_SCHEMAS_GENERATED_DIR
    "${RADAR_SCHEMAS_GENERATED_ROOT}/radar_schemas")

holoscan_add_flatbuffer_schema(
    TARGET radar_window_spec_schema
    SCHEMAS window_spec.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    DESCRIPTOR_TENSOR
    TRAITS_HEADER window_spec_schema_traits.hpp
    TRAITS_ROOT_TYPE holoscan::radar::wire::WindowSpec
    TRAITS_NATIVE_TYPE holoscan::radar::wire::WindowSpecT
    TRAITS_BINARY_SCHEMA_TYPE
        holoscan::radar::wire::WindowSpecBinarySchema
    TRAITS_PACKAGE_NAMESPACE
        holoscan::radar::wire::window_spec_package
    TRAITS_CANONICAL_NAME
        com.nvidia.holoscan.radar.wire.WindowSpec
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 1024
    TRAITS_DESCRIPTOR_METADATA_CAPACITY 1024
)

holoscan_add_flatbuffer_schema(
    TARGET radar_fmcw_waveform_schema
    SCHEMAS fmcw_waveform.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    DESCRIPTOR_TENSOR
    TRAITS_HEADER fmcw_waveform_schema_traits.hpp
    TRAITS_ROOT_TYPE holoscan::radar::wire::FMCWWaveform
    TRAITS_NATIVE_TYPE holoscan::radar::wire::FMCWWaveformT
    TRAITS_BINARY_SCHEMA_TYPE
        holoscan::radar::wire::FMCWWaveformBinarySchema
    TRAITS_PACKAGE_NAMESPACE
        holoscan::radar::wire::fmcw_waveform_package
    TRAITS_CANONICAL_NAME
        com.nvidia.holoscan.radar.wire.FMCWWaveform
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 1024
    TRAITS_DESCRIPTOR_METADATA_CAPACITY 1024
)

function(radar_normalize_schema_traits SCHEMA_TARGET TRAITS_FILE)
    set(_stamp "${TRAITS_FILE}.normalized")
    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTRAITS_FILE=${TRAITS_FILE}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/normalize_traits.cmake"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS
            ${SCHEMA_TARGET}
            "${TRAITS_FILE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/normalize_traits.cmake"
        VERBATIM)
    add_custom_target(${SCHEMA_TARGET}_normalized ALL DEPENDS "${_stamp}")
endfunction()

radar_normalize_schema_traits(
    radar_window_spec_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/window_spec_schema_traits.hpp")
radar_normalize_schema_traits(
    radar_fmcw_waveform_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/fmcw_waveform_schema_traits.hpp")

add_library(radar_schemas INTERFACE)
add_library(holoscan::radar_schemas ALIAS radar_schemas)
add_dependencies(radar_schemas
    radar_window_spec_schema
    radar_fmcw_waveform_schema
    radar_window_spec_schema_normalized
    radar_fmcw_waveform_schema_normalized)

target_compile_features(radar_schemas INTERFACE cxx_std_20)
target_include_directories(radar_schemas INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${RADAR_SCHEMAS_GENERATED_ROOT}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar>)
target_link_libraries(radar_schemas INTERFACE
    holoscan::schema
    holoscan::radar_common)

install(TARGETS radar_schemas
    EXPORT holoscan_radar_schemas_targets)
install(EXPORT holoscan_radar_schemas_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar)
install(DIRECTORY "${RADAR_SCHEMAS_GENERATED_DIR}/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/radar_schemas"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")
install(FILES
    bridge_utils.hpp
    window_spec_bridge.hpp
    fmcw_waveform_bridge.hpp
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/radar_schemas")
