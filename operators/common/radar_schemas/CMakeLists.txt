# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

set(RADAR_SCHEMAS_GENERATED_ROOT
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(RADAR_SCHEMAS_GENERATED_DIR
    "${RADAR_SCHEMAS_GENERATED_ROOT}/radar_schemas")

# HSDK 5 infers root type, native type, package, canonical name, and
# descriptor-Tensor mode from the BFBS. Traits generation is selected by
# MAX_SERIALIZED_SIZE (metadata-only bound for descriptor-Tensor schemas).
# Each .fbs root must declare a four-byte file_identifier.
holoscan_add_flatbuffer_schema(
    TARGET radar_array_geometry_schema
    SCHEMA array_geometry.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER array_geometry_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
)

holoscan_add_flatbuffer_schema(
    TARGET radar_window_spec_schema
    SCHEMA window_spec.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER window_spec_schema_traits.hpp
    MAX_SERIALIZED_SIZE 1024
)

holoscan_add_flatbuffer_schema(
    TARGET radar_fmcw_waveform_schema
    SCHEMA fmcw_waveform.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER fmcw_waveform_schema_traits.hpp
    MAX_SERIALIZED_SIZE 1024
)

holoscan_add_flatbuffer_schema(
    TARGET radar_coordinate_frame_schema
    SCHEMA coordinate_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER coordinate_frame_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
)

holoscan_add_flatbuffer_schema(
    TARGET radar_sample_geometry_schema
    SCHEMA sample_geometry.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RADAR_SCHEMAS_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER sample_geometry_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
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
    radar_array_geometry_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/array_geometry_schema_traits.hpp")
radar_normalize_schema_traits(
    radar_window_spec_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/window_spec_schema_traits.hpp")
radar_normalize_schema_traits(
    radar_fmcw_waveform_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/fmcw_waveform_schema_traits.hpp")

radar_normalize_schema_traits(
    radar_coordinate_frame_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/coordinate_frame_schema_traits.hpp")

radar_normalize_schema_traits(
    radar_sample_geometry_schema
    "${RADAR_SCHEMAS_GENERATED_DIR}/sample_geometry_schema_traits.hpp")

add_library(radar_schemas INTERFACE)
add_library(holoscan::radar_schemas ALIAS radar_schemas)
add_dependencies(radar_schemas
    radar_sample_geometry_schema
    radar_sample_geometry_schema_normalized
    radar_array_geometry_schema
    radar_coordinate_frame_schema
    radar_window_spec_schema
    radar_fmcw_waveform_schema
    radar_array_geometry_schema_normalized
    radar_coordinate_frame_schema_normalized
    radar_window_spec_schema_normalized
    radar_fmcw_waveform_schema_normalized)

target_compile_features(radar_schemas INTERFACE cxx_std_20)
target_include_directories(radar_schemas INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar>)
# The generated tree is -isystem, not -I: every flatc Pack() declares a
# _VectorArgs struct that only vector fields read, and the `(void)_va;` flatc
# emits to suppress that satisfies GCC but not nvcc's EDG frontend, which
# warns #550-D in every CUDA TU including a schema. Keep this call separate
# from the one above — the hand-written bridge headers next to this file stay
# on -I so their warnings still surface. INSTALL_INTERFACE needs no SYSTEM:
# CMake already treats an imported target's interface includes as system.
target_include_directories(radar_schemas SYSTEM INTERFACE
    $<BUILD_INTERFACE:${RADAR_SCHEMAS_GENERATED_ROOT}>)
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
    array_geometry_bridge.hpp
    window_spec_bridge.hpp
    fmcw_waveform_bridge.hpp
    lfm_waveform_bridge.hpp
    coordinate_frame_bridge.hpp
    sample_geometry_bridge.hpp
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/radar_schemas")
