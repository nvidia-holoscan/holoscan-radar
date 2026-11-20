# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)

add_executable(autofocus autofocus.cpp)
target_include_directories(autofocus PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
    ${PROJECT_SOURCE_DIR}/applications/common)
target_link_libraries(autofocus PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::aperture_manager_op
    holoscan::backprojection_op
    holoscan::autofocus_op
    holoscan::phase_correction_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::params_io)
configure_file(../config/autofocus.yaml ${CMAKE_CURRENT_BINARY_DIR}/autofocus.yaml COPYONLY)
foreach(scenario blurred corrected reference)
    configure_file(../config/autofocus_${scenario}.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_${scenario}.yaml COPYONLY)
endforeach()
# The scene-axis presets operators/sar/autofocus_op/tools/sweep.py sweeps across. Copied so
# they can be run by hand the same way as the Kourou default.
foreach(scene atlanta houston imperial)
    configure_file(../config/autofocus_${scene}.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_${scene}.yaml COPYONLY)
endforeach()
if(TARGET holoscan::visualization::op_holoviz)
    target_compile_definitions(autofocus PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    target_link_libraries(autofocus PRIVATE sar_image_viz
        holoscan::visualization::op_holoviz)
endif()

if(HOLOSCAN_RADAR_BUILD_TESTING)
    set(SAR_FIXTURE "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke_autofocus.cphd")
    configure_file(autofocus_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_generated_smoke.yaml @ONLY)
    configure_file(autofocus_smoke_lie.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_generated_smoke_lie.yaml @ONLY)
    configure_file(autofocus_smoke_oracle.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_generated_smoke_oracle.yaml @ONLY)
    configure_file(autofocus_smoke_polar.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/autofocus_generated_smoke_polar.yaml @ONLY)
endif()
