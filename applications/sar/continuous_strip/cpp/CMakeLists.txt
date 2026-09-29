# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(yaml-cpp REQUIRED)
find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)

add_executable(continuous_strip continuous_strip.cpp)
target_include_directories(continuous_strip PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
    ${PROJECT_SOURCE_DIR}/applications/common)
target_link_libraries(continuous_strip PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::aperture_manager_op
    holoscan::backprojection_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::params_io
    yaml-cpp::yaml-cpp)
configure_file(../config/continuous_strip.yaml ${CMAKE_CURRENT_BINARY_DIR}/continuous_strip.yaml COPYONLY)
if(TARGET holoscan::visualization::op_holoviz)
    target_compile_definitions(continuous_strip PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    target_link_libraries(continuous_strip PRIVATE sar_image_viz
        holoscan::visualization::op_holoviz)
endif()

if(HOLOSCAN_RADAR_BUILD_TESTING)
    set(SAR_FIXTURE "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke.cphd")
    configure_file(continuous_strip_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/continuous_strip_generated_smoke.yaml @ONLY)
    configure_file(continuous_strip_viz_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/continuous_strip_viz_generated_smoke.yaml @ONLY)
endif()
