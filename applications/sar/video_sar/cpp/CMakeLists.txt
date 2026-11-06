# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)

add_executable(video_sar video_sar.cpp)
target_include_directories(video_sar PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
    ${PROJECT_SOURCE_DIR}/applications/common)
target_link_libraries(video_sar PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::aperture_manager_op
    holoscan::backprojection_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::params_io)
configure_file(../config/video_sar.yaml ${CMAKE_CURRENT_BINARY_DIR}/video_sar.yaml COPYONLY)
if(TARGET holoscan::visualization::op_holoviz)
    target_compile_definitions(video_sar PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    target_link_libraries(video_sar PRIVATE sar_image_viz
        holoscan::visualization::op_holoviz)
endif()

if(HOLOSCAN_RADAR_BUILD_TESTING)
    set(SAR_FIXTURE "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke.cphd")
    set(SAR_FIXTURE_LONG "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke_long.cphd")
    configure_file(video_sar_retention_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/video_sar_retention_generated_smoke.yaml @ONLY)
    configure_file(video_sar_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/video_sar_generated_smoke.yaml @ONLY)
    configure_file(video_sar_viz_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/video_sar_viz_generated_smoke.yaml @ONLY)
endif()
