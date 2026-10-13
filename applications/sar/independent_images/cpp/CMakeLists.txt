# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(yaml-cpp REQUIRED)
find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)

add_executable(independent_images independent_images.cpp)
target_include_directories(independent_images PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
    ${PROJECT_SOURCE_DIR}/applications/common)
target_link_libraries(independent_images PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::aperture_manager_op
    holoscan::backprojection_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::params_io
    yaml-cpp::yaml-cpp)
configure_file(../config/independent_images.yaml ${CMAKE_CURRENT_BINARY_DIR}/independent_images.yaml COPYONLY)
if(TARGET holoscan::visualization::op_holoviz)
    target_compile_definitions(independent_images PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    target_link_libraries(independent_images PRIVATE sar_image_viz
        holoscan::visualization::op_holoviz)
endif()

if(HOLOSCAN_RADAR_BUILD_TESTING)
    set(SAR_FIXTURE "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke.cphd")
    set(SAR_FIXTURE_SHIFTED "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke_shifted.cphd")
    set(SAR_FIXTURE_WIDE "${PROJECT_BINARY_DIR}/tests/cpp/generated_sar_smoke_wide.cphd")
    configure_file(independent_images_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/independent_images_generated_smoke.yaml @ONLY)
    configure_file(independent_images_viz_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/independent_images_viz_generated_smoke.yaml @ONLY)
    configure_file(independent_images_varying_viz_smoke.yaml.in
        ${CMAKE_CURRENT_BINARY_DIR}/independent_images_varying_viz_generated_smoke.yaml @ONLY)
endif()
