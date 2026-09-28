# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(yaml-cpp REQUIRED)

find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)
if(TARGET holoscan::visualization::op_holoviz)
    add_library(sar_image_viz STATIC sar_image_viz.cu sar_image_lut.cu)
    target_include_directories(sar_image_viz PRIVATE
        ${PROJECT_SOURCE_DIR}/operators
        ${PROJECT_SOURCE_DIR}/operators/common)
    target_link_libraries(sar_image_viz PUBLIC
        holoscan::core holoscan::sensor_io
        holoscan::visualization::op_holoviz CUDA::cudart)
endif()

add_executable(sar_formation sar_formation.cpp)
target_include_directories(sar_formation PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
    ${PROJECT_SOURCE_DIR}/applications/common)
target_link_libraries(sar_formation PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::aperture_manager_op
    holoscan::backprojection_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::params_io
    yaml-cpp::yaml-cpp)
foreach(mode 01 01_smoke 02 03 04 05)
    configure_file(mode${mode}.yaml ${CMAKE_CURRENT_BINARY_DIR}/mode${mode}.yaml COPYONLY)
endforeach()

if(TARGET holoscan::visualization::op_holoviz)
    target_compile_definitions(sar_formation PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    target_link_libraries(sar_formation PRIVATE sar_image_viz
        holoscan::visualization::op_holoviz)
endif()

if(HOLOSCAN_RADAR_BUILD_TESTING)
    set(SAR_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/generated_smoke.cphd")
    foreach(mode 01 02 03 04 05 01_viz)
        configure_file(smoke/mode${mode}.yaml.in
            ${CMAKE_CURRENT_BINARY_DIR}/mode${mode}_generated_smoke.yaml @ONLY)
    endforeach()
endif()
