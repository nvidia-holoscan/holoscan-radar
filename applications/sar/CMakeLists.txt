# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve preset CPHD paths against the same data tree in CLI and devcontainer builds.
if(NOT DEFINED HOLOHUB_DATA_DIR)
    set(HOLOHUB_DATA_DIR "${PROJECT_SOURCE_DIR}/data" CACHE PATH "Application data directory")
endif()
get_filename_component(_sar_data_dir "${HOLOHUB_DATA_DIR}" ABSOLUTE
    BASE_DIR "${PROJECT_SOURCE_DIR}")
add_compile_definitions(HOLOSCAN_RADAR_DATA_DIR="${_sar_data_dir}")
unset(_sar_data_dir)

if(BUILD_ALL OR APP_spotlight OR APP_video_sar OR APP_tiled_strip OR
        APP_continuous_strip OR APP_independent_images)
    find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)
    if(TARGET holoscan::visualization::op_holoviz)
        add_library(sar_image_viz STATIC
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/sar_image_viz.cu
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/sar_image_lut.cu
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/timeline_raster.cpp)
        target_include_directories(sar_image_viz PRIVATE
        ${PROJECT_SOURCE_DIR}/operators ${PROJECT_SOURCE_DIR}/operators/common
        ${PROJECT_SOURCE_DIR}/applications/common)
        target_link_libraries(sar_image_viz PUBLIC holoscan::core holoscan::sensor_io
        holoscan::visualization::op_holoviz CUDA::cudart)
    endif()
endif()

holoscan_radar_add_application(spotlight
    DIRECTORY spotlight
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op tensor_dump_op)

holoscan_radar_add_application(video_sar
    DIRECTORY video_sar
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op tensor_dump_op)

holoscan_radar_add_application(tiled_strip
    DIRECTORY tiled_strip
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op tensor_dump_op)

holoscan_radar_add_application(continuous_strip
    DIRECTORY continuous_strip
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op tensor_dump_op)

holoscan_radar_add_application(independent_images
    DIRECTORY independent_images
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op tensor_dump_op)
