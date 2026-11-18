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

if((HOLOSCAN_RADAR_BUILD_PYTHON AND HOLOSCAN_RADAR_BUILD_PYTHON_APPS) OR BUILD_ALL OR APP_spotlight OR APP_video_sar OR APP_tiled_strip OR
        APP_continuous_strip OR APP_independent_images OR APP_autofocus)
    find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core visualization)
    # SarMosaicOp decimates its parent view with a CUDA kernel, and it is used
    # by the dump and report paths as well as the display, so unlike
    # sar_image_viz this cannot be gated on Holoviz being present. Only the
    # two strip apps build a mosaic at all, so it is gated on them rather than
    # on any SAR app -- a spotlight-only configure was compiling a translation
    # unit nothing linked.
    if((HOLOSCAN_RADAR_BUILD_PYTHON AND HOLOSCAN_RADAR_BUILD_PYTHON_APPS) OR BUILD_ALL OR APP_tiled_strip OR APP_continuous_strip)
        add_library(sar_mosaic STATIC
            ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/mosaic_sample.cu)
        target_include_directories(sar_mosaic PRIVATE
            ${PROJECT_SOURCE_DIR}/applications/common)
        target_link_libraries(sar_mosaic PUBLIC CUDA::cudart)
        set_property(TARGET sar_mosaic PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
    if(TARGET holoscan::visualization::op_holoviz)
        add_library(sar_image_viz STATIC
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/sar_image_viz.cu
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/sar_image_lut.cu
        ${PROJECT_SOURCE_DIR}/applications/common/radar_app/sar/timeline_raster.cu)
        target_include_directories(sar_image_viz PRIVATE
        ${PROJECT_SOURCE_DIR}/operators ${PROJECT_SOURCE_DIR}/operators/common
        ${PROJECT_SOURCE_DIR}/applications/common)
        target_link_libraries(sar_image_viz PUBLIC holoscan::core holoscan::sensor_io
        holoscan::visualization::op_holoviz CUDA::cudart)
        set_property(TARGET sar_image_viz PROPERTY POSITION_INDEPENDENT_CODE ON)
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

holoscan_radar_add_application(autofocus
    DIRECTORY autofocus
    DEPENDS OPERATORS cphd_reader_op sar_range_compression_op
        aperture_manager_op backprojection_op autofocus_op
        phase_correction_op tensor_dump_op)
