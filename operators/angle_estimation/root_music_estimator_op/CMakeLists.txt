# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(root_music_estimator_op
    root_music_estimator_op.cu)
add_library(holoscan::root_music_estimator_op ALIAS root_music_estimator_op)

target_include_directories(root_music_estimator_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(root_music_estimator_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_spatial_smoothing
        holoscan::radar_common_source_count
    PRIVATE
        matx::matx
        CUDA::cusolver)

set_property(TARGET root_music_estimator_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS root_music_estimator_op
    EXPORT holoscan_root_music_estimator_op_targets
)

install(EXPORT holoscan_root_music_estimator_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES root_music_estimator_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)

add_subdirectory(python)
