# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(aperture_manager_op aperture_manager_op.cpp)
add_library(holoscan::aperture_manager_op ALIAS aperture_manager_op)

target_include_directories(aperture_manager_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(aperture_manager_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_schemas
        CUDA::cudart)

holoscan_radar_set_library_properties(aperture_manager_op)

install(TARGETS aperture_manager_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES aperture_manager_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
