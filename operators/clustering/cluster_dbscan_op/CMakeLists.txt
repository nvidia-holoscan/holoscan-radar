# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cluster_dbscan_op cluster_dbscan_op.cu)
add_library(holoscan::cluster_dbscan_op ALIAS cluster_dbscan_op)

target_include_directories(cluster_dbscan_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(cluster_dbscan_op
    PUBLIC
        holoscan::core
        holoscan::radar_common)

holoscan_radar_set_library_properties(cluster_dbscan_op)

install(TARGETS cluster_dbscan_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES cluster_dbscan_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/clustering
)
