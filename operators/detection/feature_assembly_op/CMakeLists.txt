# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(feature_assembly_op feature_assembly_op.cu)
add_library(holoscan::feature_assembly_op ALIAS feature_assembly_op)

target_include_directories(feature_assembly_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(feature_assembly_op
    PUBLIC
        holoscan::core
        holoscan::radar_common)

holoscan_radar_set_library_properties(feature_assembly_op)

install(TARGETS feature_assembly_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES feature_assembly_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection
)
