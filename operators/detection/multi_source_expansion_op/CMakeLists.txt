# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(multi_source_expansion_op multi_source_expansion_op.cu)
add_library(holoscan::multi_source_expansion_op ALIAS multi_source_expansion_op)
target_include_directories(multi_source_expansion_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(multi_source_expansion_op PUBLIC
    holoscan::core holoscan::radar_common)
set_property(TARGET multi_source_expansion_op PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS multi_source_expansion_op
    EXPORT holoscan_radar_targets)
include(GNUInstallDirs)
install(FILES multi_source_expansion_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection)
