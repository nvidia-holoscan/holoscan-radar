# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(doppler_notch_op doppler_notch_op.cu)
add_library(holoscan::doppler_notch_op ALIAS doppler_notch_op)

target_include_directories(doppler_notch_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(doppler_notch_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

holoscan_radar_set_library_properties(doppler_notch_op)

install(TARGETS doppler_notch_op
    EXPORT holoscan_radar_targets)

include(GNUInstallDirs)
install(FILES doppler_notch_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)
