# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(phase_correction_op phase_correction_op.cu)
add_library(holoscan::phase_correction_op ALIAS phase_correction_op)

target_include_directories(phase_correction_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(phase_correction_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        CUDA::cudart)

holoscan_radar_set_library_properties(phase_correction_op)

install(TARGETS phase_correction_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES phase_correction_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
