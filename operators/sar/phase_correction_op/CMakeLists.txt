# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

set_property(TARGET phase_correction_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS phase_correction_op
    EXPORT holoscan_phase_correction_op_targets
)

install(EXPORT holoscan_phase_correction_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES phase_correction_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
