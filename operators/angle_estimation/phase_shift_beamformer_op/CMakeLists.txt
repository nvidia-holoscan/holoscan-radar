# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(phase_shift_beamformer_op
    phase_shift_beamformer_op.cu)
add_library(holoscan::phase_shift_beamformer_op ALIAS phase_shift_beamformer_op)

target_include_directories(phase_shift_beamformer_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(phase_shift_beamformer_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_steering
    PRIVATE
        matx::matx)

set_property(TARGET phase_shift_beamformer_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS phase_shift_beamformer_op
    EXPORT holoscan_phase_shift_beamformer_op_targets
)

install(EXPORT holoscan_phase_shift_beamformer_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES phase_shift_beamformer_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)

if(HOLOSCAN_RADAR_BUILD_PYTHON)
    add_subdirectory(python)
endif()
