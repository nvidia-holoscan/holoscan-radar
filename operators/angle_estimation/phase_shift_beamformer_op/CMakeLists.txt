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
        holoscan::radar_schemas
    PRIVATE
        matx::matx)

holoscan_radar_set_library_properties(phase_shift_beamformer_op)

install(TARGETS phase_shift_beamformer_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES phase_shift_beamformer_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)
