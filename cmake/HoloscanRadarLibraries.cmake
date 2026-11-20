# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standard properties for an installed Radar library. The holoscan_radar_ file
# prefix keeps generic names such as libradar_common.a from colliding in a
# shared libdir; VERSION and SOVERSION follow the SDK for shared builds. The
# target name and its exported holoscan::<name> name stay unchanged.
function(holoscan_radar_set_library_properties TARGET)
    set_target_properties(${TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "holoscan_radar_${TARGET}"
        VERSION "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_property(TARGET ${TARGET} PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()
endfunction()
