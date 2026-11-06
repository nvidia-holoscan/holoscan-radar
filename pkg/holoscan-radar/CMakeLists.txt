# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-radar ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_radar is ON.
include(holohub_configure_deb)

# Package selection runs before operators/; wait for the final target graph so
# dependencies also reflect CPHD enabled by apps or explicit cached preferences.
function(holoscan_radar_configure_deb)
    # Installed headers include yaml-cpp, and radar_common exports yaml-cpp::yaml-cpp.
    set(_radar_deb_depends "holoscan (>= 5.0.0), libyaml-cpp-dev (>= 0.8.0)")
    if(TARGET cphd_reader_op)
        # Installed CMake discovery needs tinyxml2's config, not just its runtime.
        string(APPEND _radar_deb_depends ", libtinyxml2-dev")
    endif()
    holohub_configure_deb(
        NAME        "holoscan-radar"
        DESCRIPTION "${PROJECT_DESCRIPTION}"
        VERSION     "${PROJECT_VERSION}"
        VENDOR      "NVIDIA"
        CONTACT     "jjomier@nvidia.com"
        DEPENDS     "${_radar_deb_depends}"
    )
endfunction()
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL holoscan_radar_configure_deb)
