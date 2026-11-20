# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The CLI builds the default target even for a single application. Limit that
# default to the selected application's dependency closure without deleting or
# rewriting cached APP_*, OP_* or PKG_* selections. Explicit --target builds can
# still address any enabled target. Full module/package/test builds keep ALL.
function(_holoscan_radar_exclude_default_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_property(type TARGET "${target}" PROPERTY TYPE)
        if(NOT type STREQUAL "INTERFACE_LIBRARY")
            set_property(TARGET "${target}" PROPERTY EXCLUDE_FROM_ALL TRUE)
        endif()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        _holoscan_radar_exclude_default_targets("${subdirectory}")
    endforeach()
endfunction()

function(holoscan_radar_select_application_build)
    set(HOLOSCAN_RADAR_APPLICATION "" CACHE STRING
        "Application whose dependencies form the default development build")
    if(NOT PROJECT_IS_TOP_LEVEL OR NOT HOLOSCAN_RADAR_APPLICATION OR BUILD_ALL OR
            HOLOSCAN_RADAR_BUILD_TESTING OR
            PKG_holoscan_radar OR MODULE_holoscan_radar OR SKBUILD)
        return()
    endif()
    if(NOT TARGET "${HOLOSCAN_RADAR_APPLICATION}")
        message(FATAL_ERROR
            "Selected application '${HOLOSCAN_RADAR_APPLICATION}' is not enabled")
    endif()
    if(HOLOSCAN_RADAR_BUILD_PYTHON)
        set(target holoscan_radar_python)
    else()
        set(target "${HOLOSCAN_RADAR_APPLICATION}")
    endif()
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Application build target '${target}' is unavailable")
    endif()
    _holoscan_radar_exclude_default_targets("${CMAKE_SOURCE_DIR}")
    add_custom_target(radar_application ALL DEPENDS "${target}")
    message(STATUS "Application default build: ${target} and its dependencies only")
endfunction()
