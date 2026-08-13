# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# [EA HSDK5 ONLY] Probe optional Holoscan SDK features and track the HSDK 5
# operator port allow-list.
#
# HSDK 5 EA ships a lean core without domain operator modules (Holoviz,
# inference, sensor I/O, …). Radar subprojects that need those modules must
# gate on the HOLOSCAN_RADAR_HAS_* flags set here. Operators still written
# against the HSDK 4.x Operator API are skipped unless listed in
# HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS. Remove this file (and every consumer)
# before the public HSDK 5 release once the modules and port are complete.

# Holoviz — visualization library + HolovizOp. Required by demo apps and the
# radar_app viz/ helpers; report sinks (NullSinkOp, …) do not need it.
set(HOLOSCAN_RADAR_HAS_HOLOVIZ FALSE)
if(TARGET holoscan::viz AND TARGET holoscan::ops::holoviz)
    set(HOLOSCAN_RADAR_HAS_HOLOVIZ TRUE)
endif()

if(HOLOSCAN_RADAR_HAS_HOLOVIZ)
    message(STATUS "Holoviz: found (demo apps + radar_app viz enabled)")
else()
    message(STATUS
        "Holoviz: not found (holoscan::viz / holoscan::ops::holoviz); "
        "demo applications and Holoviz-backed radar_app symbols are disabled")
endif()

# Python bindings need holoscan::pybind11 (ABI helper) and a ported operator
# surface. Keep them off for the HSDK 5 pilot until pybind is restored. FORCE:
# the EA package cannot build them at all, so a stale cache (or a -D from a
# 4.x-era script) must not resurrect a configuration that fails to build.
# Re-enabling is a deliberate edit to this file, not a cache flip.
set(HOLOSCAN_RADAR_BUILD_PYTHON FALSE CACHE BOOL
    "Build holoscan.radar Python bindings (off during HSDK 5 EA pilot)" FORCE)
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    message(STATUS "Python bindings: enabled")
else()
    message(STATUS
        "[EA HSDK5 ONLY] Python bindings disabled "
        "(HOLOSCAN_RADAR_BUILD_PYTHON=OFF)")
endif()

# Operators that have been rewritten for the HSDK 5 Operator<> / Graph model.
# Expand this list as each operator lands; unlisted names are skipped at
# configure time so the tree still builds.
set(HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS
    clutter_filter_op
    snapshot_gather_op
    phase_shift_beamformer_op
    beamscan_estimator_op
    subarray_selection_op
    cluster_dbscan_op
    detection_reduction_op
    mvdr_estimator_op
    music_estimator_op
    feature_assembly_op
    cfar_detector_op
    root_music_estimator_op
    esprit_estimator_op
)

# Wrapper around add_holohub_operator that skips unported operators.
function(holoscan_radar_add_operator NAME)
    if(NOT "${NAME}" IN_LIST HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS)
        message(STATUS
            "[EA HSDK5 ONLY] Skipping unported operator ${NAME}")
        return()
    endif()
    add_holohub_operator(${NAME} ${ARGN})
endfunction()

# Set <out_var> TRUE when every named operator is on the port allow-list.
function(holoscan_radar_operators_ported out_var)
    foreach(_op IN LISTS ARGN)
        if(NOT "${_op}" IN_LIST HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS)
            set(${out_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

# Wrapper around add_holohub_application that skips apps whose operator
# dependencies are unported. DEPENDS OPERATORS FORCEs OP_<dep>=ON, but
# holoscan_radar_add_operator() still skips unported names, so the app would
# link against a target that was never created.
function(holoscan_radar_add_application NAME)
    set(_deps ${ARGN})
    list(REMOVE_ITEM _deps DEPENDS OPERATORS)
    holoscan_radar_operators_ported(_deps_ported ${_deps})
    if(NOT _deps_ported)
        message(STATUS
            "[EA HSDK5 ONLY] Skipping application ${NAME} "
            "(unported operator dependencies)")
        return()
    endif()
    add_holohub_application(${NAME} ${ARGN})
endfunction()
