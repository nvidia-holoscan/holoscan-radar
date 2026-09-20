# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Probe optional Holoscan SDK features and expose the public operator list
# used by package configuration.

include(CheckCXXSourceCompiles)

# The main-5x SDK exports the Graph-native Holoviz operator as one target.
set(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ FALSE)
if(TARGET holoscan::operators::holoviz)
    get_target_property(_HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS
        holoscan::operators::holoviz INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_HOLOSCAN_RADAR_INCLUDE_DIR
            IN LISTS _HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS)
        if(EXISTS
           "${_HOLOSCAN_RADAR_INCLUDE_DIR}/holoscan/operators/holoviz/holoviz_op.hpp")
            set(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ TRUE)
            break()
        endif()
    endforeach()
    unset(_HOLOSCAN_RADAR_INCLUDE_DIR)
    unset(_HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS)
endif()

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    message(STATUS "HSDK 5 Holoviz: found (Graph-native SIMO display enabled)")
else()
    message(STATUS
        "HSDK 5 Holoviz: not found (holoscan::operators::holoviz); "
        "Graph-native SIMO display disabled")
endif()

# X11 is used only to probe whether the HSDK 5 dashboard can open a display.
set(_HOLOSCAN_RADAR_X11_PROBE_SAVE_LIBS "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_LIBRARIES X11)
check_cxx_source_compiles("
#include <X11/Xlib.h>
int main() {
  Display* d = XOpenDisplay(nullptr);
  if (d) { XCloseDisplay(d); }
  return 0;
}
" HOLOSCAN_RADAR_HAS_X11)
set(CMAKE_REQUIRED_LIBRARIES "${_HOLOSCAN_RADAR_X11_PROBE_SAVE_LIBS}")
unset(_HOLOSCAN_RADAR_X11_PROBE_SAVE_LIBS)

if(HOLOSCAN_RADAR_HAS_X11)
    message(STATUS "X11: found (dashboard display probing enabled)")
else()
    message(STATUS "X11: not found (dashboard display probing disabled)")
endif()

# The current main-5x image does not export the Python binding target. Force
# the option off so stale CMake caches cannot select an unbuildable surface;
# see SDK_DEVELOPMENT.md.
set(HOLOSCAN_RADAR_BUILD_PYTHON FALSE CACHE BOOL
    "Build holoscan.radar Python bindings (SDK component unavailable)" FORCE)
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    message(STATUS "Python bindings: enabled")
else()
    message(STATUS
        "Python bindings disabled "
        "(HOLOSCAN_RADAR_BUILD_PYTHON=OFF)")
endif()

# Public operator target names, shared with package configuration.
set(HOLOSCAN_RADAR_OPERATORS
    clutter_filter_op
    snapshot_gather_op
    phase_shift_beamformer_op
    beamscan_estimator_op
    rank_one_residual_screen_op
    subarray_selection_op
    cluster_dbscan_op
    detection_reduction_op
    mvdr_estimator_op
    music_estimator_op
    feature_assembly_op
    multi_source_expansion_op
    cfar_detector_op
    root_music_estimator_op
    esprit_estimator_op
    mimo_deinterleave_op
    tdm_dense_compensation_op
    ddm_dense_demultiplex_op
    doppler_notch_op
    ddm_power_fold_op
    ddm_sparse_demultiplex_op
    tdm_sparse_compensation_op
    virtual_array_assembly_op
    fmcw_doppler_processor_op
    fmcw_range_processor_op
    fmcw_range_doppler_processor_op
)

function(holoscan_radar_add_operator NAME)
    add_holohub_operator(${NAME} ${ARGN})
endfunction()

function(holoscan_radar_add_application NAME)
    add_holohub_application(${NAME} ${ARGN})
endfunction()
