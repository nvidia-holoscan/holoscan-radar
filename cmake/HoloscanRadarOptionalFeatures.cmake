# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Probe optional Holoscan SDK features and expose the public operator list
# used by package configuration.

include(CheckCXXSourceCompiles)

# HSDK 5 packages visualization separately from the core Graph SDK. Look in
# the same installation as the selected holoscan package, even when the CLI
# also mounts a neighboring SDK checkout into the container.
find_package(holoscan-visualization QUIET CONFIG
    PATHS "${holoscan_DIR}" NO_DEFAULT_PATH)
set(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ FALSE)
if(TARGET holoscan::visualization::op_holoviz)
    get_target_property(_HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS
        holoscan::visualization::op_holoviz INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_HOLOSCAN_RADAR_INCLUDE_DIR
            IN LISTS _HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS)
        if(EXISTS
           "${_HOLOSCAN_RADAR_INCLUDE_DIR}/holoscan/visualization/holoviz_op.hpp")
            set(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ TRUE)
            break()
        endif()
    endforeach()
    unset(_HOLOSCAN_RADAR_INCLUDE_DIR)
    unset(_HOLOSCAN_RADAR_HSDK5_HOLOVIZ_INCLUDE_DIRS)
endif()

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    message(STATUS "HSDK 5 Holoviz: found (Graph-native dashboard enabled)")
else()
    message(STATUS
        "HSDK 5 Holoviz: not found (holoscan::visualization::op_holoviz); "
        "Graph-native dashboard disabled")
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

# HSDK 5 exposes Python as an installed package, rather than a CMake pybind11
# target. Require the Core and Visualization modules from the selected SDK root.
option(HOLOSCAN_RADAR_BUILD_PYTHON "Build the HSDK 5 holoscan.radar provider" OFF)
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
    set(HOLOSCAN_RADAR_SDK_PYTHON_SITE "${holoscan_DIR}/../../python3.12/site-packages")
    cmake_path(NORMAL_PATH HOLOSCAN_RADAR_SDK_PYTHON_SITE)
    foreach(_radar_module IN ITEMS holoscan/_core.abi3.so
            holoscan_visualization/_visualization.abi3.so)
        if(NOT EXISTS "${HOLOSCAN_RADAR_SDK_PYTHON_SITE}/${_radar_module}")
            message(FATAL_ERROR
                "HOLOSCAN_RADAR_BUILD_PYTHON requires HSDK 5 Python Core and "
                "Visualization at ${HOLOSCAN_RADAR_SDK_PYTHON_SITE}; missing ${_radar_module}")
        endif()
    endforeach()
    unset(_radar_module)
    message(STATUS "HSDK 5 Python Core and Visualization: found")
else()
    message(STATUS "Python bindings disabled (HOLOSCAN_RADAR_BUILD_PYTHON=OFF)")
endif()

# FMCW operators registered by the Python provider.
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
