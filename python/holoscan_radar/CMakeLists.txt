# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The SDK pins nanobind 3.0.1. Resolve the same version from the selected
# Python installation instead of the removed holoscan::pybind11 target.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _radar_nanobind_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _radar_nanobind_result ERROR_QUIET)
if(_radar_nanobind_result EQUAL 0)
    find_package(nanobind 3.0.1 EXACT CONFIG REQUIRED
        PATHS "${_radar_nanobind_dir}" NO_DEFAULT_PATH)
else()
    include(FetchContent)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v3.0.1
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE)
    FetchContent_MakeAvailable(nanobind)
endif()
nanobind_add_module(holoscan_radar_python NB_STATIC STABLE_ABI _radar.cpp _siso.cpp _tdm.cpp _viz.cpp)
set_target_properties(holoscan_radar_python PROPERTIES
    OUTPUT_NAME _radar
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar"
    BUILD_RPATH "$ORIGIN/../../lib"
    INSTALL_RPATH "$ORIGIN/../lib")
if(BUILD_SHARED_LIBS)
    # HSDK compares payload and parameter type tokens by address. Export the
    # module's copies so they unify with those in the shared operator libraries.
    set_target_properties(holoscan_radar_python PROPERTIES CXX_VISIBILITY_PRESET default)
endif()
target_include_directories(holoscan_radar_python PRIVATE
    "${PROJECT_SOURCE_DIR}/operators"
    "${PROJECT_SOURCE_DIR}/operators/common"
    "${PROJECT_SOURCE_DIR}/applications/common"
    "${PROJECT_SOURCE_DIR}")
target_link_libraries(holoscan_radar_python PRIVATE holoscan::core holoscan::params_io
    holoscan::radar_common holoscan::radar_common_simulator holoscan::radar_schemas
    radar_app
    fmcw_dashboard
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
configure_file(__init__.py "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/__init__.py" COPYONLY)
configure_file(py.typed "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/py.typed" COPYONLY)
# HSDK 5's holoscan_add_nanobind_stubs() and holoscan_add_operator_stubs()
# describe native bindings and registry proxies. The public functions in
# __init__.py add Radar's extents/config conversion, so keep their facade stub
# until a generator can describe that call shape.
configure_file(__init__.pyi "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/__init__.pyi" COPYONLY)
install(TARGETS holoscan_radar_python LIBRARY DESTINATION holoscan_radar)
install(FILES __init__.py __init__.pyi py.typed DESTINATION holoscan_radar)
# App graphs live beside their C++ counterparts. Stage them under the installed
# holoscan_radar package so source builds and wheels expose the same imports.
foreach(_radar_app IN ITEMS siso simo dense_beamforming mimo_tdm mimo_ddm adaptive_aoa multi_waveform)
    set(_radar_app_source
        "${PROJECT_SOURCE_DIR}/applications/fmcw/${_radar_app}/python/${_radar_app}.py")
    configure_file("${_radar_app_source}"
        "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/applications/${_radar_app}.py" COPYONLY)
    install(FILES "${_radar_app_source}" DESTINATION holoscan_radar/applications)
endforeach()
install(FILES applications/__init__.py applications/fmcw_dashboard.py applications/cli.py
    DESTINATION holoscan_radar/applications)
foreach(_radar_helper IN ITEMS __init__.py fmcw_dashboard.py cli.py)
    configure_file("applications/${_radar_helper}"
        "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/applications/${_radar_helper}" COPYONLY)
endforeach()
