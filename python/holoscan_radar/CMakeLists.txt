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
set(_radar_python_operators ${HOLOSCAN_RADAR_OPERATORS})
if(NOT HOLOSCAN_RADAR_PYTHON_TENSOR_DUMP)
    list(REMOVE_ITEM _radar_python_operators tensor_dump_op)
endif()
# Selecting any example app with a Python graph (or BUILD_ALL) adds the app
# bindings: simulator sources, reports, dashboards, presenters, config loaders,
# and, for SAR apps, the SAR graph runtime. Wheels select no apps.
set(_radar_python_apps)
set(_radar_python_sar_apps)
foreach(_radar_app IN LISTS HOLOSCAN_RADAR_PYTHON_FMCW_APPS HOLOSCAN_RADAR_PYTHON_SAR_APPS)
    if(BUILD_ALL OR APP_${_radar_app})
        list(APPEND _radar_python_apps ${_radar_app})
        if(_radar_app IN_LIST HOLOSCAN_RADAR_PYTHON_SAR_APPS)
            list(APPEND _radar_python_sar_apps ${_radar_app})
        endif()
    endif()
endforeach()
set(_radar_python_sources _radar.cpp _calibration.cpp)
if(_radar_python_apps)
    list(APPEND _radar_python_sources _siso.cpp _tdm.cpp _viz.cpp)
endif()
if(_radar_python_sar_apps)
    list(APPEND _radar_python_sources _sar.cpp)
endif()
nanobind_add_module(holoscan_radar_python NB_STATIC STABLE_ABI ${_radar_python_sources})
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
    "${PROJECT_SOURCE_DIR}")
target_link_libraries(holoscan_radar_python PRIVATE holoscan::core holoscan::params_io
    holoscan::radar_common holoscan::radar_schemas ${_radar_python_operators})
if(NOT HOLOSCAN_RADAR_PYTHON_TENSOR_DUMP)
    target_compile_definitions(holoscan_radar_python PRIVATE HOLOSCAN_RADAR_PYTHON_NO_TENSOR_DUMP=1)
endif()
configure_file(__init__.py "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/__init__.py" COPYONLY)
configure_file(py.typed "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/py.typed" COPYONLY)
configure_file(__init__.pyi "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/__init__.pyi" COPYONLY)
install(TARGETS holoscan_radar_python LIBRARY DESTINATION holoscan_radar)
install(FILES __init__.py __init__.pyi py.typed DESTINATION holoscan_radar)

# The wheel component covers only the provider and its native libraries.
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/wheel.cmake")

if(NOT _radar_python_apps)
    return()
endif()
if(NOT EXISTS "${HOLOSCAN_RADAR_SDK_PYTHON_SITE}/holoscan_visualization/_visualization.abi3.so")
    message(FATAL_ERROR "Python bindings for ${_radar_python_apps} require HSDK 5 Python "
        "Visualization at ${HOLOSCAN_RADAR_SDK_PYTHON_SITE}")
endif()
target_compile_definitions(holoscan_radar_python PRIVATE HOLOSCAN_RADAR_PYTHON_APPS=1)
target_include_directories(holoscan_radar_python PRIVATE "${PROJECT_SOURCE_DIR}/applications/common")
target_link_libraries(holoscan_radar_python PRIVATE
    holoscan::radar_common_simulator radar_app fmcw_dashboard)

# App graphs live beside their C++ counterparts. Stage them under
# holoscan_radar.applications so build-tree and installed imports match.
set(_radar_python_app_modules)
foreach(_radar_app IN LISTS HOLOSCAN_RADAR_PYTHON_FMCW_APPS)
    list(APPEND _radar_python_app_modules
        "${PROJECT_SOURCE_DIR}/applications/fmcw/${_radar_app}/python/${_radar_app}.py")
endforeach()
foreach(_radar_helper IN ITEMS __init__.py fmcw_dashboard.py cli.py presentation.py)
    list(APPEND _radar_python_app_modules "${CMAKE_CURRENT_SOURCE_DIR}/applications/${_radar_helper}")
endforeach()
if(_radar_python_sar_apps)
    target_compile_definitions(holoscan_radar_python PRIVATE HOLOSCAN_RADAR_PYTHON_SAR_APPS=1)
    target_link_libraries(holoscan_radar_python PRIVATE sar_mosaic)
    if(TARGET sar_image_viz)
        target_link_libraries(holoscan_radar_python PRIVATE sar_image_viz)
        target_compile_definitions(holoscan_radar_python PRIVATE HOLOSCAN_RADAR_SAR_HOLOVIZ=1)
    endif()
    get_filename_component(_radar_python_data_dir "${HOLOHUB_DATA_DIR}" ABSOLUTE
        BASE_DIR "${PROJECT_SOURCE_DIR}")
    target_compile_definitions(holoscan_radar_python PRIVATE
        HOLOSCAN_RADAR_DATA_DIR="${_radar_python_data_dir}")
    foreach(_radar_app IN LISTS _radar_python_sar_apps)
        list(APPEND _radar_python_app_modules
            "${PROJECT_SOURCE_DIR}/applications/sar/${_radar_app}/python/${_radar_app}.py")
    endforeach()
    list(APPEND _radar_python_app_modules "${CMAKE_CURRENT_SOURCE_DIR}/applications/sar_graph.py")
endif()
foreach(_radar_app_source IN LISTS _radar_python_app_modules)
    get_filename_component(_radar_app_module "${_radar_app_source}" NAME)
    configure_file("${_radar_app_source}"
        "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar/applications/${_radar_app_module}" COPYONLY)
    install(FILES "${_radar_app_source}" DESTINATION holoscan_radar/applications)
endforeach()
