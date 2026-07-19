# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# radar_app is header-only (INTERFACE). pybind11_add_holohub_module needs a
# real linkable CPP_CMAKE_TARGET, so link radar_common and then attach the
# INTERFACE radar_app deps (viz / cudart / include roots) explicitly. Inline
# operator methods compile into this .so from the headers.
include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   radar_common
    CLASS_NAME         DetectionReportOp
    PYTHON_MODULE_NAME radar_app
    SOURCES            _radar_app_bindings.cpp
)

target_link_libraries(radar_app_python PRIVATE
    holoscan::radar_app
    holoscan::viz
    CUDA::cudart
)

target_include_directories(radar_app_python PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators/common
)
