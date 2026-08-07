# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   radar_common
    CLASS_NAME         ArrayGeometry
    PYTHON_MODULE_NAME radar_common
    SOURCES            _radar_common_bindings.cpp
)

target_include_directories(radar_common_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
