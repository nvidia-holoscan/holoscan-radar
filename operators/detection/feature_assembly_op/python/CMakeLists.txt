# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   feature_assembly_op
    CLASS_NAME         FeatureAssemblyOpF32
    SOURCES            _feature_assembly_op_bindings.cpp
)

target_include_directories(feature_assembly_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
