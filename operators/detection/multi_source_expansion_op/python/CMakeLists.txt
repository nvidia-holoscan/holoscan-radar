# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   multi_source_expansion_op
    CLASS_NAME         MultiSourceExpansionOpF32
    SOURCES            _multi_source_expansion_op_bindings.cpp
)

target_include_directories(multi_source_expansion_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
