# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   virtual_array_assembly_op
    CLASS_NAME         VirtualArrayAssemblyOpF32
    SOURCES            _virtual_array_assembly_op_bindings.cpp
)

target_include_directories(virtual_array_assembly_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
