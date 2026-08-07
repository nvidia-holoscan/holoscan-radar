# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   mimo_coherent_compensation_op
    CLASS_NAME         MIMOCoherentCompensationOpF32
    SOURCES            _mimo_coherent_compensation_op_bindings.cpp
)

target_include_directories(mimo_coherent_compensation_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
