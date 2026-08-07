# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   beamscan_estimator_op
    CLASS_NAME         BeamscanEstimatorOpF32
    SOURCES            _beamscan_estimator_op_bindings.cpp
)

target_include_directories(beamscan_estimator_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
