# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   snapshot_gather_op
    CLASS_NAME         SnapshotGatherOpF32
    SOURCES            _snapshot_gather_op_bindings.cpp
)

target_include_directories(snapshot_gather_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
