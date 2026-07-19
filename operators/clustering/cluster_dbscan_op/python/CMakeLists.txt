# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   cluster_dbscan_op
    CLASS_NAME         ClusterDBSCANOpF32
    SOURCES            _cluster_dbscan_op_bindings.cpp
)

target_include_directories(cluster_dbscan_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
