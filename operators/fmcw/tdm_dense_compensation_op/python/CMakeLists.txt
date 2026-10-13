# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include(pybind11_add_holohub_module)
pybind11_add_holohub_module(CPP_CMAKE_TARGET tdm_dense_compensation_op CLASS_NAME TDMDenseCompensationOpF32 SOURCES _tdm_dense_compensation_op_bindings.cpp)
target_include_directories(tdm_dense_compensation_op_python PRIVATE ${PROJECT_SOURCE_DIR}/operators ${PROJECT_SOURCE_DIR}/operators/common)
