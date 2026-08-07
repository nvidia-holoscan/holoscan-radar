# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   fmcw_range_doppler_processor_op
    CLASS_NAME         FMCWRangeDopplerProcessorOpF32
    SOURCES            _fmcw_range_doppler_processor_op_bindings.cpp
)

target_include_directories(fmcw_range_doppler_processor_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
