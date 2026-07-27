# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   root_music_estimator_op
    CLASS_NAME         RootMUSICEstimatorOpF32
    SOURCES            _root_music_estimator_op_bindings.cpp
)

target_include_directories(root_music_estimator_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
