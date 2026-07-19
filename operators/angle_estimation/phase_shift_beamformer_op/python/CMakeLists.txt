# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   phase_shift_beamformer_op
    CLASS_NAME         PhaseShiftBeamformerOpF32
    SOURCES            _phase_shift_beamformer_op_bindings.cpp
)

target_include_directories(phase_shift_beamformer_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators
    ${PROJECT_SOURCE_DIR}/operators/common
)
