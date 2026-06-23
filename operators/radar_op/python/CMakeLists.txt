# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pybind11_add_holohub_module:
#   - Fetches pybind11 at the version pinned to Holoscan SDK (so the same
#     FetchContent_Declare entry is shared if this module is consumed as a
#     dependency in HoloHub or another module).
#   - Auto-links holoscan::core and holoscan::pybind11 (ABI-aligned with HSDK 3.3+).
#   - Generates ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/radar_op/__init__.py
#     with helpful import-error diagnostics for ABI mismatches.
include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET   radar_op
    CLASS_NAME         RadarOp
    SOURCES            _radar_op_bindings.cpp
)

target_include_directories(radar_op_python PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
