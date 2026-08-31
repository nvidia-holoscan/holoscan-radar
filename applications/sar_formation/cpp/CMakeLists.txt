# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# compute() is a .cpp so nvcc does not parse Holoscan (that was
# the slow sar_image_lut.cu.o). The .cu is |z|, box-average,
# GPU percentile, and the uint8 LUT.
add_executable(sar_formation
    sar_formation.cpp
    sar_image_lut.cpp
    sar_image_lut.cu
    timeline.cpp)

# yaml-cpp ships with the Holoscan SDK (found transitively via
# find_package(holoscan)); the app parses its sidecar JSON (valid YAML)
# and the app-owned config sections with it.
find_package(yaml-cpp REQUIRED)

target_link_libraries(sar_formation PRIVATE
    holoscan::cphd_reader_op
    holoscan::sar_range_compression_op
    holoscan::backprojection_op
    holoscan::tensor_dump_op
    holoscan::radar_app
    holoscan::radar_common
    holoscan::aperture_manager_op
    holoscan::core
    holoscan::ops::holoviz
    holoscan::viz
    yaml-cpp::yaml-cpp
    CUDA::cudart
    ${CMAKE_DL_LIBS})

target_include_directories(sar_formation PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)

# Stage the default config next to the executable so it is found without
# a CLI argument (the run command's workdir is the app bin directory).
configure_file(mode01.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode01.yaml COPYONLY)
configure_file(mode01_smoke.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode01_smoke.yaml COPYONLY)
configure_file(mode02.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode02.yaml COPYONLY)
configure_file(mode03.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode03.yaml COPYONLY)
configure_file(mode04.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode04.yaml COPYONLY)
configure_file(mode05.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/mode05.yaml COPYONLY)
