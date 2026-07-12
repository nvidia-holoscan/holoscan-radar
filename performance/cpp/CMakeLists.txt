# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reusable harness headers (utils/) plus one thin per-operator consumer. When a
# second consumer lands, shared code already lives in utils/ as an INTERFACE
# library; only the scenario matrix is operator-specific.
add_library(radar_perf_utils INTERFACE)
target_include_directories(radar_perf_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/operators)
target_link_libraries(radar_perf_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart)

add_executable(perf_cfar_detector perf_cfar_detector.cpp)
target_link_libraries(perf_cfar_detector PRIVATE
    radar_perf_utils
    holoscan::cfar_detector_op)

add_executable(perf_detection_reduction perf_detection_reduction.cpp)
target_link_libraries(perf_detection_reduction PRIVATE
    radar_perf_utils
    holoscan::detection_reduction_op)

# Optional smoke: runs a tiny scenario subset and verifies a report file is
# written. No timing thresholds — CI only checks the tool executes and emits.
if(HOLOSCAN_RADAR_BUILD_TESTING)
    add_test(NAME perf_cfar_detector_smoke
        COMMAND perf_cfar_detector --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_smoke.json)
    set_tests_properties(perf_cfar_detector_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_detection_reduction_smoke
        COMMAND perf_detection_reduction --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_detection_reduction_smoke.json)
    set_tests_properties(perf_detection_reduction_smoke PROPERTIES
        LABELS "performance")
endif()
