# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reusable harness headers (utils/) plus thin per-operator consumers. Shared
# code lives in utils/ as an INTERFACE library; only the scenario matrix is
# operator-specific.
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

add_executable(perf_feature_assembly perf_feature_assembly.cpp)
target_link_libraries(perf_feature_assembly PRIVATE
    radar_perf_utils
    holoscan::feature_assembly_op)

add_executable(perf_fmcw_range_processor perf_fmcw_range_processor.cpp)
target_link_libraries(perf_fmcw_range_processor PRIVATE
    radar_perf_utils
    holoscan::fmcw_range_processor_op)

add_executable(perf_fmcw_doppler_processor perf_fmcw_doppler_processor.cpp)
target_link_libraries(perf_fmcw_doppler_processor PRIVATE
    radar_perf_utils
    holoscan::fmcw_doppler_processor_op)

add_executable(perf_clutter_filter perf_clutter_filter.cpp)
target_link_libraries(perf_clutter_filter PRIVATE
    radar_perf_utils
    holoscan::clutter_filter_op)

add_executable(perf_fmcw_range_doppler_processor
    perf_fmcw_range_doppler_processor.cpp)
target_link_libraries(perf_fmcw_range_doppler_processor PRIVATE
    radar_perf_utils
    holoscan::fmcw_range_doppler_processor_op)

add_executable(perf_cluster_dbscan perf_cluster_dbscan.cpp)
target_link_libraries(perf_cluster_dbscan PRIVATE
    radar_perf_utils
    holoscan::cluster_dbscan_op)

add_executable(perf_snapshot_gather perf_snapshot_gather.cpp)
target_link_libraries(perf_snapshot_gather PRIVATE
    radar_perf_utils
    holoscan::snapshot_gather_op)

add_executable(perf_beamscan_estimator perf_beamscan_estimator.cpp)
target_link_libraries(perf_beamscan_estimator PRIVATE
    radar_perf_utils
    holoscan::beamscan_estimator_op)

add_executable(perf_phase_shift_beamformer perf_phase_shift_beamformer.cpp)
target_link_libraries(perf_phase_shift_beamformer PRIVATE
    radar_perf_utils
    holoscan::phase_shift_beamformer_op)

add_executable(perf_mimo_deinterleave perf_mimo_deinterleave.cpp)
target_link_libraries(perf_mimo_deinterleave PRIVATE
    radar_perf_utils
    holoscan::mimo_deinterleave_op)

add_executable(perf_virtual_array_assembly perf_virtual_array_assembly.cpp)
target_link_libraries(perf_virtual_array_assembly PRIVATE
    radar_perf_utils
    holoscan::virtual_array_assembly_op)

add_executable(perf_mimo_coherent_compensation perf_mimo_coherent_compensation.cpp)
target_link_libraries(perf_mimo_coherent_compensation PRIVATE
    radar_perf_utils
    holoscan::mimo_coherent_compensation_op)

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

    add_test(NAME perf_feature_assembly_smoke
        COMMAND perf_feature_assembly --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_feature_assembly_smoke.json)
    set_tests_properties(perf_feature_assembly_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_fmcw_range_processor_smoke
        COMMAND perf_fmcw_range_processor --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_fmcw_range_processor_smoke.json)
    set_tests_properties(perf_fmcw_range_processor_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_fmcw_doppler_processor_smoke
        COMMAND perf_fmcw_doppler_processor --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_fmcw_doppler_processor_smoke.json)
    set_tests_properties(perf_fmcw_doppler_processor_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_clutter_filter_smoke
        COMMAND perf_clutter_filter --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_clutter_filter_smoke.json)
    set_tests_properties(perf_clutter_filter_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_fmcw_range_doppler_processor_smoke
        COMMAND perf_fmcw_range_doppler_processor --sizes xs --warmup 1 --frames 2
            --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_fmcw_range_doppler_processor_smoke.json)
    set_tests_properties(perf_fmcw_range_doppler_processor_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_cluster_dbscan_smoke
        COMMAND perf_cluster_dbscan --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_cluster_dbscan_smoke.json)
    set_tests_properties(perf_cluster_dbscan_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_snapshot_gather_smoke
        COMMAND perf_snapshot_gather --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_snapshot_gather_smoke.json)
    set_tests_properties(perf_snapshot_gather_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_beamscan_estimator_smoke
        COMMAND perf_beamscan_estimator --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_beamscan_estimator_smoke.json)
    set_tests_properties(perf_beamscan_estimator_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_phase_shift_beamformer_smoke
        COMMAND perf_phase_shift_beamformer --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_phase_shift_beamformer_smoke.json)
    set_tests_properties(perf_phase_shift_beamformer_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_mimo_deinterleave_smoke
        COMMAND perf_mimo_deinterleave --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_mimo_deinterleave_smoke.json)
    set_tests_properties(perf_mimo_deinterleave_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_virtual_array_assembly_smoke
        COMMAND perf_virtual_array_assembly --sizes xs --warmup 1 --frames 2
            --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_virtual_array_assembly_smoke.json)
    set_tests_properties(perf_virtual_array_assembly_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_mimo_coherent_compensation_smoke
        COMMAND perf_mimo_coherent_compensation --sizes xs --warmup 1 --frames 2
            --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_mimo_coherent_compensation_smoke.json)
    set_tests_properties(perf_mimo_coherent_compensation_smoke PROPERTIES
        LABELS "performance")
endif()
