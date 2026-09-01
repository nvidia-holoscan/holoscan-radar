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

add_executable(perf_fmcw_split_vs_fused perf_fmcw_split_vs_fused.cpp)
target_link_libraries(perf_fmcw_split_vs_fused PRIVATE
    radar_perf_utils
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
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

add_executable(perf_mvdr_estimator perf_mvdr_estimator.cpp)
target_link_libraries(perf_mvdr_estimator PRIVATE
    radar_perf_utils
    holoscan::mvdr_estimator_op)

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

# The reader characterizes against a file, so it needs to know where the
# generated fixtures land. Same directory the C++ tests use, since the same
# CTest setup fixture produces them; HOLOSCAN_RADAR_PERF_CPHD overrides it at
# run time for measuring against a real collection.
add_executable(perf_cphd_reader perf_cphd_reader.cpp)
target_link_libraries(perf_cphd_reader PRIVATE
    radar_perf_utils
    holoscan::cphd_reader_op)
target_compile_definitions(perf_cphd_reader PRIVATE
    SAR_FIXTURE_DIR="${PROJECT_BINARY_DIR}/tests/cpp/sar_fixtures")

add_executable(perf_sar_range_compression perf_sar_range_compression.cpp)
target_link_libraries(perf_sar_range_compression PRIVATE
    radar_perf_utils
    holoscan::sar_range_compression_op)

add_executable(perf_backprojection perf_backprojection.cpp)
target_link_libraries(perf_backprojection PRIVATE
    radar_perf_utils
    holoscan::backprojection_op)

add_executable(perf_phase_correction perf_phase_correction.cpp)
target_link_libraries(perf_phase_correction PRIVATE
    radar_perf_utils
    holoscan::phase_correction_op)

add_executable(perf_tensor_dump perf_tensor_dump.cpp)
target_link_libraries(perf_tensor_dump PRIVATE
    radar_perf_utils
    holoscan::tensor_dump_op)

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

    add_test(NAME perf_fmcw_split_vs_fused_smoke
        COMMAND perf_fmcw_split_vs_fused --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_fmcw_split_vs_fused_smoke.json)
    set_tests_properties(perf_fmcw_split_vs_fused_smoke PROPERTIES
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

    add_test(NAME perf_mvdr_estimator_smoke
        COMMAND perf_mvdr_estimator --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_mvdr_estimator_smoke.json)
    set_tests_properties(perf_mvdr_estimator_smoke PROPERTIES
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

    # The only smoke here that needs a data file, hence the fixture
    # dependency the others do not carry.
    add_test(NAME perf_cphd_reader_smoke
        COMMAND perf_cphd_reader --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_cphd_reader_smoke.json)
    set_tests_properties(perf_cphd_reader_smoke PROPERTIES
        LABELS "performance"
        FIXTURES_REQUIRED sar_fixtures)

    add_test(NAME perf_sar_range_compression_smoke
        COMMAND perf_sar_range_compression --sizes xs --warmup 1 --frames 2
            --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_sar_range_compression_smoke.json)
    set_tests_properties(perf_sar_range_compression_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_backprojection_smoke
        COMMAND perf_backprojection --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_backprojection_smoke.json)
    set_tests_properties(perf_backprojection_smoke PROPERTIES
        LABELS "performance")

    add_test(NAME perf_phase_correction_smoke
        COMMAND perf_phase_correction --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_phase_correction_smoke.json)
    set_tests_properties(perf_phase_correction_smoke PROPERTIES
        LABELS "performance")

    # Writes its dumps under the build tree rather than /tmp so a CI runner
    # with a small tmpfs is not where the harness discovers its size limits.
    add_test(NAME perf_tensor_dump_smoke
        COMMAND perf_tensor_dump --sizes xs --warmup 1 --frames 2 --quiet
            --output ${CMAKE_CURRENT_BINARY_DIR}/perf_tensor_dump_smoke.json)
    set_tests_properties(perf_tensor_dump_smoke PROPERTIES
        LABELS "performance"
        ENVIRONMENT
            "HOLOSCAN_RADAR_PERF_DUMP_DIR=${CMAKE_CURRENT_BINARY_DIR}")
endif()
