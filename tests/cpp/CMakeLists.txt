# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest REQUIRED)
include(GoogleTest)

# Shared test utilities: forward bin model and radar-specific Tensor fixtures.
# Single-operator tests use the SDK's native holoscan::testing fixture. The
# analytic point-target simulator lives in radar_common/sim (reached via
# holoscan::radar_common).
add_library(radar_test_utils INTERFACE)
target_include_directories(radar_test_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radar_test_utils INTERFACE
    holoscan::core
    holoscan::testing
    holoscan::radar_common
    holoscan::radar_schemas
    CUDA::cudart
    GTest::gtest)

# One executable per operator (plus the shared-library suite) so the
# parallel implementation phase owns disjoint test files. NAME is a
# family-relative source path without the extension (e.g. detection/test_cfar);
# the target name is the flat basename so `ctest -R <name>` stays stable
# regardless of where the source lives in the family tree.
# Optional EXT overrides the source extension (default .cpp); use .cu when
# the suite includes device stage kernels (e.g. FeatureAssembly).
function(radar_add_test NAME)
    cmake_parse_arguments(_RAT "" "EXT;LABELS" "" ${ARGN})
    if(NOT _RAT_EXT)
        set(_RAT_EXT ".cpp")
    endif()
    if(NOT _RAT_LABELS)
        set(_RAT_LABELS "unit")
    endif()
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}${_RAT_EXT})
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${_RAT_UNPARSED_ARGUMENTS})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "${_RAT_LABELS}")
endfunction()

radar_add_test(common/test_window)
radar_add_test(common/test_uniform_grid)
radar_add_test(common/test_steering_weights_device
    holoscan::radar_common_steering)
radar_add_test(common/test_spatial_smoothing)
radar_add_test(common/test_subarray_selection)
radar_add_test(common/test_source_count)
radar_add_test(common/test_angle_convention)
radar_add_test(common/test_descriptors)
radar_add_test(common/test_ddm_code_generators
    holoscan::radar_common_mimo_compensation)
radar_add_test(common/test_live_override_payloads)
radar_add_test(common/test_wire_descriptors holoscan::radar_schemas)
# The performance harness parses its YAML recipes with yaml-cpp, which HSDK 5
# no longer re-exports.
find_package(yaml-cpp REQUIRED)
radar_add_test(common/test_sdk holoscan::config_io)
radar_add_test(common/test_emit_required)
radar_add_test(common/test_max_tensor_extents holoscan::config_io)
radar_add_test(utils/test_perf_profiles yaml-cpp)
radar_add_test(utils/test_perf_runner)
radar_add_test(utils/test_perf_yaml_standard yaml-cpp holoscan::doppler_notch_op
    holoscan::fmcw_range_processor_op holoscan::subarray_selection_op)
target_include_directories(test_perf_yaml_standard PRIVATE
    ${PROJECT_SOURCE_DIR}/performance/harness/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_include_directories(test_perf_runner PRIVATE ${PROJECT_SOURCE_DIR}/performance/harness/cpp)
target_include_directories(test_perf_profiles PRIVATE
    ${PROJECT_SOURCE_DIR}/performance/harness/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_perf_profiles PRIVATE
    HOLOSCAN_RADAR_PERF_OPERATOR_ROOT="${PROJECT_SOURCE_DIR}/performance/operators")
target_compile_definitions(test_perf_yaml_standard PRIVATE
    HOLOSCAN_RADAR_PERF_OPERATOR_ROOT="${PROJECT_SOURCE_DIR}/performance/operators")
radar_add_test(common/test_sim)
radar_add_test(sar/test_phase_correction holoscan::phase_correction_op)
radar_add_test(sar/test_aperture_manager holoscan::aperture_manager_op)
radar_add_test(sar/test_backprojection holoscan::backprojection_op)
radar_add_test(sar/test_sar_range_compression holoscan::sar_range_compression_op)
radar_add_test(sar/test_autofocus holoscan::autofocus_op)
radar_add_test(io/test_tensor_dump holoscan::tensor_dump_op)
radar_add_test(io/test_cphd_reader holoscan::cphd_reader_op)
add_executable(generate_cphd_fixture io/generate_cphd_fixture.cpp)
target_include_directories(generate_cphd_fixture PRIVATE ${PROJECT_SOURCE_DIR}/tests/cpp)
radar_add_test(common/test_sim_gpu holoscan::radar_common_simulator)
radar_add_test(common/test_tensor)
radar_add_test(common/test_virtual_array)
radar_add_test(utils/test_native_operator_testing
    holoscan::clutter_filter_op)
radar_add_test(utils/test_fixture_completion
    holoscan::clutter_filter_op)
radar_add_test(fmcw/test_clutter_filter
    holoscan::clutter_filter_op)
radar_add_test(fmcw/test_mimo_deinterleave
    holoscan::mimo_deinterleave_op)
radar_add_test(fmcw/test_tdm_dense_compensation
    holoscan::tdm_dense_compensation_op)
radar_add_test(fmcw/test_ddm_dense_demultiplex
    holoscan::ddm_dense_demultiplex_op
    holoscan::fmcw_doppler_processor_op)
radar_add_test(fmcw/test_doppler_notch
    holoscan::doppler_notch_op)
radar_add_test(fmcw/test_doppler_notch_pipeline
    holoscan::fmcw_range_processor_op
    holoscan::fmcw_doppler_processor_op
    holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op)
radar_add_test(fmcw/test_ddm_power_fold
    holoscan::ddm_power_fold_op)
radar_add_test(fmcw/test_ddm_sparse_demultiplex
    holoscan::ddm_sparse_demultiplex_op)
radar_add_test(fmcw/test_tdm_sparse_compensation
    holoscan::tdm_sparse_compensation_op)
radar_add_test(fmcw/test_virtual_array_assembly
    holoscan::virtual_array_assembly_op
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering)
radar_add_test(fmcw/test_fmcw_doppler_processor
    holoscan::fmcw_doppler_processor_op)
radar_add_test(fmcw/test_fmcw_range_processor
    holoscan::fmcw_range_processor_op)
radar_add_test(fmcw/test_fmcw_range_doppler_processor
    holoscan::cfar_detector_op
    holoscan::fmcw_range_doppler_processor_op
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op)
radar_add_test(angle_estimation/test_snapshot_gather
    holoscan::snapshot_gather_op)
radar_add_test(angle_estimation/test_beamscan_estimator
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering)
radar_add_test(angle_estimation/test_rank_one_residual_screen
    EXT .cu
    holoscan::rank_one_residual_screen_op)
radar_add_test(angle_estimation/test_phase_shift_beamformer
    holoscan::phase_shift_beamformer_op
    holoscan::radar_common_steering)
radar_add_test(angle_estimation/test_subarray_selection_op
    holoscan::subarray_selection_op
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering
    holoscan::radar_common_spatial_smoothing)
radar_add_test(angle_estimation/test_mvdr_estimator
    holoscan::mvdr_estimator_op
    holoscan::radar_common_steering
    holoscan::radar_common_spatial_smoothing)
radar_add_test(angle_estimation/test_root_music_estimator
    holoscan::root_music_estimator_op
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(angle_estimation/test_root_music_solver EXT .cu
    holoscan::root_music_estimator_op matx::matx CUDA::cusolver)
radar_add_test(angle_estimation/test_esprit_estimator
    holoscan::esprit_estimator_op
    holoscan::subarray_selection_op
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(angle_estimation/test_music_estimator
    holoscan::music_estimator_op
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(clustering/test_cluster_dbscan
    holoscan::cluster_dbscan_op)
radar_add_test(detection/test_detection_reduction
    holoscan::detection_reduction_op)
radar_add_test(detection/test_detection_reduction_custom
    EXT .cu
    holoscan::detection_reduction_op)
radar_add_test(detection/test_feature_assembly
    EXT .cu
    holoscan::feature_assembly_op)
radar_add_test(detection/test_multi_source_expansion
    EXT .cu
    holoscan::multi_source_expansion_op)
radar_add_test(integration/test_selective_aoa_operators
    holoscan::rank_one_residual_screen_op
    holoscan::multi_source_expansion_op
    holoscan::feature_assembly_op)
radar_add_test(integration/test_sar_autofocus_graph
    holoscan::aperture_manager_op
    holoscan::autofocus_op
    holoscan::backprojection_op
    holoscan::phase_correction_op)
# The retain/reform cases drive the autofocus app's own azimuth-to-pulse
# resampler. Reimplementing it here would assert against a copy, so the app
# directory is an include root for this one target.
target_include_directories(test_sar_autofocus_graph PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/sar)
radar_add_test(detection/test_cfar_calibration cfar_detector_op)
radar_add_test(detection/test_cfar_detector
    holoscan::cfar_detector_op
    holoscan::feature_assembly_op)
radar_add_test(detection/test_cfar_detector_alpha_lut
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_os
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_so
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_go
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_statistical
    LABELS statistical
    holoscan::cfar_detector_op)

radar_add_test(app/test_gpu_placement holoscan::radar_app)
radar_add_test(app/test_fmcw_config_loader holoscan::params_io holoscan::radar_app)

radar_add_test(app/test_fmcw_simo_config
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::phase_shift_beamformer_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_simo_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_simo_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

radar_add_test(app/test_fmcw_siso_config
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_siso_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/fmcw/siso/cpp
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_siso_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

radar_add_test(app/test_fmcw_mimo_config
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op
    holoscan::virtual_array_assembly_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_mimo_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_mimo_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

radar_add_test(app/test_adaptive_aoa_config
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::cfar_detector_op
    holoscan::beamscan_estimator_op
    holoscan::rank_one_residual_screen_op
    holoscan::music_estimator_op
    holoscan::root_music_estimator_op
    holoscan::esprit_estimator_op
    holoscan::mvdr_estimator_op
    holoscan::feature_assembly_op
    holoscan::multi_source_expansion_op)
target_include_directories(test_adaptive_aoa_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/applications/fmcw/adaptive_aoa/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_adaptive_aoa_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")


radar_add_test(app/test_multi_profile_source
    holoscan::params_io
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::cfar_detector_op
    holoscan::beamscan_estimator_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_multi_profile_source PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/applications/fmcw/multi_waveform/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_multi_profile_source PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")


radar_add_test(app/test_fmcw_simo_report
    holoscan::radar_app)
target_include_directories(test_fmcw_simo_report PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)

radar_add_test(app/test_report_shapes holoscan::radar_app)
radar_add_test(app/test_sar_diagnostics holoscan::radar_app)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    radar_add_test(app/test_fmcw_dashboard holoscan::fmcw_dashboard)
endif()

# All enabled applications get a real executable smoke. Window tests explicitly
# choose their presenter and use private preference caches in the shared probe.
if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    find_program(RADAR_TEST_PYTHON python3 REQUIRED)
    if(HOLOSCAN_RADAR_HAS_X11)
        find_program(RADAR_XVFB_RUN xvfb-run REQUIRED)
    endif()
endif()

function(radar_add_fmcw_workflow_checks application preset smoke_name)
    if(NOT TARGET ${application})
        return()
    endif()
    set(config_dir "${PROJECT_SOURCE_DIR}/applications/fmcw/${application}/config")
    add_test(NAME ${smoke_name}
        COMMAND $<TARGET_FILE:${application}> "${config_dir}/${preset}_smoke.yaml")
    set_tests_properties(${smoke_name} PROPERTIES LABELS "application;smoke" TIMEOUT 60)

    if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
        set(headless_config "${config_dir}/${preset}_smoke.yaml")
        if(EXISTS "${config_dir}/${preset}_viz_smoke.yaml")
            set(headless_config "${config_dir}/${preset}_viz_smoke.yaml")
        endif()
        add_test(NAME FmcwHeadless_${preset}_cpp
            COMMAND ${RADAR_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/check_headless.py
                $<TARGET_FILE:${application}> "${headless_config}")
        set_tests_properties(FmcwHeadless_${preset}_cpp PROPERTIES
            LABELS "application;smoke;integration;visualization" TIMEOUT 60 RESOURCE_LOCK fmcw_gui)
    endif()

    if(NOT HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ OR NOT HOLOSCAN_RADAR_HAS_X11)
        return()
    endif()
    # Some GUI demonstrations reserve more Tensor capacity than CI needs.
    set(window_config "${config_dir}/${preset}.yaml")
    if(EXISTS "${config_dir}/${preset}_viz_smoke.yaml")
        set(window_config "${config_dir}/${preset}_viz_smoke.yaml")
    endif()
    foreach(language IN ITEMS cpp python)
        if(language STREQUAL "python" AND NOT HOLOSCAN_RADAR_BUILD_PYTHON)
            continue()
        endif()
        set(probe_args --application ${application} --presenter x11)
        set(xvfb_args -a)
        set(name "FmcwX11_${preset}_${language}")
        if(language STREQUAL "python")
            list(APPEND probe_args --python)
        endif()
        if(preset STREQUAL "siso")
            # Retain detailed SISO checks, including Python's TCP display path.
            set(name "FmcwX11_cpp")
            if(language STREQUAL "python")
                set(name "FmcwX11_pythonTcp")
                list(APPEND probe_args --tcp)
                list(APPEND xvfb_args -l -s "-screen 0 1280x1024x24 -listen tcp")
            endif()
        else()
            list(APPEND probe_args --smoke)
        endif()
        add_test(NAME ${name}
            COMMAND ${RADAR_XVFB_RUN} ${xvfb_args} ${RADAR_TEST_PYTHON}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_fmcw_x11.py $<TARGET_FILE:${application}>
                "${window_config}" ${probe_args})
        set_tests_properties(${name} PROPERTIES
            LABELS "application;smoke;integration;visualization"
            RESOURCE_LOCK fmcw_gui TIMEOUT 60)
        if(language STREQUAL "python")
            set_tests_properties(${name} PROPERTIES ENVIRONMENT
                "PYTHONPATH=${HOLOSCAN_RADAR_PYTHON_ROOT}:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}")
        endif()

        set(name "FmcwNative_${preset}_${language}")
        set(native_args --application ${application} --presenter native)
        if(NOT preset STREQUAL "siso")
            list(APPEND native_args --smoke)
        endif()
        if(language STREQUAL "python")
            list(APPEND native_args --python)
        endif()
        add_test(NAME ${name}
            COMMAND ${RADAR_XVFB_RUN} -a ${RADAR_TEST_PYTHON}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_fmcw_x11.py $<TARGET_FILE:${application}>
                "${window_config}" ${native_args})
        set_tests_properties(${name} PROPERTIES
            LABELS "application;smoke;integration;visualization"
            RESOURCE_LOCK fmcw_gui TIMEOUT 60)
        if(language STREQUAL "python")
            set_tests_properties(${name} PROPERTIES ENVIRONMENT
                "PYTHONPATH=${HOLOSCAN_RADAR_PYTHON_ROOT}:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}")
        endif()
    endforeach()
endfunction()

radar_add_fmcw_workflow_checks(siso siso SisoSmoke)
radar_add_fmcw_workflow_checks(simo simo SimoSmoke)
radar_add_fmcw_workflow_checks(dense_beamforming dense_beamforming DenseBeamformingSmoke)
radar_add_fmcw_workflow_checks(mimo_tdm mimo_tdm MimoTdmSmoke)
radar_add_fmcw_workflow_checks(mimo_ddm mimo_ddm_linear MimoDdmLinearSmoke)
radar_add_fmcw_workflow_checks(mimo_ddm mimo_ddm_nonlinear MimoDdmNonlinearSmoke)
radar_add_fmcw_workflow_checks(adaptive_aoa adaptive_aoa AdaptiveAoASmoke)
radar_add_fmcw_workflow_checks(multi_waveform multi_waveform MultiWaveformSmoke)

set(SAR_SMOKE_DIR "${PROJECT_BINARY_DIR}/tests/cpp")
set(SAR_APP_NAMES spotlight video_sar tiled_strip continuous_strip independent_images)
set(SAR_COUNTS 1 5 5 5 2)
set(SAR_DIMS "5x5" "5x5" "15x6" "15x6" "5x5")
# `autofocus` is not in SAR_APP_NAMES -- it drives its own smoke block below
# rather than the generic count/dims loop -- but it consumes the same fixture,
# so an autofocus-only build still needs the CPHD generated.
if(TARGET spotlight OR TARGET video_sar OR TARGET tiled_strip OR
        TARGET continuous_strip OR TARGET independent_images OR TARGET autofocus)
    add_test(NAME SarFixture
        COMMAND $<TARGET_FILE:generate_cphd_fixture>
            "${SAR_SMOKE_DIR}/generated_sar_smoke.cphd")
    set_tests_properties(SarFixture PROPERTIES FIXTURES_SETUP sar_cphd)
endif()
# Derived, not a literal 0..4: the three lists above must stay the same
# length, and a hardcoded bound silently drops an app appended to them.
list(LENGTH SAR_APP_NAMES sar_app_count)
math(EXPR sar_app_last "${sar_app_count} - 1")
foreach(index RANGE 0 ${sar_app_last})
    list(GET SAR_APP_NAMES ${index} name)
    list(GET SAR_COUNTS ${index} count)
    list(GET SAR_DIMS ${index} dims)
    if(TARGET ${name})
        set(config_dir "${PROJECT_BINARY_DIR}/applications/sar/${name}/cpp")
        foreach(variant smoke viz)
            if(variant STREQUAL "viz")
                set(config "${config_dir}/${name}_viz_generated_smoke.yaml")
            else()
                set(config "${config_dir}/${name}_generated_smoke.yaml")
            endif()
            add_test(NAME Sar_${name}_${variant}
                COMMAND ${CMAKE_COMMAND}
                    -DSAR_EXECUTABLE=$<TARGET_FILE:${name}>
                    -DSAR_CONFIG=${config}
                    -DSAR_COUNT=${count}
                    -DSAR_DIMS=${dims}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
            set_tests_properties(Sar_${name}_${variant} PROPERTIES
                FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
        endforeach()
        math(EXPR next_index "(${index} + 1) % 5")
        list(GET SAR_APP_NAMES ${next_index} wrong_name)
        set(wrong_config
            "${PROJECT_BINARY_DIR}/applications/sar/${wrong_name}/cpp/${wrong_name}_generated_smoke.yaml")
        if(TARGET ${wrong_name})
            add_test(NAME Sar_${name}_rejects_other_topology
                COMMAND ${CMAKE_COMMAND}
                    -DSAR_EXECUTABLE=$<TARGET_FILE:${name}>
                    -DSAR_CONFIG=${wrong_config}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_reject.cmake)
            set_tests_properties(Sar_${name}_rejects_other_topology PROPERTIES
                FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
        endif()
    endif()
endforeach()

if(TARGET autofocus)
    # Figure 2: retained aperture replayed once, auxiliary conjugate estimate,
    # corrected product. One image, the product lattice.
    set(af_dir "${PROJECT_BINARY_DIR}/applications/sar/autofocus/cpp")
    add_test(NAME Sar_autofocus_smoke
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:autofocus>
            -DSAR_CONFIG=${af_dir}/autofocus_generated_smoke.yaml
            -DSAR_COUNT=1
            -DSAR_DIMS=5x5
            "-DSAR_REQUIRE_REGEX=sharpness .*x; [0-9]+ iteration.s. accepted; budget 8"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
    set_tests_properties(Sar_autofocus_smoke PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")

    # Test the actual saved complex samples through the application in both
    # supported sample precisions. The oracle must undo a nontrivial injection.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    foreach(precision single double)
        add_test(NAME Sar_autofocus_pixels_${precision}
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_autofocus_pixels.py
                --binary $<TARGET_FILE:autofocus>
                --config ${af_dir}/autofocus_generated_smoke.yaml
                --precision ${precision})
        set_tests_properties(Sar_autofocus_pixels_${precision} PROPERTIES
            FIXTURES_REQUIRED sar_cphd LABELS "application;integration" TIMEOUT 180)
    endforeach()
    add_test(NAME AutofocusSweepReports
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${PROJECT_SOURCE_DIR}/tools/sar_autofocus)
    set_tests_properties(AutofocusSweepReports PROPERTIES LABELS "python;unit")

    # The same graph with a known look-axis error injected. The 5-pulse fixture
    # is far too small for PGA to recover anything, so these pin the injection,
    # the sign convention and the apply path, not the recovery. The scoring is
    # reported, not enforced: AutofocusScoreOp only warns past its threshold and
    # does not affect the exit code, so what each variant actually asserts is
    # the required line below.
    # Each variant must state a number or a verdict of its own, or it asserts
    # nothing the plain smoke test above does not. The regexes match digits so
    # that a "nan" or "inf" fails instead of counting as output.
    #
    # Every required line is chosen to be neutral about estimate quality. A
    # regex that only passes while the operator declines to act -- the polar
    # variant used to require `autofocus REFUSED` -- is a tripwire pointing
    # the wrong way: a genuine improvement turns it red. `lie` additionally
    # pins the sign convention, which is what the comment above claims these
    # variants are for and what the bare RMS number did not check.
    set(AF_DIAG_NAMES lie oracle polar)
    set(AF_DIAG_REQUIRE
        "AutofocusScoreOp: affine-removed RMS [0-9]+\\.[0-9]+ rad.*sign .4pi f d / c"
        "OraclePhaseOp: published [0-9]+ pulses  [0-9]+\\.[0-9]+ rad RMS"
        "AuxPointsSourceOp: staged [0-9]+x[0-9]+ range-bearing points")
    list(LENGTH AF_DIAG_NAMES af_diag_count)
    math(EXPR af_diag_last "${af_diag_count} - 1")
    foreach(af_index RANGE 0 ${af_diag_last})
        list(GET AF_DIAG_NAMES ${af_index} af_diag)
        list(GET AF_DIAG_REQUIRE ${af_index} af_require)
        add_test(NAME Sar_autofocus_${af_diag}
            COMMAND ${CMAKE_COMMAND}
                -DSAR_EXECUTABLE=$<TARGET_FILE:autofocus>
                -DSAR_CONFIG=${af_dir}/autofocus_generated_smoke_${af_diag}.yaml
                -DSAR_COUNT=1
                -DSAR_DIMS=5x5
                "-DSAR_REQUIRE_REGEX=${af_require}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
        set_tests_properties(Sar_autofocus_${af_diag} PROPERTIES
            FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
    endforeach()
endif()

if(TARGET video_sar)
    set(video_dir "${PROJECT_BINARY_DIR}/applications/sar/video_sar/cpp")
    add_test(NAME Sar_video_sar_retention
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:video_sar>
            -DSAR_CONFIG=${video_dir}/video_sar_retention_generated_smoke.yaml
            -DSAR_COUNT=5
            -DSAR_DIMS=5x5
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
    set_tests_properties(Sar_video_sar_retention PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
endif()

if(TARGET tiled_strip)
    set(tiled_dir "${PROJECT_BINARY_DIR}/applications/sar/tiled_strip/cpp")
    add_test(NAME Sar_tiled_strip_seam_viz
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:tiled_strip>
            -DSAR_CONFIG=${tiled_dir}/tiled_strip_seam_viz_generated_smoke.yaml
            -DSAR_COUNT=2
            -DSAR_DIMS=11x6
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
    set_tests_properties(Sar_tiled_strip_seam_viz PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
endif()

if(TARGET independent_images)
    set(independent_dir "${PROJECT_BINARY_DIR}/applications/sar/independent_images/cpp")
    add_test(NAME Sar_independent_images_varying_viz
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:independent_images>
            -DSAR_CONFIG=${independent_dir}/independent_images_varying_viz_generated_smoke.yaml
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_varying_viz.cmake)
    set_tests_properties(Sar_independent_images_varying_viz PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
endif()

if(TARGET spotlight)
    set(spotlight_dir "${PROJECT_BINARY_DIR}/applications/sar/spotlight/cpp")
    add_test(NAME Sar_spotlight_dump
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:spotlight>
            -DSAR_CONFIG=${spotlight_dir}/spotlight_dump_generated_smoke.yaml
            -DSAR_DUMP_PATH=${spotlight_dir}/spotlight_dump_0.npy
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_dump.cmake)
    set_tests_properties(Sar_spotlight_dump PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
endif()

# Shared SAR presentation: native cases require a presentable CUDA/Vulkan display.
foreach(application IN ITEMS spotlight video_sar tiled_strip continuous_strip independent_images autofocus)
    if(NOT TARGET ${application} OR NOT HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
        continue()
    endif()
    add_test(NAME SarHeadless_${application}_cpp
        COMMAND ${RADAR_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/check_headless.py
            $<TARGET_FILE:${application}>
            ${PROJECT_BINARY_DIR}/applications/sar/${application}/cpp/${application}_generated_smoke.yaml)
    set_tests_properties(SarHeadless_${application}_cpp PROPERTIES
        LABELS "application;smoke;integration;visualization" TIMEOUT 60
        RESOURCE_LOCK fmcw_gui FIXTURES_REQUIRED sar_cphd)
    if(NOT HOLOSCAN_RADAR_HAS_X11)
        continue()
    endif()
    foreach(language IN ITEMS cpp python)
        if(language STREQUAL "python" AND NOT HOLOSCAN_RADAR_BUILD_PYTHON)
            continue()
        endif()
        foreach(presenter IN ITEMS x11 native)
            set(gui_args --application ${application} --presenter ${presenter})
            if(language STREQUAL "python")
                list(APPEND gui_args --python)
            endif()
            if(language STREQUAL "python")
                list(APPEND gui_args --close timeline --resize-mode fit --escape)
            endif()
            set(gui_name "SarWindow_${application}_${language}_${presenter}")
            add_test(NAME ${gui_name}
                COMMAND ${RADAR_XVFB_RUN} -a ${RADAR_TEST_PYTHON}
                    ${PROJECT_SOURCE_DIR}/tests/gui/check_sar_windows.py
                    $<TARGET_FILE:${application}>
                    ${PROJECT_BINARY_DIR}/applications/sar/${application}/cpp/${application}_generated_smoke.yaml
                    ${gui_args})
            set_tests_properties(${gui_name} PROPERTIES
                LABELS "application;smoke;integration;visualization"
                RESOURCE_LOCK fmcw_gui TIMEOUT 60
                FIXTURES_REQUIRED sar_cphd
                ENVIRONMENT "PYTHONPATH=${HOLOSCAN_RADAR_PYTHON_ROOT}:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}")
        endforeach()
    endforeach()
endforeach()

# Keep the explicit legacy layout covered alongside the new docked default.
if(TARGET spotlight AND HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ AND HOLOSCAN_RADAR_HAS_X11)
    foreach(language IN ITEMS cpp python)
        if(language STREQUAL "python" AND NOT HOLOSCAN_RADAR_BUILD_PYTHON)
            continue()
        endif()
        set(separate_args --application spotlight --presenter x11 --timeline-placement window)
        if(language STREQUAL "python")
            list(APPEND separate_args --python --close timeline --escape)
        endif()
        add_test(NAME SarWindow_spotlight_${language}_x11_separate
            COMMAND ${RADAR_XVFB_RUN} -a ${RADAR_TEST_PYTHON}
                ${PROJECT_SOURCE_DIR}/tests/gui/check_sar_windows.py $<TARGET_FILE:spotlight>
                ${PROJECT_BINARY_DIR}/applications/sar/spotlight/cpp/spotlight_generated_smoke.yaml
                ${separate_args})
        set_tests_properties(SarWindow_spotlight_${language}_x11_separate PROPERTIES
            LABELS "application;smoke;integration;visualization"
            RESOURCE_LOCK fmcw_gui TIMEOUT 60 FIXTURES_REQUIRED sar_cphd
            ENVIRONMENT "PYTHONPATH=${HOLOSCAN_RADAR_PYTHON_ROOT}:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}")
    endforeach()
endif()
