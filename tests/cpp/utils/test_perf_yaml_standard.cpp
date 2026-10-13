// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "utils/perf_yaml_standard.hpp"
#include "utils/perf_yaml_pilot.hpp"

namespace perf = holoscan::radar::perf;
namespace hr = holoscan::radar;

TEST(PerfYamlStandard, NativeCapturePreservesConditionalMaskSuppression) {
  using Op = hr::SubarraySelectionOp<float>;
  const std::map<std::string, int64_t> axes{{"items", 2}, {"channels", 4}, {"snapshots", 2}};
  const auto snapshot = perf::generate_recipe_input(
      {"snapshot", {"items", "channels", "snapshots"}, "complex64", "device", "iq_cube", {}}, axes);
  const auto mask = perf::generate_recipe_input(
      {"is_valid", {"items", "snapshots"}, "bool", "device", "valid_mask", {}}, axes);
  hr::MaxTensorExtents extents;
  extents.max_items = 2;
  extents.max_channels = 4;
  extents.max_subarray_channels = 4;
  extents.max_snapshots = 2;
  hr::SubarraySelectionParams params;
  params.array_geometry = hr::ArrayGeometry{};
  params.array_geometry.wavelength = 1.0;
  params.array_geometry.element_positions = {0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
                                             0.0, 1.0, 0.0, 0.0, 1.5, 0.0};
  params.min_subarray_size = 2;
  const perf::RegisteredOutputRecipe outputs{"snapshot_out",
                                             {"array_geometry", "is_valid_out"},
                                             {},
                                             4U,
                                             "",
                                             {{"array_geometry", "array_geometry"}}};
  const auto capture = [&](const auto& inputs, const auto& ports) {
    std::ostringstream stream;
    struct CaptureScope {
      ~CaptureScope() { perf::NativeCaptureState::stream = nullptr; }
    } scope;
    perf::NativeCaptureState::stream = &stream;
    perf::NativeCaptureState::first = true;
    perf::NativeCaptureState::case_id = "native-conditional-mask-probe";
    (void)perf::capture_native_recipe<Op>(
        inputs, "radar.SubarraySelectionOp.float.native_mask_probe", extents, params, ports,
        outputs, [](auto&, std::size_t) {});
    return YAML::Load(stream.str());
  };
  const auto absent = capture(std::array{snapshot}, std::array<std::string, 1>{"snapshot"});
  EXPECT_EQ(absent["outputs"]["snapshot_out"]["emissions"].size(), 3U);
  EXPECT_EQ(absent["outputs"]["array_geometry"]["emissions"].size(), 3U);
  EXPECT_EQ(absent["outputs"]["is_valid_out"]["emissions"].size(), 0U);
  const auto present =
      capture(std::array{mask, snapshot}, std::array<std::string, 2>{"is_valid", "snapshot"});
  EXPECT_EQ(present["outputs"]["snapshot_out"]["emissions"].size(), 3U);
  const auto masks = present["outputs"]["is_valid_out"]["emissions"];
  ASSERT_EQ(masks.size(), 3U);
  EXPECT_EQ(masks[1]["dtype"].as<std::string>(), "bool");
  EXPECT_EQ(masks[1]["data_base64"].as<std::string>(), "AAAAAA==");
}

TEST(PerfYamlStandard, NativeCaptureWaitsForSecondaryOutputsAndPreservesSuppression) {
  using Op = hr::FMCWRangeProcessorOp<float>;
  const std::map<std::string, int64_t> axes{{"rx", 1}, {"chirps", 8}, {"samples", 64}};
  const std::array<std::shared_ptr<holoscan::Tensor>, 1> inputs{perf::generate_recipe_input(
      {"adc_cube", {"rx", "chirps", "samples"}, "complex64", "device", "iq_cube", {}}, axes)};
  hr::MaxTensorExtents extents;
  extents.max_rx = 1;
  extents.max_chirps = 8;
  extents.max_samples_per_chirp = 64;
  extents.max_range_bins = 64;
  hr::FMCWWaveformSpec spec;
  spec.carrier_frequency = 77e9;
  spec.bandwidth = 150e6;
  spec.chirp_duration = 20e-6;
  spec.sample_rate = 12.8e6;
  spec.chirp_repetition_interval = 25e-6;
  const hr::FMCWWaveformDescriptor waveform(spec);
  const perf::RegisteredOutputRecipe outputs{"range_profiles", {"range_bins"}, {}, 4U};
  for (const bool waveform_present : {false, true}) {
    hr::FMCWRangeProcessorParams params;
    if (waveform_present)
      params.waveform = waveform;
    std::ostringstream stream;
    struct CaptureScope {
      ~CaptureScope() { perf::NativeCaptureState::stream = nullptr; }
    } scope;
    perf::NativeCaptureState::stream = &stream;
    perf::NativeCaptureState::first = true;
    perf::NativeCaptureState::case_id = "native-secondary-output-probe";
    (void)perf::capture_native_recipe<Op>(
        inputs, "radar.FMCWRangeProcessorOp.float.native_secondary_probe", extents, params,
        std::array<std::string, 1>{"adc_cube"}, outputs, [&](auto& staged, std::size_t frame) {
          if (frame != 1U)
            return;
          auto entry = holoscan::testing::frame_at(
              std::chrono::milliseconds{frame + 1U},
              holoscan::testing::input("waveform", hr::to_wire(waveform)));
          staged.fields.insert(staged.fields.begin(), std::move(entry.fields.front()));
        });
    const auto result = YAML::Load(stream.str());
    EXPECT_EQ(result["outputs"]["range_profiles"]["emissions"].size(), 3U);
    const auto bins = result["outputs"]["range_bins"]["emissions"];
    ASSERT_EQ(bins.size(), waveform_present ? 3U : 1U);
    if (!waveform_present) {
      // Sequence counts this output port's emissions; capture time joins the
      // single axis emission to the second input frame.
      EXPECT_EQ(bins[0]["sequence"].as<uint64_t>(), 1U);
      EXPECT_EQ(bins[0]["capture_time_ns"].as<int64_t>(), 2000000);
    }
  }
}

TEST(PerfYamlStandard, RunsTwoInputsWithCanonicalScalarParams) {
  using Op = hr::DopplerNotchOp<float>;
  perf::PerfRegisteredOperator<Op>::ensure_registered("radar.DopplerNotchOp.float.probe");
  perf::YamlRecipe recipe;
  recipe.operator_type = "radar.DopplerNotchOp.float.probe";
  recipe.inputs = {{"velocity_bins", {"doppler"}, "float32", "device", "centered_axis_ramp", {}},
                   {"rd_maps", {"doppler", "range"}, "complex64", "device", "iq_cube", {}}};
  recipe.constructor = YAML::Load("{extents: {max_doppler_bins: doppler, max_range_bins: range}}");
  recipe.outputs = {"rd_maps_out", {}, {}, 4U};
  perf::YamlStandardScenario<Op> scenario{
      "standard-small",
      "small",
      recipe,
      {{"doppler", 32}, {"range", 128}},
      YAML::Load("{center_velocity: 0.0, stop_half_width: 0.5, "
                 "transition_width: 0.0, cyclic_velocity_axis: true}")};
  perf::RunSettings settings;
  settings.warmup_frames = 1;
  settings.measured_frames = 3;
  settings.repeats = 1;
  const auto plan = perf::make_run_plan(settings, false);
  const auto collector = perf::run_graph(scenario, plan);
  EXPECT_EQ(collector.starts.size(), static_cast<size_t>(plan.total_frames));
  EXPECT_EQ(collector.ends.size(), collector.starts.size());
  EXPECT_TRUE(collector.live_emissions.empty());
}

TEST(PerfYamlStandard, RunsRegisteredScalarLiveForOneFrame) {
  using Op = hr::DopplerNotchOp<float>;
  perf::PerfRegisteredOperator<Op>::ensure_registered("radar.DopplerNotchOp.float.live_probe");
  perf::YamlRecipe recipe;
  recipe.operator_type = "radar.DopplerNotchOp.float.live_probe";
  recipe.inputs = {{"velocity_bins", {"doppler"}, "float32", "device", "centered_axis_ramp", {}},
                   {"rd_maps", {"doppler", "range"}, "complex64", "device", "iq_cube", {}}};
  recipe.constructor = YAML::Load("{extents: {max_doppler_bins: doppler, max_range_bins: range}}");
  recipe.outputs = {"rd_maps_out", {}, {}, 4U};
  recipe.live_ports = {{"center_velocity", "center_velocity"}};
  perf::YamlStandardScenario<Op> scenario{
      "center-live-small",
      "small",
      recipe,
      {{"doppler", 32}, {"range", 128}},
      YAML::Load("{center_velocity: 0.0, stop_half_width: 0.5, "
                 "transition_width: 0.0, cyclic_velocity_axis: true}"),
      "center_velocity",
      "double",
      "center_velocity",
      YAML::Load("0.1")};
  perf::RunSettings settings;
  settings.warmup_frames = 1;
  settings.measured_frames = 3;
  settings.repeats = 1;
  const auto plan = perf::make_run_plan(settings, true);
  const auto collector = perf::run_graph(scenario, plan);
  ASSERT_EQ(collector.starts.size(), static_cast<size_t>(plan.total_frames));
  ASSERT_EQ(collector.ends.size(), collector.starts.size());
  ASSERT_EQ(collector.live_emissions.size(), 1U);
  EXPECT_EQ(collector.live_emissions.front(), plan.override_at);
}

TEST(PerfYamlStandard, RunsDoubleDataWithoutChangingScalarLiveConfiguration) {
  using Op = hr::DopplerNotchOp<double>;
  perf::PerfRegisteredOperator<Op>::ensure_registered("radar.DopplerNotchOp.double.probe");
  perf::YamlRecipe recipe;
  recipe.operator_type = "radar.DopplerNotchOp.double.probe";
  recipe.inputs = {{"velocity_bins", {"doppler"}, "float64", "device", "centered_axis_ramp", {}},
                   {"rd_maps", {"doppler", "range"}, "complex128", "device", "iq_cube", {}}};
  recipe.constructor = YAML::Load("{extents: {max_doppler_bins: doppler, max_range_bins: range}}");
  recipe.outputs = {"rd_maps_out", {}, {}, 4U};
  recipe.live_ports = {{"center_velocity", "center_velocity"}};
  perf::YamlStandardScenario<Op> scenario{
      "double-live-small",
      "small",
      recipe,
      {{"doppler", 32}, {"range", 128}},
      YAML::Load("{center_velocity: 0.0, stop_half_width: 0.5, "
                 "transition_width: 0.0, cyclic_velocity_axis: true}"),
      "center_velocity",
      "double",
      "center_velocity",
      YAML::Load("0.1")};
  perf::RunSettings settings;
  settings.warmup_frames = 1;
  settings.measured_frames = 3;
  settings.repeats = 1;
  const auto plan = perf::make_run_plan(settings, true);
  const auto collector = perf::run_graph(scenario, plan);
  EXPECT_EQ(collector.starts.size(), static_cast<size_t>(plan.total_frames));
  EXPECT_EQ(collector.ends.size(), collector.starts.size());
  ASSERT_EQ(collector.live_emissions.size(), 1U);
  EXPECT_EQ(collector.live_emissions.front(), plan.override_at);
}

TEST(PerfYamlStandard, NativeCaptureExportsObservedCoordinatesAndTensorRepresentation) {
  using Op = hr::DopplerNotchOp<float>;
  const std::map<std::string, int64_t> axes{{"doppler", 32}, {"range", 128}};
  const std::array<std::shared_ptr<holoscan::Tensor>, 2> inputs{
      perf::generate_recipe_input(
          {"velocity_bins", {"doppler"}, "float32", "device", "centered_axis_ramp", {}}, axes),
      perf::generate_recipe_input(
          {"rd_maps", {"doppler", "range"}, "complex64", "device", "iq_cube", {}}, axes)};
  hr::MaxTensorExtents extents;
  extents.max_doppler_bins = 32;
  extents.max_range_bins = 128;
  const perf::RegisteredOutputRecipe outputs{"rd_maps_out", {}, {}, 4U};
  std::ostringstream stream;
  struct CaptureScope {
    ~CaptureScope() { perf::NativeCaptureState::stream = nullptr; }
  } scope;
  perf::NativeCaptureState::stream = &stream;
  perf::NativeCaptureState::first = true;
  perf::NativeCaptureState::case_id = "native-coordinate-probe";
  (void)perf::capture_native_recipe<Op>(
      inputs, "radar.DopplerNotchOp.float.native_probe", extents, hr::DopplerNotchParams{},
      std::array<std::string, 2>{"velocity_bins", "rd_maps"}, outputs, [](auto&, std::size_t) {});
  const auto result = YAML::Load(stream.str());
  const auto emissions = result["outputs"]["rd_maps_out"]["emissions"];
  ASSERT_EQ(emissions.size(), 3U);
  EXPECT_NE(emissions[0]["data_base64"].as<std::string>(),
            emissions[1]["data_base64"].as<std::string>());
  EXPECT_NE(emissions[1]["data_base64"].as<std::string>(),
            emissions[2]["data_base64"].as<std::string>());
  for (std::size_t index = 0; index < emissions.size(); ++index) {
    EXPECT_EQ(emissions[index]["sequence"].as<uint64_t>(), index + 1U);
    EXPECT_EQ(emissions[index]["capture_time_ns"].as<int64_t>(),
              static_cast<int64_t>(index + 1U) * 1000000);
    EXPECT_GT(emissions[index]["clock_id"].as<uint64_t>(), 0U);
    EXPECT_EQ(emissions[index]["memory_kind"].as<std::string>(), "device");
    EXPECT_EQ(emissions[index]["dtype"].as<std::string>(), "complex64");
    EXPECT_EQ(emissions[index]["shape"].as<std::vector<int64_t>>(),
              (std::vector<int64_t>{32, 128}));
    EXPECT_FALSE(emissions[index]["data_base64"].as<std::string>().empty());
  }
}

namespace {
struct SelectionScenario : perf::RecipeScenario {};
struct SelectionTraits {
  using Scenario = SelectionScenario;
  static Scenario parse_case(const perf::ProfileCase& profile_case,
                             const std::vector<int64_t>& shape, const perf::ProfileSuite& suite) {
    return perf::pilot_scenario<Scenario>(suite, profile_case, shape, shape.size());
  }
};
perf::ResultRow base_row(const SelectionScenario& scenario) {
  perf::ResultRow row;
  row.live_port = scenario.live_port;
  return row;
}
}  // namespace

TEST(PerfYamlStandard, FilteringAndReversalPreserveLiveYamlValues) {
  using Op = hr::DopplerNotchOp<float>;
  perf::PerfRegisteredOperator<Op>::ensure_registered("radar.DopplerNotchOp.float");
  const auto path = perf::operator_recipe_path(HOLOSCAN_RADAR_PERF_OPERATOR_ROOT, "doppler_notch");
  const auto suite = perf::load_profile_suite(path, "doppler_notch");
  perf::Options options;
  options.scope = "thorough";
  options.sizes = {"small", "medium", "large"};
  options.cases = {"baseline-small", "center-small", "transition-medium"};
  options.reverse_order = true;
  const auto selected = perf::detail::select_scenarios<SelectionTraits>(suite, options);
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_EQ(selected[0].scenario.id, "transition-medium");
  EXPECT_DOUBLE_EQ(selected[0].scenario.live_value.as<double>(), 1.0);
  EXPECT_EQ(selected[1].scenario.id, "center-small");
  EXPECT_DOUBLE_EQ(selected[1].scenario.live_value.as<double>(), 1.0);
  EXPECT_DOUBLE_EQ(selected[1].scenario.params["center_velocity"].as<double>(), 0.0);
  EXPECT_EQ(selected[2].scenario.id, "baseline-small");
  EXPECT_TRUE(selected[2].scenario.live_port.empty());
  // The suite can be decoded again without mutations from moved YAML handles.
  const auto again = perf::detail::select_scenarios<SelectionTraits>(suite, options);
  EXPECT_DOUBLE_EQ(again[1].scenario.live_value.as<double>(), 1.0);
}

TEST(PerfYamlStandard, MinimalWiringRetainsRequiredScreenIndices) {
  const auto path =
      perf::operator_recipe_path(HOLOSCAN_RADAR_PERF_OPERATOR_ROOT, "rank_one_residual_screen");
  const auto suite = perf::load_profile_suite(path, "rank_one_residual_screen");
  auto selected = suite.cases.front();
  selected.output_mode = "minimal";
  const auto recipe = perf::case_recipe(suite, selected);
  EXPECT_EQ(recipe.outputs.timed, "snapshot_out");
  EXPECT_EQ(recipe.outputs.drain, std::vector<std::string>{"selected_idx"});
  EXPECT_EQ(recipe.outputs.unwired,
            (std::vector<std::string>{"is_valid_out", "residual_score", "selection_mask"}));
}

TEST(PerfYamlStandard, MinimalWiringUnwiresOptionalPairedReductionOutput) {
  const auto path =
      perf::operator_recipe_path(HOLOSCAN_RADAR_PERF_OPERATOR_ROOT, "detection_reduction");
  const auto suite = perf::load_profile_suite(path, "detection_reduction");
  auto selected = suite.cases.front();
  selected.output_mode = "minimal";
  const auto recipe = perf::case_recipe(suite, selected);
  EXPECT_EQ(recipe.outputs.timed, "rep_idx");
  EXPECT_TRUE(recipe.outputs.timed_pair.empty());
  EXPECT_EQ(recipe.outputs.unwired, std::vector<std::string>{"reduced_points"});
}
