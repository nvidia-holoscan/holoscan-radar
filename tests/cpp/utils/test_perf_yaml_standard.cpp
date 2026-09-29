// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "utils/perf_yaml_standard.hpp"

namespace perf = holoscan::radar::perf;
namespace hr = holoscan::radar;

TEST(PerfYamlStandard, RunsTwoInputsWithCanonicalScalarParams) {
  using Op = hr::DopplerNotchOp<float>;
  perf::PerfRegisteredOperator<Op>::ensure_registered("radar.DopplerNotchOp.float.probe");
  perf::YamlRecipe recipe;
  recipe.operator_type = "radar.DopplerNotchOp.float.probe";
  recipe.inputs = {{"velocity_bins", {"doppler"}, "float32", "device", "power_map", {}},
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
  recipe.inputs = {{"velocity_bins", {"doppler"}, "float32", "device", "power_map", {}},
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
