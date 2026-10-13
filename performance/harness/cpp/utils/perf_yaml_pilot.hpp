// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/sdk/live_override_payloads.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"
#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_codecs.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/perf_yaml_recipe.hpp"

namespace holoscan::radar::perf {

// Common identity and recipe state for semantic pilot extensions. The YAML
// case, rather than an adapter helper, owns each ID and live selection.
struct RecipeScenario {
  std::string id;
  std::string size_id;
  YamlRecipe recipe;
  YAML::Node params;
  std::map<std::string, int64_t> values;
  std::string live_port;
  YAML::Node live_value;
  bool cached_override{false};

  [[nodiscard]] int64_t axis(std::string_view name) const { return values.at(std::string(name)); }
};

inline RecipeScenario recipe_scenario(const ProfileSuite& suite, const ProfileCase& c) {
  validate_recipe_case(suite, c);
  RecipeScenario result;
  result.id = c.id;
  result.size_id = c.profile;
  result.recipe = case_recipe(suite, c);
  result.params = merge_recipe_map(result.recipe.params, c.parameters);
  result.values = recipe_values(suite, c);
  if (c.override_value) {
    result.live_port = (*c.override_value)["port"].as<std::string>();
    result.live_value = (*c.override_value)["value"];
    result.cached_override =
        (*c.override_value)["cache"] && (*c.override_value)["cache"].as<std::string>() == "cached";
  }
  return result;
}

// Build the typed scenario only after the common recipe and profile checks.
template <typename Scenario>
Scenario pilot_scenario(const ProfileSuite& suite, const ProfileCase& profile_case,
                        const std::vector<int64_t>& shape, std::size_t expected_rank) {
  if (shape.size() != expected_rank)
    throw std::invalid_argument("recipe profile rank mismatch");
  Scenario scenario;
  static_cast<RecipeScenario&>(scenario) = recipe_scenario(suite, profile_case);
  return scenario;
}

inline void require_recipe_params(const YAML::Node& params,
                                  std::initializer_list<std::string_view> allowed,
                                  std::initializer_list<std::string_view> required) {
  require_keys(params, allowed, "parameters");
  for (const auto field : required)
    if (!params[std::string(field)])
      throw std::invalid_argument("missing required Params field: " + std::string(field));
}

inline void require_recipe_params(const YAML::Node& params,
                                  std::initializer_list<std::string_view> fields) {
  require_recipe_params(params, fields, fields);
}

template <typename Value>
std::array<Value, 2> recipe_pair(const YAML::Node& node, std::string_view field) {
  if (!node.IsSequence() || node.size() != 2)
    throw std::invalid_argument(std::string(field) + " must have two values");
  return {node[0].as<Value>(), node[1].as<Value>()};
}

inline int64_t scaled_recipe_axis(const YAML::Node& node, int64_t extent, std::string_view field) {
  const auto scale = node["scale"].as<int64_t>();
  if (scale <= 0 || extent <= 0 || scale > std::numeric_limits<int64_t>::max() / extent)
    throw std::invalid_argument(std::string(field) + " scale must be positive and fit int64");
  return scale * extent;
}

template <typename Op>
YamlRecipe validate_pilot_suite(const ProfileSuite& suite, std::string_view operator_type,
                                std::string_view bounds_policy,
                                const std::vector<RecipeInput>& expected_inputs,
                                const RegisteredOutputRecipe& expected_outputs,
                                const std::map<std::string, std::string>& expected_live,
                                const std::map<std::string, std::string>& expected_codecs = {}) {
  auto recipe = parse_yaml_recipe(suite);
  if (recipe.operator_type != precision_operator(operator_type))
    throw std::invalid_argument("recipe operator type differs from typed pilot registration");
  validate_recipe_manifest(recipe, expected_inputs, expected_outputs, expected_live);
  if (recipe.live_codecs != expected_codecs)
    throw std::invalid_argument("recipe live codecs disagree with typed payload manifest");
  if (bounds_policy.empty()) {
    if (!recipe.constructor["extents"])
      throw std::invalid_argument("pilot recipe needs direct constructor extents");
  } else if (!recipe.constructor["bounds_policy"] ||
             recipe.constructor["bounds_policy"].as<std::string>() != bounds_policy) {
    throw std::invalid_argument("pilot recipe bounds policy differs from typed builder");
  }
  PerfRegisteredOperator<Op>::ensure_registered(recipe.operator_type);
  return recipe;
}

inline SizeDef recipe_size_def(const std::string& id, const std::vector<int64_t>& shape,
                               std::size_t rank, std::size_t work_axes,
                               std::size_t trailing_axes = 0) {
  if (shape.size() != rank || work_axes > rank || trailing_axes > rank)
    throw std::invalid_argument("recipe profile rank mismatch");
  const auto work = std::accumulate(shape.begin(), shape.begin() + work_axes, int64_t{1},
                                    std::multiplies<int64_t>{});
  const auto map =
      trailing_axes == 0 ? shape : std::vector<int64_t>(shape.end() - trailing_axes, shape.end());
  return {id, map, work, id};
}

inline void set_recipe_throughput(ResultRow& row, double cells,
                                  std::optional<double> slices = std::nullopt,
                                  std::optional<double> estimated_bytes = std::nullopt) {
  if (row.latency.median_ms <= 0.0)
    return;
  const double seconds = row.latency.median_ms / 1000.0;
  row.cells_per_second = cells / seconds;
  if (slices)
    row.slices_per_second = *slices / seconds;
  if (estimated_bytes)
    row.estimated_effective_gbps = *estimated_bytes / seconds / 1e9;
}

template <std::size_t N>
struct RecipeInputs {
  std::array<std::shared_ptr<holoscan::Tensor>, N> tensors;
  std::array<std::string, N> ports;
};

template <std::size_t N>
RecipeInputs<N> recipe_inputs(const RecipeScenario& scenario) {
  if (scenario.recipe.inputs.size() != N)
    throw std::invalid_argument("recipe input count disagrees with typed operator arity");
  RecipeInputs<N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result.tensors[i] = generate_recipe_input(scenario.recipe.inputs[i], scenario.values);
    result.ports[i] = scenario.recipe.inputs[i].port;
  }
  return result;
}

template <typename Op, std::size_t N>
TimingCollector run_recipe_case(const RecipeScenario& scenario, const RunPlan& plan,
                                const MaxTensorExtents& extents,
                                const typename Op::Params& params) {
  if (!scenario.live_port.empty())
    throw std::invalid_argument("baseline recipe has a selected live port");
  auto inputs = recipe_inputs<N>(scenario);
  return run_registered_recipe<Op>(std::move(inputs.tensors), plan, scenario.recipe.operator_type,
                                   extents, params, inputs.ports, scenario.recipe.outputs,
                                   recipe_input_contracts(scenario.recipe, scenario.values),
                                   scenario.recipe.publication);
}

template <typename Op, std::size_t N>
TimingCollector run_recipe_staged_case(
    const RecipeScenario& scenario, const RunPlan& plan, const MaxTensorExtents& extents,
    const typename Op::Params& params,
    std::array<std::shared_ptr<holoscan::Tensor>, N> staged_inputs) {
  if (!scenario.live_port.empty() || scenario.recipe.inputs.size() != N)
    throw std::invalid_argument("staged recipe input count or live selection mismatch");
  std::array<std::string, N> ports;
  for (std::size_t i = 0; i < N; ++i)
    ports[i] = scenario.recipe.inputs[i].port;
  return run_registered_recipe<Op>(std::move(staged_inputs), plan, scenario.recipe.operator_type,
                                   extents, params, ports, scenario.recipe.outputs,
                                   recipe_input_contracts(scenario.recipe, scenario.values),
                                   scenario.recipe.publication);
}

template <typename Op, std::size_t N, typename Value>
TimingCollector run_recipe_live_case(const RecipeScenario& scenario, const RunPlan& plan,
                                     const MaxTensorExtents& extents,
                                     const typename Op::Params& params, Value initial,
                                     Value changed) {
  if (scenario.live_port.empty())
    throw std::invalid_argument("live recipe has no selected port");
  auto inputs = recipe_inputs<N>(scenario);
  const auto live =
      RegisteredLiveRecipe<Value>{scenario.recipe.live_ports.at(scenario.live_port),
                                  std::move(initial), std::move(changed), scenario.cached_override};
  return run_registered_live_recipe<Op>(
      std::move(inputs.tensors), plan, scenario.recipe.operator_type, extents, params, inputs.ports,
      scenario.recipe.outputs, live, recipe_input_contracts(scenario.recipe, scenario.values),
      scenario.recipe.publication);
}

template <typename Op, std::size_t N, typename Value, typename Connector>
TimingCollector run_recipe_live_case_with_connector(
    const RecipeScenario& scenario, const RunPlan& plan, const MaxTensorExtents& extents,
    const typename Op::Params& params, Value initial, Value changed, Connector connector) {
  if (scenario.live_port.empty())
    throw std::invalid_argument("live recipe has no selected port");
  auto inputs = recipe_inputs<N>(scenario);
  const auto live =
      RegisteredLiveRecipe<Value>{scenario.recipe.live_ports.at(scenario.live_port),
                                  std::move(initial), std::move(changed), scenario.cached_override};
  return run_registered_live_recipe_with_connector<Op>(
      std::move(inputs.tensors), plan, scenario.recipe.operator_type, extents, params, inputs.ports,
      scenario.recipe.outputs, live, connector,
      recipe_input_contracts(scenario.recipe, scenario.values), scenario.recipe.publication);
}

// The YAML codec selects the typed payload; the registered runner resolves the
// physical port and authors the source connection only for the selected case.
// Semantic codecs (for example CFAR alpha calibration) stay with the adapter.
template <typename Op, std::size_t N, typename CustomLive>
TimingCollector run_recipe_selected_case(const RecipeScenario& scenario, const RunPlan& plan,
                                         const MaxTensorExtents& extents,
                                         const typename Op::Params& params,
                                         CustomLive custom_live) {
  if (scenario.live_port.empty())
    return run_recipe_case<Op, N>(scenario, plan, extents, params);
  const auto found = scenario.recipe.live_codecs.find(scenario.live_port);
  if (found == scenario.recipe.live_codecs.end())
    throw std::invalid_argument("recipe lacks live codec: " + scenario.live_port);
  const auto& codec = found->second;
  const auto& value = scenario.live_value;
  const auto emit = [&](const auto& changed) {
    return run_recipe_live_case<Op, N>(scenario, plan, extents, params, changed, changed);
  };
  if (codec == "bool")
    return emit(value.as<bool>());
  if (codec == "int64")
    return emit(value.as<int64_t>());
  if (codec == "double")
    return emit(value.as<double>());
  if (codec == "scaled_axis")
    return emit(scaled_recipe_axis(value,
                                   scenario.axis(scenario.recipe.live_axes.at(scenario.live_port)),
                                   scenario.live_port));
  if (codec == "window_spec")
    return emit(recipe_window(recipe_window_type(value["type"].as<std::string>())));
  if (codec == "waveform_descriptor")
    return emit(to_wire(recipe_fmcw_waveform(scenario.params[scenario.live_port],
                                             value["pri_seconds"].as<double>())));
  if (codec == "cfar_cell_window") {
    const auto pair = recipe_pair<int64_t>(value, scenario.live_port);
    return emit(CfarCellWindowOverride{pair[0], pair[1]});
  }
  if (codec == "spatial_smoothing") {
    const auto sizes = value.as<std::vector<int64_t>>();
    if (sizes.empty() || sizes.size() > 2)
      throw std::invalid_argument("spatial_smoothing needs one or two extents");
    SpatialSmoothingOverride payload{};
    payload.count = static_cast<int64_t>(sizes.size());
    payload.first = sizes[0];
    if (sizes.size() == 2)
      payload.second = sizes[1];
    return emit(payload);
  }
  if (codec == "host_int64_vector")
    return emit(value.as<std::vector<int64_t>>());
  if (codec == "host_float64_vector")
    return emit(value.as<std::vector<double>>());
  if (codec == "shifted_linspace")
    return emit(recipe_coordinates(scenario.params[scenario.live_port], scenario.values,
                                   value["shift"].as<double>()));
  if (codec == "ula_geometry_patch") {
    auto geometry = YAML::Clone(scenario.params[scenario.live_port]);
    geometry["spacing_wavelengths"] = value["spacing_wavelengths"];
    return emit(recipe_ula_geometry(geometry, scenario.axis("channels")));
  }
  return custom_live(codec);
}

template <typename Op, std::size_t N>
TimingCollector run_recipe_selected_case(const RecipeScenario& scenario, const RunPlan& plan,
                                         const MaxTensorExtents& extents,
                                         const typename Op::Params& params) {
  return run_recipe_selected_case<Op, N>(
      scenario, plan, extents, params, [&](const std::string& codec) -> TimingCollector {
        throw std::invalid_argument("unsupported recipe live codec: " + codec);
      });
}

}  // namespace holoscan::radar::perf
