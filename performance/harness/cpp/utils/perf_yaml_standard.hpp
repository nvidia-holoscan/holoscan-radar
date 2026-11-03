// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <concepts>
#include <functional>
#include <map>
#include <span>
#include <deque>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/authoring/parameter_materializer.hpp>

#include "utils/perf_driver.hpp"
#include "utils/perf_yaml_recipe.hpp"

namespace holoscan::radar::perf {

// Generic path for operators whose fixture, Params and bounds are covered by
// the declared recipe. Contract-dependent objects use a typed adapter.
class YamlParameterAssignments {
 public:
  YamlParameterAssignments(const holoscan::ParamSchemaDescriptor& schema, const YAML::Node& yaml) {
    for (const auto& item : yaml) {
      const auto name = item.first.as<std::string>();
      const auto* field = holoscan::authoring::detail::find_parameter_field(schema, name);
      if (!field)
        throw std::invalid_argument("unknown canonical Params field: " + name);
      names_.push_back(name);
      assignments_.push_back({names_.back(), convert(item.second, field->type_shape)});
    }
  }

  [[nodiscard]] const std::vector<holoscan::authoring::ParameterAssignment>& values() const {
    return assignments_;
  }

 private:
  holoscan::authoring::ParameterValue convert(const YAML::Node& node,
                                              const holoscan::ParamFieldTypeShape& shape) {
    using Kind = holoscan::ParamLogicalType;
    using Value = holoscan::authoring::ParameterValue;
    if (shape.type == Kind::kSequence) {
      if (!node.IsSequence())
        throw std::invalid_argument("Params sequence requires YAML sequence");
      sequences_.emplace_back();
      auto& sequence = sequences_.back();
      sequence.reserve(node.size());
      auto element_shape = shape;
      element_shape.type = shape.element_or_rep_type;
      for (const auto& element : node)
        sequence.push_back(convert(element, element_shape));
      return Value::from(std::span<const Value>(sequence));
    }
    if (!node.IsScalar())
      throw std::invalid_argument("Params field requires scalar or typed adapter");
    switch (shape.type) {
      case Kind::kBool:
        return Value::from(node.as<bool>());
      case Kind::kSignedInteger:
        try {
          return Value::from(node.as<int64_t>());
        } catch (const YAML::Exception&) {
          strings_.push_back(node.as<std::string>());
          return Value::from(std::string_view(strings_.back()));
        }
      case Kind::kUnsignedInteger:
        return Value::from(node.as<uint64_t>());
      case Kind::kFloatingPoint:
        return Value::from(node.as<double>());
      case Kind::kString:
        strings_.push_back(node.as<std::string>());
        return Value::from(std::string_view(strings_.back()));
      default:
        throw std::invalid_argument("Params field needs a typed adapter");
    }
  }

  std::deque<std::string> names_;
  std::deque<std::string> strings_;
  std::deque<std::vector<holoscan::authoring::ParameterValue>> sequences_;
  std::vector<holoscan::authoring::ParameterAssignment> assignments_;
};

template <typename Value>
std::vector<Value> yaml_live_vector(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() == 0)
    throw std::invalid_argument("host-vector live value requires a nonempty YAML sequence");
  std::vector<Value> result;
  result.reserve(node.size());
  for (const auto& element : node)
    result.push_back(element.as<Value>());
  return result;
}

template <typename Op>
struct YamlStandardScenario {
  std::string id;
  std::string size_id;
  YamlRecipe recipe;
  std::map<std::string, int64_t> values;
  YAML::Node params;
  std::string live_port;
  std::string live_codec;
  std::string physical_port;
  YAML::Node changed;
};

template <typename Op>
ResultRow base_row(const YamlStandardScenario<Op>& scenario) {
  ResultRow row;
  row.op_name = scenario.recipe.operator_type;
  row.variant = "default";
  row.live_port = scenario.live_port;
  const auto shape = recipe_shape(scenario.recipe.inputs.at(0), scenario.values);
  set_input_and_trailing_map_shape(row, shape);
  return row;
}

template <typename Op, std::size_t N>
TimingCollector run_standard_arity(const YamlStandardScenario<Op>& scenario, const RunPlan& plan,
                                   const MaxTensorExtents& extents,
                                   const typename Op::Params& params) {
  std::array<std::shared_ptr<holoscan::Tensor>, N> tensors;
  std::array<std::string, N> ports;
  for (std::size_t i = 0; i < N; ++i) {
    tensors[i] = generate_recipe_input(scenario.recipe.inputs[i], scenario.values);
    ports[i] = scenario.recipe.inputs[i].port;
  }
  const auto contracts = recipe_input_contracts(scenario.recipe, scenario.values);
  if (scenario.live_port.empty())
    return run_registered_recipe<Op>(std::move(tensors), plan, scenario.recipe.operator_type,
                                     extents, params, ports, scenario.recipe.outputs, contracts);
  if (scenario.live_codec == "bool") {
    const auto value = scenario.changed.template as<bool>();
    return run_registered_live_recipe<Op>(
        std::move(tensors), plan, scenario.recipe.operator_type, extents, params, ports,
        scenario.recipe.outputs,
        RegisteredLiveRecipe<bool>{scenario.physical_port, value, value, false}, contracts);
  }
  if (scenario.live_codec == "int64") {
    const auto value = scenario.changed.template as<int64_t>();
    return run_registered_live_recipe<Op>(
        std::move(tensors), plan, scenario.recipe.operator_type, extents, params, ports,
        scenario.recipe.outputs,
        RegisteredLiveRecipe<int64_t>{scenario.physical_port, value, value, false}, contracts);
  }
  if (scenario.live_codec == "double") {
    const auto value = scenario.changed.template as<double>();
    return run_registered_live_recipe<Op>(
        std::move(tensors), plan, scenario.recipe.operator_type, extents, params, ports,
        scenario.recipe.outputs,
        RegisteredLiveRecipe<double>{scenario.physical_port, value, value, false}, contracts);
  }
  if (scenario.live_codec == "host_float64_vector") {
    const auto value = yaml_live_vector<double>(scenario.changed);
    return run_registered_live_recipe<Op>(
        std::move(tensors), plan, scenario.recipe.operator_type, extents, params, ports,
        scenario.recipe.outputs,
        RegisteredLiveRecipe<std::vector<double>>{scenario.physical_port, value, value, false},
        contracts);
  }
  if (scenario.live_codec == "host_int64_vector") {
    const auto value = yaml_live_vector<int64_t>(scenario.changed);
    return run_registered_live_recipe<Op>(
        std::move(tensors), plan, scenario.recipe.operator_type, extents, params, ports,
        scenario.recipe.outputs,
        RegisteredLiveRecipe<std::vector<int64_t>>{scenario.physical_port, value, value, false},
        contracts);
  }
  throw std::invalid_argument("standard YAML recipe needs a registered live codec");
}

template <typename Op>
TimingCollector run_graph(const YamlStandardScenario<Op>& scenario, const RunPlan& plan) {
  const auto bounds = direct_recipe_extents(scenario.recipe, scenario.values);
  auto registered =
      holoscan::authoring::authoring_registry().operator_type(scenario.recipe.operator_type);
  if (!registered)
    throw std::invalid_argument("unregistered recipe operator");
  typename Op::Params params{};
  if constexpr (!std::same_as<typename Op::Params, holoscan::NoParams>) {
    YamlParameterAssignments assignments(*registered->get().param_schema, scenario.params);
    auto materialized = holoscan::authoring::materialize_parameters(
        *registered->get().param_type, registered->get().param_schema, assignments.values());
    if (!materialized)
      throw std::invalid_argument("invalid recipe Params: " + materialized.error().message);
    params = *static_cast<const typename Op::Params*>(materialized->data());
  }
  switch (scenario.recipe.inputs.size()) {
    case 1:
      return run_standard_arity<Op, 1>(scenario, plan, bounds, params);
    case 2:
      return run_standard_arity<Op, 2>(scenario, plan, bounds, params);
    case 3:
      return run_standard_arity<Op, 3>(scenario, plan, bounds, params);
    case 6:
      return run_standard_arity<Op, 6>(scenario, plan, bounds, params);
    default:
      throw std::invalid_argument("unsupported recipe input arity");
  }
}

template <typename Op>
struct YamlStandardTraits : PerfDriverDefaults {
  using Scenario = YamlStandardScenario<Op>;
  static const char* tool_description() { return "YAML-defined operator performance assessment"; }
  static SizeDef size_def(const std::string& id, const std::vector<int64_t>& shape) {
    return {id, shape,
            std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{}),
            id};
  }
  static void validate_suite(const ProfileSuite& suite) {
    const auto recipe = parse_yaml_recipe(suite);
    if (!recipe.constructor["extents"])
      throw std::invalid_argument("standard YAML recipe needs direct extents");
    PerfRegisteredOperator<Op>::ensure_registered(recipe.operator_type);
  }
  static Scenario parse_case(const ProfileCase& profile_case, const std::vector<int64_t>&,
                             const ProfileSuite& suite) {
    validate_recipe_case(suite, profile_case);
    const auto recipe = case_recipe(suite, profile_case);
    const auto params = merge_recipe_map(recipe.params, profile_case.parameters);
    const auto values = recipe_values(suite, profile_case);
    auto registered = holoscan::authoring::authoring_registry().operator_type(recipe.operator_type);
    if constexpr (std::same_as<typename Op::Params, holoscan::NoParams>) {
      if (params.size() != 0)
        throw std::invalid_argument("operator has no Params");
    } else {
      YamlParameterAssignments assignments(*registered->get().param_schema, params);
      auto materialized = holoscan::authoring::materialize_parameters(
          *registered->get().param_type, registered->get().param_schema, assignments.values());
      if (!materialized)
        throw std::invalid_argument("invalid recipe Params: " + materialized.error().message);
    }
    Scenario result{profile_case.id, profile_case.profile, recipe, values, params};
    if (profile_case.override_value) {
      result.live_port = (*profile_case.override_value)["port"].as<std::string>();
      const auto declared = suite.recipe["live"][result.live_port];
      if (!declared["codec"])
        throw std::invalid_argument("standard live recipe requires a codec: " + result.live_port);
      result.live_codec = declared["codec"].template as<std::string>();
      result.physical_port = recipe.live_ports.at(result.live_port);
      result.changed = (*profile_case.override_value)["value"];
      if ((*profile_case.override_value)["cache"])
        throw std::invalid_argument("standard scalar live recipe cannot prime a cache");
      if (result.live_codec == "bool")
        (void)result.changed.template as<bool>();
      else if (result.live_codec == "int64")
        (void)result.changed.template as<int64_t>();
      else if (result.live_codec == "double")
        (void)result.changed.template as<double>();
      else if (result.live_codec == "host_float64_vector")
        (void)yaml_live_vector<double>(result.changed);
      else if (result.live_codec == "host_int64_vector")
        (void)yaml_live_vector<int64_t>(result.changed);
      else
        throw std::invalid_argument("standard YAML recipe needs a registered live codec");
    }
    return result;
  }
};

}  // namespace holoscan::radar::perf
