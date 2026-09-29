// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "utils/perf_profiles.hpp"
#include "utils/perf_registered_recipe.hpp"
#include "utils/tensor_factory.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::perf {

struct RecipeInput {
  std::string port;
  std::vector<std::string> shape;
  std::string dtype;
  std::string memory;
  std::string generator;
  std::vector<std::string> omit_if_one;
};

struct YamlRecipe {
  std::string operator_type;
  std::vector<RecipeInput> inputs;
  YAML::Node constructor;
  YAML::Node params;
  YAML::Node fixtures;
  RegisteredOutputRecipe outputs;
  std::map<std::string, std::string> live_ports;
  std::map<std::string, std::string> live_codecs;
  std::map<std::string, std::string> live_axes;
};

inline std::vector<std::string> recipe_strings(const YAML::Node& node, std::string_view where) {
  if (!node.IsSequence())
    throw std::invalid_argument(std::string(where) + " must be a sequence");
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto& item : node) {
    const auto value = item.as<std::string>();
    if (value.empty() || !seen.insert(value).second)
      throw std::invalid_argument(std::string(where) + " contains empty or duplicate name");
    result.push_back(value);
  }
  return result;
}

inline YamlRecipe parse_yaml_recipe(const ProfileSuite& suite) {
  if (suite.version != 2)
    throw std::invalid_argument("YAML recipe requires profile schema version 2");
  const auto root = suite.recipe;
  require_keys(root,
               {"operator_type", "inputs", "constructor", "params", "fixtures", "outputs", "live",
                "extension"},
               "recipe");
  YamlRecipe result;
  result.operator_type = root["operator_type"].as<std::string>();
  if (result.operator_type.empty())
    throw std::invalid_argument("recipe operator_type is empty");
  const auto declared_inputs = root["inputs"];
  if (!declared_inputs.IsSequence() || declared_inputs.size() == 0 ||
      (declared_inputs.size() != 1 && declared_inputs.size() != 2 && declared_inputs.size() != 3 &&
       declared_inputs.size() != 6))
    throw std::invalid_argument("recipe inputs require one, two, three, or six data ports");
  std::set<std::string> input_names;
  for (const auto& node : declared_inputs) {
    require_keys(node, {"port", "shape", "dtype", "memory", "generator", "omit_if_one"},
                 "recipe input");
    for (const char* key : {"port", "shape", "dtype", "memory", "generator"})
      if (!node[key])
        throw std::invalid_argument("recipe input missing " + std::string(key));
    RecipeInput input;
    input.port = node["port"].as<std::string>();
    input.shape = recipe_strings(node["shape"], "recipe input shape");
    input.dtype = node["dtype"].as<std::string>();
    input.memory = node["memory"].as<std::string>();
    input.generator = node["generator"].as<std::string>();
    if (node["omit_if_one"])
      input.omit_if_one = recipe_strings(node["omit_if_one"], "recipe omit_if_one");
    if (!input_names.insert(input.port).second)
      throw std::invalid_argument("duplicate recipe input port: " + input.port);
    if (input.memory != "device")
      throw std::invalid_argument("recipe data input must use device memory: " + input.port);
    if (input.dtype != "complex64" && input.dtype != "float32" && input.dtype != "int64")
      throw std::invalid_argument("unsupported recipe dtype: " + input.dtype);
    if (input.generator != "iq_cube" && input.generator != "power_map" &&
        input.generator != "in_bounds_rd_indices")
      throw std::invalid_argument("unknown recipe generator: " + input.generator);
    if ((input.generator == "iq_cube" && input.dtype != "complex64") ||
        (input.generator == "power_map" && input.dtype != "float32") ||
        (input.generator == "in_bounds_rd_indices" && input.dtype != "int64"))
      throw std::invalid_argument("recipe generator/dtype mismatch: " + input.port);
    for (const auto& axis : input.omit_if_one)
      if (std::find(input.shape.begin(), input.shape.end(), axis) == input.shape.end())
        throw std::invalid_argument("omit_if_one names absent input axis: " + axis);
    result.inputs.push_back(std::move(input));
  }
  result.constructor = root["constructor"];
  require_keys(result.constructor, {"extents", "bounds_policy"}, "recipe constructor");
  if (static_cast<bool>(result.constructor["extents"]) ==
      static_cast<bool>(result.constructor["bounds_policy"]))
    throw std::invalid_argument("recipe constructor needs extents or bounds_policy");
  result.params = root["params"];
  if (!result.params.IsMap())
    throw std::invalid_argument("recipe params must be a map");
  if (root["fixtures"]) {
    result.fixtures = root["fixtures"];
    if (!result.fixtures.IsMap())
      throw std::invalid_argument("recipe fixtures must be a map");
  }
  const auto output = root["outputs"];
  require_keys(output, {"timed", "drain", "unwired", "queue_depth"}, "recipe outputs");
  if (!output["timed"])
    throw std::invalid_argument("recipe outputs requires timed");
  result.outputs.timed = output["timed"].as<std::string>();
  if (output["queue_depth"]) {
    result.outputs.queue_depth = output["queue_depth"].as<unsigned int>();
    if (result.outputs.queue_depth == 0)
      throw std::invalid_argument("recipe queue_depth must be positive");
  }
  if (output["drain"])
    result.outputs.drain = recipe_strings(output["drain"], "recipe drained outputs");
  if (output["unwired"])
    result.outputs.unwired = recipe_strings(output["unwired"], "recipe unwired outputs");
  std::set<std::string> output_names{result.outputs.timed};
  for (const auto& name : result.outputs.drain)
    if (!output_names.insert(name).second)
      throw std::invalid_argument("duplicate recipe output: " + name);
  for (const auto& name : result.outputs.unwired)
    if (!output_names.insert(name).second)
      throw std::invalid_argument("duplicate recipe output: " + name);
  const auto live = root["live"];
  if (live) {
    if (!live.IsMap())
      throw std::invalid_argument("recipe live must be a map");
    for (const auto& entry : live) {
      const auto name = entry.first.as<std::string>();
      require_keys(entry.second, {"physical_port", "codec", "axis"}, "recipe live port");
      if (!entry.second["physical_port"].IsScalar() || !entry.second["codec"].IsScalar())
        throw std::invalid_argument("recipe live needs physical_port and codec: " + name);
      const auto physical = entry.second["physical_port"].as<std::string>();
      const auto codec = entry.second["codec"].as<std::string>();
      if (physical.empty() || codec.empty())
        throw std::invalid_argument("recipe live has empty port or codec: " + name);
      result.live_ports.emplace(name, physical);
      result.live_codecs.emplace(name, codec);
      if (entry.second["axis"])
        result.live_axes.emplace(name, entry.second["axis"].as<std::string>());
    }
  }
  if (result.live_ports.size() != suite.live_ports.size())
    throw std::invalid_argument("recipe live manifest differs from suite live_ports");
  for (const auto& port : suite.live_ports)
    if (!result.live_ports.contains(port))
      throw std::invalid_argument("recipe lacks live port: " + port);
  return result;
}

inline void validate_recipe_manifest(const YamlRecipe& recipe,
                                     const std::vector<RecipeInput>& expected_inputs,
                                     const RegisteredOutputRecipe& expected_outputs,
                                     const std::map<std::string, std::string>& expected_live) {
  if (recipe.inputs.size() != expected_inputs.size())
    throw std::invalid_argument("recipe data input count disagrees with locked contract");
  for (size_t i = 0; i < recipe.inputs.size(); ++i) {
    if (recipe.inputs[i].port != expected_inputs[i].port ||
        recipe.inputs[i].shape != expected_inputs[i].shape ||
        recipe.inputs[i].omit_if_one != expected_inputs[i].omit_if_one ||
        recipe.inputs[i].dtype != expected_inputs[i].dtype ||
        recipe.inputs[i].memory != expected_inputs[i].memory ||
        recipe.inputs[i].generator != expected_inputs[i].generator)
      throw std::invalid_argument("recipe data input disagrees with locked contract: " +
                                  recipe.inputs[i].port);
  }
  if (recipe.outputs.timed != expected_outputs.timed ||
      recipe.outputs.drain != expected_outputs.drain ||
      recipe.outputs.unwired != expected_outputs.unwired ||
      recipe.outputs.queue_depth != expected_outputs.queue_depth)
    throw std::invalid_argument("recipe output activation disagrees with locked contract");
  if (recipe.live_ports != expected_live)
    throw std::invalid_argument("recipe live port mapping disagrees with locked contract");
}

inline YAML::Node merge_recipe_map(const YAML::Node& base, const YAML::Node& change) {
  if (!base.IsMap() || !change.IsMap())
    throw std::invalid_argument("recipe values must be maps");
  auto result = YAML::Clone(base);
  for (const auto& item : change)
    result[item.first.as<std::string>()] = item.second;
  return result;
}

inline std::map<std::string, int64_t> recipe_values(const ProfileSuite& suite,
                                                    const ProfileCase& profile_case) {
  auto result = suite.axes_for(profile_case.profile);
  const auto defaults = suite.recipe["fixtures"];
  if (defaults) {
    for (const auto& item : defaults)
      result[item.first.as<std::string>()] = item.second.as<int64_t>();
  }
  if (profile_case.fixture) {
    for (const auto& item : profile_case.fixture) {
      const auto name = item.first.as<std::string>();
      if (!defaults || !defaults[name])
        throw std::invalid_argument("unknown recipe fixture: " + name);
      result[name] = item.second.as<int64_t>();
    }
  }
  for (const auto& [name, value] : result)
    if (value <= 0)
      throw std::invalid_argument("nonpositive recipe axis or fixture: " + name);
  return result;
}

inline std::vector<int64_t> recipe_shape(const RecipeInput& input,
                                         const std::map<std::string, int64_t>& values) {
  std::vector<int64_t> result;
  for (const auto& axis : input.shape) {
    const auto found = values.find(axis);
    if (found == values.end())
      throw std::invalid_argument("unknown recipe shape axis: " + axis);
    if (std::find(input.omit_if_one.begin(), input.omit_if_one.end(), axis) !=
            input.omit_if_one.end() &&
        found->second == 1)
      continue;
    result.push_back(found->second);
  }
  return result;
}

inline std::vector<RegisteredInputContract> recipe_input_contracts(
    const YamlRecipe& recipe, const std::map<std::string, int64_t>& values) {
  std::vector<RegisteredInputContract> result;
  for (const auto& input : recipe.inputs)
    result.push_back({input.port, input.dtype, input.memory, recipe_shape(input, values).size()});
  return result;
}

inline std::shared_ptr<holoscan::Tensor> generate_recipe_input(
    const RecipeInput& input, const std::map<std::string, int64_t>& values) {
  const auto shape = recipe_shape(input, values);
  if (input.generator == "iq_cube" && input.dtype == "complex64")
    return make_iq_cube_device<float>(shape);
  if (input.generator == "power_map" && input.dtype == "float32")
    return make_power_map_device<float>(shape);
  if (input.generator == "in_bounds_rd_indices" && input.dtype == "int64") {
    if (shape.size() != 2 || shape[1] != 2)
      throw std::invalid_argument("in_bounds_rd_indices requires [detections, 2]");
    const auto doppler = values.at("doppler");
    const auto range = values.at("range");
    std::vector<int64_t> host(static_cast<size_t>(shape[0] * 2));
    for (int64_t i = 0; i < shape[0]; ++i) {
      const auto cell = (i * 997) % (doppler * range);
      host[static_cast<size_t>(i * 2)] = cell / range;
      host[static_cast<size_t>(i * 2 + 1)] = cell % range;
    }
    auto tensor = make_device_tensor(shape, DLDataType{kDLInt, 64, 1});
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(int64_t) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage in-bounds detection indices");
    return tensor;
  }
  throw std::invalid_argument("generator/dtype mismatch at input " + input.port);
}

inline MaxTensorExtents direct_recipe_extents(const YamlRecipe& recipe,
                                              const std::map<std::string, int64_t>& values) {
  const auto declared = recipe.constructor["extents"];
  if (!declared)
    throw std::invalid_argument("recipe uses a typed bounds policy");
  require_keys(
      declared,
      {"max_samples_per_chirp", "max_chirps", "max_range_bins", "max_doppler_bins", "max_rx",
       "max_tx", "max_interleaved_chirps", "max_virtual_channels", "max_channels", "max_az",
       "max_el", "max_detections", "max_points", "max_items", "max_snapshots", "max_features",
       "max_sources", "max_clusters", "max_subarray_channels"},
      "recipe extents");
  using Member = std::optional<int64_t> MaxTensorExtents::*;
  static constexpr std::array<std::pair<std::string_view, Member>, 19> fields{{
      {"max_samples_per_chirp", &MaxTensorExtents::max_samples_per_chirp},
      {"max_chirps", &MaxTensorExtents::max_chirps},
      {"max_range_bins", &MaxTensorExtents::max_range_bins},
      {"max_doppler_bins", &MaxTensorExtents::max_doppler_bins},
      {"max_rx", &MaxTensorExtents::max_rx},
      {"max_tx", &MaxTensorExtents::max_tx},
      {"max_interleaved_chirps", &MaxTensorExtents::max_interleaved_chirps},
      {"max_virtual_channels", &MaxTensorExtents::max_virtual_channels},
      {"max_channels", &MaxTensorExtents::max_channels},
      {"max_az", &MaxTensorExtents::max_az},
      {"max_el", &MaxTensorExtents::max_el},
      {"max_detections", &MaxTensorExtents::max_detections},
      {"max_points", &MaxTensorExtents::max_points},
      {"max_items", &MaxTensorExtents::max_items},
      {"max_snapshots", &MaxTensorExtents::max_snapshots},
      {"max_features", &MaxTensorExtents::max_features},
      {"max_sources", &MaxTensorExtents::max_sources},
      {"max_clusters", &MaxTensorExtents::max_clusters},
      {"max_subarray_channels", &MaxTensorExtents::max_subarray_channels},
  }};
  MaxTensorExtents result;
  for (const auto& item : declared) {
    const auto field = item.first.as<std::string>();
    const auto axis = item.second.as<std::string>();
    const auto found = values.find(axis);
    if (found == values.end())
      throw std::invalid_argument("unknown recipe extent axis: " + axis);
    for (const auto& [name, member] : fields) {
      if (field == name) {
        result.*member = found->second;
        break;
      }
    }
  }
  return result;
}

inline void validate_recipe_live_axis(const YamlRecipe& recipe, std::string_view logical,
                                      const std::map<std::string, int64_t>& values) {
  const auto port = std::string(logical);
  if (recipe.live_codecs.at(port) != "scaled_axis")
    return;
  const auto axis = recipe.live_axes.find(port);
  if (axis == recipe.live_axes.end() || !values.contains(axis->second))
    throw std::invalid_argument("scaled live codec needs a declared profile axis: " + port);
}

inline void validate_recipe_case(const ProfileSuite& suite, const ProfileCase& profile_case) {
  const auto recipe = parse_yaml_recipe(suite);
  const auto values = recipe_values(suite, profile_case);
  for (const auto& input : recipe.inputs) {
    const auto shape = recipe_shape(input, values);
    if (input.generator == "in_bounds_rd_indices" &&
        (shape.size() != 2 || shape[1] != 2 || !values.contains("doppler") ||
         !values.contains("range")))
      throw std::invalid_argument(
          "in_bounds_rd_indices requires [detections, 2] and doppler/range axes");
  }
  if (recipe.constructor["extents"])
    (void)direct_recipe_extents(recipe, values);
  const auto params = merge_recipe_map(recipe.params, profile_case.parameters);
  auto registered = holoscan::authoring::authoring_registry().operator_type(recipe.operator_type);
  if (!registered)
    throw std::invalid_argument("unregistered recipe operator: " + recipe.operator_type);
  const auto* schema = registered->get().param_schema;
  if (!schema && params.size() != 0)
    throw std::invalid_argument("registered recipe operator has no Params schema");
  for (const auto& field : params) {
    const auto name = field.first.as<std::string>();
    if (!schema || !holoscan::authoring::detail::find_parameter_field(*schema, name))
      throw std::invalid_argument("unknown canonical Params field: " + name);
  }
  if (profile_case.override_value) {
    const auto logical = (*profile_case.override_value)["port"].as<std::string>();
    if (!recipe.live_ports.contains(logical))
      throw std::invalid_argument("recipe has no live port: " + logical);
    validate_recipe_live_axis(recipe, logical, values);
  }
}

}  // namespace holoscan::radar::perf
