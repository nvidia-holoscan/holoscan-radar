// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "utils/perf_profiles.hpp"
#include "utils/perf_precision.hpp"
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
  std::string index_doppler_divisor;
};

struct YamlRecipe {
  std::string operator_type;
  std::vector<RecipeInput> inputs;
  YAML::Node constructor;
  YAML::Node params;
  YAML::Node fixtures;
  YAML::Node derived_axes;
  YAML::Node extension;
  PerfPublishMode publication{PerfPublishMode::kCopy};
  RegisteredOutputRecipe outputs;
  std::map<std::string, std::string> live_ports;
  std::map<std::string, std::string> live_codecs;
  std::map<std::string, std::string> live_axes;
  std::vector<std::string> optional_tensor_outputs;
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
                "extension", "derived_axes", "publication"},
               "recipe");
  YamlRecipe result;
  result.operator_type = root["operator_type"].as<std::string>();
  if (result.operator_type.empty())
    throw std::invalid_argument("recipe operator_type is empty");
  if (root["publication"]) {
    const auto mode = root["publication"].as<std::string>();
    if (mode == "copy")
      result.publication = PerfPublishMode::kCopy;
    else if (mode == "fill_in_place")
      result.publication = PerfPublishMode::kFillInPlace;
    else
      throw std::invalid_argument("unknown recipe publication mode: " + mode);
  }
  const auto declared_inputs = root["inputs"];
  if (!declared_inputs.IsSequence() || declared_inputs.size() == 0 ||
      (declared_inputs.size() != 1 && declared_inputs.size() != 2 && declared_inputs.size() != 3 &&
       declared_inputs.size() != 4 && declared_inputs.size() != 5 && declared_inputs.size() != 6))
    throw std::invalid_argument("recipe inputs require one through six data ports");
  std::set<std::string> input_names;
  for (const auto& node : declared_inputs) {
    require_keys(
        node,
        {"port", "shape", "dtype", "memory", "generator", "omit_if_one", "index_doppler_divisor"},
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
    if (node["index_doppler_divisor"])
      input.index_doppler_divisor = node["index_doppler_divisor"].as<std::string>();
    if (!input.index_doppler_divisor.empty() && input.generator != "in_bounds_rd_indices")
      throw std::invalid_argument("index_doppler_divisor requires detection index generator");
    if (!input_names.insert(input.port).second)
      throw std::invalid_argument("duplicate recipe input port: " + input.port);
    if (input.memory != "device")
      throw std::invalid_argument("recipe data input must use device memory: " + input.port);
    if (input.dtype != "complex64" && input.dtype != "float32" && input.dtype != "int64" &&
        input.dtype != "int32" && input.dtype != "uint8" && input.dtype != "bool")
      throw std::invalid_argument("unsupported recipe dtype: " + input.dtype);
    if (input.generator != "iq_cube" && input.generator != "power_map" &&
        input.generator != "in_bounds_rd_indices" && input.generator != "cluster_blobs" &&
        input.generator != "rank_one_mix" && input.generator != "zero_float" &&
        input.generator != "cluster_labels" && input.generator != "reduction_indices" &&
        input.generator != "reduction_points" && input.generator != "axis_ramp" &&
        input.generator != "centered_axis_ramp" && input.generator != "valid_mask" &&
        input.generator != "typed")
      throw std::invalid_argument("unknown recipe generator: " + input.generator);
    if ((input.generator == "iq_cube" && input.dtype != "complex64") ||
        (input.generator == "power_map" && input.dtype != "float32") ||
        (input.generator == "cluster_blobs" && input.dtype != "float32") ||
        (input.generator == "rank_one_mix" && input.dtype != "complex64") ||
        (input.generator == "zero_float" && input.dtype != "float32") ||
        (input.generator == "cluster_labels" && input.dtype != "int64") ||
        (input.generator == "reduction_indices" && input.dtype != "int64") ||
        (input.generator == "reduction_points" && input.dtype != "float32") ||
        ((input.generator == "axis_ramp" || input.generator == "centered_axis_ramp") &&
         input.dtype != "float32") ||
        (input.generator == "in_bounds_rd_indices" && input.dtype != "int64") ||
        (input.generator == "valid_mask" && input.dtype != "uint8" && input.dtype != "bool"))
      throw std::invalid_argument("recipe generator/dtype mismatch: " + input.port);
    for (const auto& axis : input.omit_if_one)
      if (std::find(input.shape.begin(), input.shape.end(), axis) == input.shape.end())
        throw std::invalid_argument("omit_if_one names absent input axis: " + axis);
    result.inputs.push_back(std::move(input));
  }
  result.constructor = root["constructor"];
  require_keys(result.constructor, {"extents", "bounds_policy", "features"}, "recipe constructor");
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
  if (root["extension"]) {
    result.extension = root["extension"];
    if (!result.extension.IsMap())
      throw std::invalid_argument("recipe extension must be a map");
  }
  if (root["derived_axes"]) {
    result.derived_axes = root["derived_axes"];
    if (!result.derived_axes.IsMap())
      throw std::invalid_argument("recipe derived_axes must be a map");
    for (const auto& item : result.derived_axes) {
      const auto name = item.first.as<std::string>();
      if (name.empty())
        throw std::invalid_argument("recipe derived axis name is empty");
      require_keys(item.second, {"product", "rounded_percent"}, "recipe derived axis");
      if (static_cast<bool>(item.second["product"]) ==
          static_cast<bool>(item.second["rounded_percent"]))
        throw std::invalid_argument("recipe derived axis needs one sizing rule: " + name);
      if (item.second["product"]) {
        (void)recipe_strings(item.second["product"], "recipe derived product");
      } else {
        require_keys(item.second["rounded_percent"], {"axis", "percent", "minimum"},
                     "recipe rounded_percent");
        if (!item.second["rounded_percent"]["axis"] || !item.second["rounded_percent"]["percent"])
          throw std::invalid_argument("rounded_percent needs axis and percent");
      }
    }
  }
  const auto output = root["outputs"];
  require_keys(output,
               {"timed", "timed_pair", "drain", "drain_codecs", "unwired", "queue_depth",
                "optional_tensor_outputs"},
               "recipe outputs");
  if (!output["timed"])
    throw std::invalid_argument("recipe outputs requires timed");
  result.outputs.timed = output["timed"].as<std::string>();
  if (output["timed_pair"])
    result.outputs.timed_pair = output["timed_pair"].as<std::string>();
  if (output["queue_depth"]) {
    result.outputs.queue_depth = output["queue_depth"].as<unsigned int>();
    if (result.outputs.queue_depth == 0)
      throw std::invalid_argument("recipe queue_depth must be positive");
  }
  if (output["drain"])
    result.outputs.drain = recipe_strings(output["drain"], "recipe drained outputs");
  if (output["drain_codecs"]) {
    const auto codecs = output["drain_codecs"];
    if (!codecs.IsMap())
      throw std::invalid_argument("recipe drain_codecs must be a map");
    for (const auto& item : codecs) {
      const auto port = item.first.as<std::string>();
      const auto codec = item.second.as<std::string>();
      if (std::find(result.outputs.drain.begin(), result.outputs.drain.end(), port) ==
              result.outputs.drain.end() ||
          codec != "array_geometry")
        throw std::invalid_argument("unsupported recipe drain codec: " + port);
      result.outputs.drain_codecs.emplace(port, codec);
    }
  }
  if (output["unwired"])
    result.outputs.unwired = recipe_strings(output["unwired"], "recipe unwired outputs");
  std::set<std::string> output_names{result.outputs.timed};
  if (!result.outputs.timed_pair.empty() && !output_names.insert(result.outputs.timed_pair).second)
    throw std::invalid_argument("duplicate recipe timed output");
  for (const auto& name : result.outputs.drain)
    if (!output_names.insert(name).second)
      throw std::invalid_argument("duplicate recipe output: " + name);
  for (const auto& name : result.outputs.unwired)
    if (!output_names.insert(name).second)
      throw std::invalid_argument("duplicate recipe output: " + name);
  if (output["optional_tensor_outputs"]) {
    result.optional_tensor_outputs =
        recipe_strings(output["optional_tensor_outputs"], "optional Tensor outputs");
    for (const auto& port : result.optional_tensor_outputs) {
      if (!output_names.contains(port) || port == result.outputs.timed ||
          result.outputs.drain_codecs.contains(port))
        throw std::invalid_argument("invalid optional Tensor output: " + port);
    }
  }
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
  result.operator_type = precision_operator(result.operator_type);
  for (auto& input : result.inputs)
    input.dtype = precision_dtype(input.dtype);
  return result;
}

// Case wiring varies optional Tensor sinks while retaining causal descriptor
// drains and required outputs. Base manifests still validate every port.
inline YamlRecipe case_recipe(const ProfileSuite& suite, const ProfileCase& profile_case) {
  auto recipe = parse_yaml_recipe(suite);
  if (profile_case.output_mode == "all") {
    recipe.outputs.drain.insert(recipe.outputs.drain.end(), recipe.outputs.unwired.begin(),
                                recipe.outputs.unwired.end());
    recipe.outputs.unwired.clear();
  } else if (profile_case.output_mode == "minimal") {
    std::vector<std::string> retained;
    for (const auto& port : recipe.outputs.drain) {
      if (std::find(recipe.optional_tensor_outputs.begin(), recipe.optional_tensor_outputs.end(),
                    port) == recipe.optional_tensor_outputs.end())
        retained.push_back(port);
      else
        recipe.outputs.unwired.push_back(port);
    }
    recipe.outputs.drain = std::move(retained);
    const auto& pair = recipe.outputs.timed_pair;
    if (!pair.empty() &&
        std::find(recipe.optional_tensor_outputs.begin(), recipe.optional_tensor_outputs.end(),
                  pair) != recipe.optional_tensor_outputs.end()) {
      recipe.outputs.unwired.push_back(pair);
      recipe.outputs.timed_pair.clear();
    }
  }
  return recipe;
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
        recipe.inputs[i].dtype != precision_dtype(expected_inputs[i].dtype) ||
        recipe.inputs[i].memory != expected_inputs[i].memory ||
        recipe.inputs[i].generator != expected_inputs[i].generator ||
        recipe.inputs[i].index_doppler_divisor != expected_inputs[i].index_doppler_divisor)
      throw std::invalid_argument("recipe data input disagrees with locked contract: " +
                                  recipe.inputs[i].port);
  }
  if (recipe.outputs.timed != expected_outputs.timed ||
      recipe.outputs.drain != expected_outputs.drain ||
      recipe.outputs.unwired != expected_outputs.unwired ||
      recipe.outputs.queue_depth != expected_outputs.queue_depth ||
      recipe.outputs.timed_pair != expected_outputs.timed_pair ||
      recipe.outputs.drain_codecs != expected_outputs.drain_codecs)
    throw std::invalid_argument("recipe output activation disagrees with locked contract");
  if (recipe.live_ports != expected_live)
    throw std::invalid_argument("recipe live port mapping disagrees with locked contract");
}

inline YAML::Node merge_recipe_map(const YAML::Node& base, const YAML::Node& change) {
  if (!base.IsMap() || !change.IsMap())
    throw std::invalid_argument("recipe values must be maps");
  YAML::Node result(YAML::NodeType::Map);
  for (const auto& item : base) {
    const auto name = item.first.as<std::string>();
    const auto replacement = change[name];
    if (!replacement)
      result[name] = YAML::Clone(item.second);
    else if (item.second.IsMap() && replacement.IsMap())
      result[name] = merge_recipe_map(item.second, replacement);
    else
      result[name] = YAML::Clone(replacement);
  }
  for (const auto& item : change) {
    const auto name = item.first.as<std::string>();
    if (!base[name])
      result[name] = YAML::Clone(item.second);
  }
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
  const auto derived = suite.recipe["derived_axes"];
  if (derived) {
    for (const auto& item : derived) {
      const auto name = item.first.as<std::string>();
      if (result.contains(name))
        throw std::invalid_argument("duplicate recipe derived axis: " + name);
      if (item.second["product"]) {
        int64_t product = 1;
        for (const auto& term : item.second["product"]) {
          const auto axis = term.as<std::string>();
          const auto found = result.find(axis);
          if (found == result.end() || found->second <= 0 ||
              product > std::numeric_limits<int64_t>::max() / found->second)
            throw std::invalid_argument("invalid recipe derived product: " + name);
          product *= found->second;
        }
        result.emplace(name, product);
      } else {
        const auto rule = item.second["rounded_percent"];
        const auto axis = rule["axis"].as<std::string>();
        const auto percent_name = rule["percent"].as<std::string>();
        const auto count = result.at(axis);
        const auto percent = result.at(percent_name);
        const auto minimum = rule["minimum"] ? rule["minimum"].as<int64_t>() : 0;
        if (percent < 0 || percent > 100 || count < 0 ||
            count > (std::numeric_limits<int64_t>::max() - 50) / std::max<int64_t>(percent, 1))
          throw std::invalid_argument("invalid recipe rounded percent: " + name);
        result.emplace(name, std::max<int64_t>(minimum, (count * percent + 50) / 100));
      }
    }
  }
  for (const auto& [name, value] : result)
    if (value < 0)
      throw std::invalid_argument("negative recipe axis or fixture: " + name);
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

template <typename T>
std::shared_ptr<holoscan::Tensor> generate_precision_input(
    const RecipeInput& input, const std::map<std::string, int64_t>& values) {
  const auto shape = recipe_shape(input, values);
  if (input.generator == "iq_cube" && input.dtype == "complex64")
    return make_iq_cube_device<T>(shape);
  if (input.generator == "power_map" && input.dtype == "float32")
    return make_power_map_device<T>(shape);
  if ((input.generator == "axis_ramp" || input.generator == "centered_axis_ramp") &&
      input.dtype == "float32") {
    if (shape.size() != 1)
      throw std::invalid_argument("axis_ramp requires one dimension");
    std::vector<T> host(static_cast<size_t>(shape[0]));
    for (int64_t i = 0; i < shape[0]; ++i)
      host[static_cast<size_t>(i)] =
          static_cast<T>(input.generator == "centered_axis_ramp" ? i - shape[0] / 2 : i);
    auto tensor = make_device_tensor(shape, real_dtype<T>());
    auto data = tensor->mutable_data();
    if (!host.empty() && (!data || cudaMemcpy(data.value(), host.data(), sizeof(T) * host.size(),
                                              cudaMemcpyHostToDevice) != cudaSuccess))
      throw std::runtime_error("cannot stage axis ramp");
    return tensor;
  }
  if (input.generator == "valid_mask" && (input.dtype == "bool" || input.dtype == "uint8")) {
    const auto count =
        std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
    std::vector<uint8_t> host(count, 1U);
    const DLDataType dtype{static_cast<uint8_t>(input.dtype == "bool" ? kDLBool : kDLUInt), 8U, 1U};
    auto tensor = make_device_tensor(shape, dtype);
    auto data = tensor->mutable_data();
    if (!data ||
        cudaMemcpy(data.value(), host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage validity mask");
    return tensor;
  }
  if (input.generator == "cluster_labels" && input.dtype == "int64") {
    if (shape.size() != 1 || !values.contains("clusters"))
      throw std::invalid_argument("cluster_labels requires [points] and clusters axis");
    std::vector<int64_t> host(static_cast<size_t>(shape[0]));
    for (int64_t i = 0; i < shape[0]; ++i)
      host[static_cast<size_t>(i)] = i % 8 == 0 ? -1 : i % values.at("clusters");
    auto tensor = make_device_tensor(shape, DLDataType{kDLInt, 64, 1});
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(int64_t) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage cluster labels");
    return tensor;
  }
  if (input.generator == "reduction_indices" && input.dtype == "int64") {
    if (shape.size() != 2 || shape[1] != 2)
      throw std::invalid_argument("reduction_indices requires [points, 2]");
    std::vector<int64_t> host(static_cast<size_t>(shape[0] * 2));
    for (int64_t i = 0; i < shape[0]; ++i) {
      host[static_cast<size_t>(i * 2)] = i % 1024;
      host[static_cast<size_t>(i * 2 + 1)] = (i * 3) % 2048;
    }
    auto tensor = make_device_tensor(shape, DLDataType{kDLInt, 64, 1});
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(int64_t) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage reduction indices");
    return tensor;
  }
  if (input.generator == "reduction_points" && input.dtype == "float32") {
    if (shape.size() != 2)
      throw std::invalid_argument("reduction_points requires [points, features]");
    std::vector<T> host(static_cast<size_t>(shape[0] * shape[1]));
    for (int64_t i = 0; i < shape[0]; ++i)
      for (int64_t f = 0; f < shape[1]; ++f)
        host[static_cast<size_t>(i * shape[1] + f)] =
            static_cast<T>((i * 17 + f * 3) % 100) * 0.01F;
    auto tensor = make_device_tensor(shape, real_dtype<T>());
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(T) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage reduction points");
    return tensor;
  }
  if (input.generator == "zero_float" && input.dtype == "float32") {
    const auto count =
        std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
    std::vector<T> host(static_cast<size_t>(count), 0.0F);
    auto tensor = make_device_tensor(shape, real_dtype<T>());
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(T) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage zero float input");
    return tensor;
  }
  if (input.generator == "rank_one_mix" && input.dtype == "complex64") {
    if (shape.size() != 3 || !values.contains("selection_percent"))
      throw std::invalid_argument(
          "rank_one_mix requires [items, channels, looks] and selection_percent");
    const auto selected = static_cast<int64_t>(
        std::llround(static_cast<double>(shape[0] * values.at("selection_percent")) / 100.0));
    std::vector<std::complex<T>> host(static_cast<size_t>(shape[0] * shape[1] * shape[2]));
    for (int64_t item = 0; item < shape[0]; ++item) {
      for (int64_t channel = 0; channel < shape[1]; ++channel) {
        const double phase =
            2.0 * 3.14159265358979323846 * 0.5 * static_cast<double>(channel) * std::sin(0.25);
        const auto secondary =
            std::complex<T>(static_cast<T>(std::cos(phase)), static_cast<T>(std::sin(phase)));
        const auto value =
            std::complex<T>(1.0F, 0.0F) + (item < selected ? secondary : std::complex<T>{});
        for (int64_t look = 0; look < shape[2]; ++look)
          host[static_cast<size_t>((item * shape[1] + channel) * shape[2] + look)] = value;
      }
    }
    auto tensor = make_device_tensor(shape, complex_dtype<T>());
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(std::complex<T>) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage rank-one snapshot input");
    return tensor;
  }
  if (input.generator == "cluster_blobs" && input.dtype == "float32") {
    if (shape.size() != 2 || !values.contains("clusters"))
      throw std::invalid_argument("cluster_blobs requires [points, features] and clusters axis");
    const auto clusters = values.at("clusters");
    std::vector<T> host(static_cast<size_t>(shape[0] * shape[1]));
    for (int64_t i = 0; i < shape[0]; ++i) {
      for (int64_t f = 0; f < shape[1]; ++f) {
        const double base = f == 0 ? static_cast<double>(i % clusters) * 100.0 : 0.0;
        const double jitter = static_cast<double>(((i * 131 + f * 17) % 21) - 10) / 100.0;
        host[static_cast<size_t>(i * shape[1] + f)] = static_cast<T>(base + jitter);
      }
    }
    auto tensor = make_device_tensor(shape, real_dtype<T>());
    auto data = tensor->mutable_data();
    if (!data || cudaMemcpy(data.value(), host.data(), sizeof(T) * host.size(),
                            cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("cannot stage cluster blob points");
    return tensor;
  }
  if (input.generator == "in_bounds_rd_indices" && input.dtype == "int64") {
    if (shape.size() != 2 || shape[1] != 2)
      throw std::invalid_argument("in_bounds_rd_indices requires [detections, 2]");
    auto doppler = values.at("doppler");
    if (!input.index_doppler_divisor.empty()) {
      const auto divisor = values.at(input.index_doppler_divisor);
      if (divisor <= 0 || doppler % divisor != 0)
        throw std::invalid_argument("detection Doppler axis is not divisible by " +
                                    input.index_doppler_divisor);
      doppler /= divisor;
    }
    const auto range = values.at("range");
    std::vector<int64_t> host(static_cast<size_t>(shape[0] * 2));
    for (int64_t i = 0; i < shape[0]; ++i) {
      const auto cell = (i * 997) % (doppler * range);
      host[static_cast<size_t>(i * 2)] = cell / range;
      host[static_cast<size_t>(i * 2 + 1)] = cell % range;
    }
    auto tensor = make_device_tensor(shape, DLDataType{kDLInt, 64, 1});
    auto data = tensor->mutable_data();
    if (!host.empty() &&
        (!data || cudaMemcpy(data.value(), host.data(), sizeof(int64_t) * host.size(),
                             cudaMemcpyHostToDevice) != cudaSuccess))
      throw std::runtime_error("cannot stage in-bounds detection indices");
    return tensor;
  }
  throw std::invalid_argument("generator/dtype mismatch at input " + input.port);
}

inline std::shared_ptr<holoscan::Tensor> generate_recipe_input(
    const RecipeInput& input, const std::map<std::string, int64_t>& values) {
  auto canonical = input;
  if (input.dtype == "float64" || input.dtype == "complex128") {
    canonical.dtype = input.dtype == "float64" ? "float32" : "complex64";
    return generate_precision_input<double>(canonical, values);
  }
  return generate_precision_input<float>(canonical, values);
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
    for (const auto extent : shape)
      if (extent < 0)
        throw std::invalid_argument("negative recipe input shape at " + input.port);
    if (input.generator == "typed")
      continue;
    if (input.generator == "in_bounds_rd_indices" &&
        (shape.size() != 2 || shape[1] != 2 || !values.contains("doppler") ||
         !values.contains("range")))
      throw std::invalid_argument(
          "in_bounds_rd_indices requires [detections, 2] and doppler/range axes");
    if (!input.index_doppler_divisor.empty()) {
      const auto divisor = values.at(input.index_doppler_divisor);
      if (divisor <= 0 || values.at("doppler") % divisor != 0)
        throw std::invalid_argument("detection Doppler axis is not divisible by " +
                                    input.index_doppler_divisor);
    }
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
