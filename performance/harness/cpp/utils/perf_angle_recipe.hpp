// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/perf_yaml_pilot.hpp"

namespace holoscan::radar::perf {

template <typename Params>
Params angle_recipe_params(const RecipeScenario& scenario) {
  Params params;
  params.array_geometry =
      recipe_ula_geometry(scenario.params["array_geometry"], scenario.axis("channels"));
  if (scenario.params["angle_convention"]["preset"].as<std::string>() != "radar_boresight")
    throw std::invalid_argument("angle recipe needs radar_boresight convention");
  params.angle_convention.preset = AngleConventionPreset::kRadarBoresight;
  if constexpr (requires { params.az_coordinates; })
    params.az_coordinates = recipe_coordinates(scenario.params["az_coordinates"], scenario.values);
  if constexpr (requires { params.el_coordinates; })
    params.el_coordinates = recipe_coordinates(scenario.params["el_coordinates"], scenario.values);
  if constexpr (requires { params.num_sources; })
    params.num_sources = scenario.params["num_sources"].as<int64_t>();
  if constexpr (requires { params.source_count_pfa; })
    params.source_count_pfa = scenario.params["source_count_pfa"].as<double>();
  if constexpr (requires { params.max_sources; })
    params.max_sources = scenario.params["max_sources"].as<int64_t>();
  if constexpr (requires { params.spatial_smoothing; })
    params.spatial_smoothing = scenario.params["spatial_smoothing"].as<std::vector<int64_t>>();
  if constexpr (requires { params.diagonal_loading; })
    params.diagonal_loading = scenario.params["diagonal_loading"].as<double>();
  if constexpr (requires { params.array_taper; })
    params.array_taper =
        recipe_window(recipe_window_type(scenario.params["array_taper"]["type"].as<std::string>()));
  return params;
}

template <typename Params>
void validate_angle_recipe(const RecipeScenario& scenario) {
  if (scenario.axis("channels") < 2)
    throw std::invalid_argument("angle recipe needs at least two channels");
  const auto params = angle_recipe_params<Params>(scenario);
  if constexpr (requires { params.num_sources; })
    if (params.num_sources != -1 && (params.num_sources < 1 || params.num_sources > 3))
      throw std::invalid_argument("angle recipe model order out of range");
  if constexpr (requires { params.max_sources; })
    if (params.max_sources < 1 || params.max_sources > 4)
      throw std::invalid_argument("angle recipe max_sources out of range");
  if constexpr (requires { params.source_count_pfa; })
    if (params.source_count_pfa <= 0.0 || params.source_count_pfa >= 1.0)
      throw std::invalid_argument("angle recipe source_count_pfa out of range");
  if constexpr (requires { params.diagonal_loading; })
    if (params.diagonal_loading < -1.0)
      throw std::invalid_argument("angle recipe diagonal_loading out of range");
  if (scenario.live_port == "array_geometry") {
    auto geometry = YAML::Clone(scenario.params["array_geometry"]);
    geometry["spacing_wavelengths"] = scenario.live_value["spacing_wavelengths"];
    (void)recipe_ula_geometry(geometry, scenario.axis("channels"));
  } else if (scenario.live_port == "az_coordinates" || scenario.live_port == "el_coordinates") {
    (void)recipe_coordinates(scenario.params[scenario.live_port], scenario.values,
                             scenario.live_value["shift"].as<double>());
  } else if (scenario.live_port == "spatial_smoothing") {
    const auto sizes = scenario.live_value.as<std::vector<int64_t>>();
    if (sizes.empty() || sizes.size() > 2)
      throw std::invalid_argument("angle smoothing needs one or two extents");
    for (auto size : sizes)
      if (size < 2 || size > scenario.axis("channels"))
        throw std::invalid_argument("angle smoothing extent out of bounds");
  } else if (scenario.live_port == "num_sources" || scenario.live_port == "max_sources") {
    if (scenario.live_value.as<int64_t>() < 1 || scenario.live_value.as<int64_t>() > 4)
      throw std::invalid_argument("angle live source count out of range");
  } else if (scenario.live_port == "source_count_pfa") {
    const auto value = scenario.live_value.as<double>();
    if (value <= 0.0 || value >= 1.0)
      throw std::invalid_argument("angle live PFA out of range");
  }
}

inline ResultRow angle_recipe_row(const RecipeScenario& scenario, std::string op_name,
                                  std::string variant) {
  ResultRow row;
  row.op_name = std::move(op_name);
  row.variant = std::move(variant);
  row.live_port = scenario.live_port;
  row.input_shape = recipe_shape(scenario.recipe.inputs.at(0), scenario.values);
  row.out_batch = scenario.axis("detections");
  row.k = scenario.axis("channels");
  if (scenario.values.contains("az") && scenario.values.contains("el"))
    row.map_shape = {scenario.axis("az"), scenario.axis("el")};
  return row;
}

}  // namespace holoscan::radar::perf
