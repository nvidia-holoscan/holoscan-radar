// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/sdk/yaml_converters.hpp"

namespace holoscan::radar::sar_yaml_detail {
template <typename T, size_t N>
std::array<T, N> fixed(const YAML::Node& node, const char* name) {
  if (!node || !node.IsSequence() || node.size() != N)
    throw std::invalid_argument(std::string(name) + " needs " + std::to_string(N) + " values");
  std::array<T, N> out{};
  for (size_t i = 0; i < N; ++i)
    out[i] = node[i].as<T>();
  return out;
}
}  // namespace holoscan::radar::sar_yaml_detail

namespace YAML {
template <>
struct convert<holoscan::radar::SampleGeometry> {
  static Node encode(const holoscan::radar::SampleGeometry& geometry) {
    Node node;
    node["origin"] = std::vector<double>(geometry.origin().begin(), geometry.origin().end());
    node["axis_u"] = std::vector<double>(geometry.axis_u().begin(), geometry.axis_u().end());
    node["axis_v"] = std::vector<double>(geometry.axis_v().begin(), geometry.axis_v().end());
    node["spacing"] = std::vector<double>(geometry.spacing().begin(), geometry.spacing().end());
    node["counts"] = std::vector<int64_t>(geometry.counts().begin(), geometry.counts().end());
    return node;
  }
  static bool decode(const Node& node, holoscan::radar::SampleGeometry& geometry) {
    if (!node.IsMap())
      return false;
    using holoscan::radar::sar_yaml_detail::fixed;
    geometry = holoscan::radar::SampleGeometry(
        holoscan::radar::SampleGeometrySpec{.origin = fixed<double, 3>(node["origin"], "origin"),
                                            .axis_u = fixed<double, 3>(node["axis_u"], "axis_u"),
                                            .axis_v = fixed<double, 3>(node["axis_v"], "axis_v"),
                                            .spacing = fixed<double, 2>(node["spacing"], "spacing"),
                                            .counts = fixed<int64_t, 2>(node["counts"], "counts")});
    return true;
  }
};
}  // namespace YAML
