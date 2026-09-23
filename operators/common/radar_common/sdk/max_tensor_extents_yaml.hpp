// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "radar_common/descriptors/max_tensor_extents.hpp"

// YAML helpers for MaxTensorExtents. Lives here (not in descriptors/) so the
// descriptor layer stays yaml-cpp-free. Apps call from_yaml; operators stay
// yaml-free and take MaxTensorExtents by ctor capture.
//
// Keys under the node must match struct field names (max_channels, …).
// Parent/group keys that select which node to pass are app-local.

namespace holoscan::radar {
namespace detail {

inline std::optional<std::int64_t> optional_positive_i64(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    return std::nullopt;
  }
  const auto value = node[key].as<std::int64_t>();
  if (value <= 0) {
    throw std::invalid_argument(std::string("MaxTensorExtents: '") + key +
                                "' must be positive, got " + std::to_string(value));
  }
  return value;
}

}  // namespace detail

inline MaxTensorExtents MaxTensorExtentsFromYaml(const YAML::Node& node) {
  if (!node || !node.IsMap()) {
    throw std::invalid_argument("MaxTensorExtents: from_yaml expects a YAML map of max_* fields");
  }
  MaxTensorExtents extents;
  extents.max_samples_per_chirp = detail::optional_positive_i64(node, "max_samples_per_chirp");
  extents.max_chirps = detail::optional_positive_i64(node, "max_chirps");
  extents.max_range_bins = detail::optional_positive_i64(node, "max_range_bins");
  extents.max_doppler_bins = detail::optional_positive_i64(node, "max_doppler_bins");
  extents.max_rx = detail::optional_positive_i64(node, "max_rx");
  extents.max_tx = detail::optional_positive_i64(node, "max_tx");
  extents.max_interleaved_chirps = detail::optional_positive_i64(node, "max_interleaved_chirps");
  extents.max_virtual_channels = detail::optional_positive_i64(node, "max_virtual_channels");
  extents.max_channels = detail::optional_positive_i64(node, "max_channels");
  extents.max_az = detail::optional_positive_i64(node, "max_az");
  extents.max_el = detail::optional_positive_i64(node, "max_el");
  extents.max_detections = detail::optional_positive_i64(node, "max_detections");
  extents.max_points = detail::optional_positive_i64(node, "max_points");
  extents.max_items = detail::optional_positive_i64(node, "max_items");
  extents.max_snapshots = detail::optional_positive_i64(node, "max_snapshots");
  extents.max_features = detail::optional_positive_i64(node, "max_features");
  extents.max_sources = detail::optional_positive_i64(node, "max_sources");
  extents.max_clusters = detail::optional_positive_i64(node, "max_clusters");
  extents.max_subarray_channels = detail::optional_positive_i64(node, "max_subarray_channels");
  return extents;
}

}  // namespace holoscan::radar

namespace YAML {

template <>
struct convert<holoscan::radar::MaxTensorExtents> {
  static Node encode(const holoscan::radar::MaxTensorExtents& rhs) {
    Node node;
    auto put = [&](const char* key, const std::optional<std::int64_t>& v) {
      if (v.has_value()) {
        node[key] = *v;
      }
    };
    put("max_samples_per_chirp", rhs.max_samples_per_chirp);
    put("max_chirps", rhs.max_chirps);
    put("max_range_bins", rhs.max_range_bins);
    put("max_doppler_bins", rhs.max_doppler_bins);
    put("max_rx", rhs.max_rx);
    put("max_tx", rhs.max_tx);
    put("max_interleaved_chirps", rhs.max_interleaved_chirps);
    put("max_virtual_channels", rhs.max_virtual_channels);
    put("max_channels", rhs.max_channels);
    put("max_az", rhs.max_az);
    put("max_el", rhs.max_el);
    put("max_detections", rhs.max_detections);
    put("max_points", rhs.max_points);
    put("max_items", rhs.max_items);
    put("max_snapshots", rhs.max_snapshots);
    put("max_features", rhs.max_features);
    put("max_sources", rhs.max_sources);
    put("max_clusters", rhs.max_clusters);
    put("max_subarray_channels", rhs.max_subarray_channels);
    return node;
  }

  static bool decode(const Node& node, holoscan::radar::MaxTensorExtents& rhs) {
    rhs = holoscan::radar::MaxTensorExtentsFromYaml(node);
    return true;
  }
};

}  // namespace YAML
