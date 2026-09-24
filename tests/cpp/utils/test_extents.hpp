// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar::testing {

inline MaxTensorExtents default_test_extents() {
  MaxTensorExtents extents;
  extents.max_samples_per_chirp = 512;
  extents.max_chirps = 256;
  extents.max_range_bins = 512;
  extents.max_doppler_bins = 256;
  extents.max_rx = 8;
  extents.max_tx = 4;
  extents.max_interleaved_chirps = 256;
  extents.max_virtual_channels = 64;
  extents.max_channels = 64;
  extents.max_az = 64;
  extents.max_el = 32;
  extents.max_detections = 4096;
  extents.max_points = 4096;
  extents.max_items = 4096;
  extents.max_snapshots = 16;
  extents.max_features = 16;
  extents.max_sources = 8;
  extents.max_clusters = 256;
  extents.max_subarray_channels = 32;
  return extents;
}

inline MaxTensorExtents extents_for_scan_grid(std::size_t num_az, std::size_t num_el,
                                              std::optional<std::size_t> num_doppler = std::nullopt,
                                              std::optional<std::size_t> num_range = std::nullopt) {
  const auto at_least_one = [](std::size_t value) {
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(value));
  };
  MaxTensorExtents extents = default_test_extents();
  extents.max_az = at_least_one(num_az);
  extents.max_el = at_least_one(num_el);
  if (num_doppler) {
    extents.max_doppler_bins = at_least_one(*num_doppler);
  }
  if (num_range) {
    extents.max_range_bins = at_least_one(*num_range);
  }
  return extents;
}

}  // namespace holoscan::radar::testing
