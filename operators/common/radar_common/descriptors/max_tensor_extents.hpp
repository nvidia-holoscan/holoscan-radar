// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace holoscan::radar {

// Graph-build maxima for HSDK 5 Tensor output pool sizing (ctor capture on
// fat-output operators). Field names are the module contract; every dimension
// is max_-prefixed. Optional/absent fields are allowed — each operator
// documents which fields it requires. Not one of the seven shared descriptors
// (those ride Params / [live] ports); this type is ctor-only and never a
// Tensor port payload. See docs/planning/MAX_TENSOR_EXTENTS.md.
//
// No opaque leading-batch field: multi-item throughput is an app concern.
struct MaxTensorExtents {
  std::optional<std::int64_t> max_samples_per_chirp;
  std::optional<std::int64_t> max_chirps;
  std::optional<std::int64_t> max_range_bins;
  std::optional<std::int64_t> max_doppler_bins;
  std::optional<std::int64_t> max_rx;
  std::optional<std::int64_t> max_tx;
  std::optional<std::int64_t> max_interleaved_chirps;
  std::optional<std::int64_t> max_virtual_channels;
  std::optional<std::int64_t> max_channels;
  std::optional<std::int64_t> max_az;
  std::optional<std::int64_t> max_el;
  std::optional<std::int64_t> max_detections;
  std::optional<std::int64_t> max_points;
  std::optional<std::int64_t> max_items;
  std::optional<std::int64_t> max_snapshots;
  std::optional<std::int64_t> max_features;
  std::optional<std::int64_t> max_sources;
  std::optional<std::int64_t> max_clusters;
  std::optional<std::int64_t> max_subarray_channels;

  friend bool operator==(const MaxTensorExtents& a,
                         const MaxTensorExtents& b) = default;
};

// Return *field or throw naming the operator and field (fail-closed sizing).
inline std::int64_t require_extent(const std::optional<std::int64_t>& field,
                                   const char* field_name,
                                   const char* who) {
  if (!field.has_value() || *field <= 0) {
    throw std::invalid_argument(
        std::string(who) + ": MaxTensorExtents." + field_name +
        " must be a positive value (required for Tensor pool sizing)");
  }
  return *field;
}

}  // namespace holoscan::radar
