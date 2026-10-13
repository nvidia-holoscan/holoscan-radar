// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan::radar {

struct CfarCellWindowOverride {
  std::int64_t doppler{0};
  std::int64_t range{0};
};

struct SpatialSmoothingOverride {
  std::int64_t count{0};
  std::int64_t first{0};
  std::int64_t second{0};
};

static_assert(sizeof(CfarCellWindowOverride) == 16U);
static_assert(alignof(CfarCellWindowOverride) == 8U);
static_assert(std::is_standard_layout_v<CfarCellWindowOverride>);
static_assert(std::is_trivially_copyable_v<CfarCellWindowOverride>);
static_assert(sizeof(SpatialSmoothingOverride) == 24U);
static_assert(alignof(SpatialSmoothingOverride) == 8U);
static_assert(std::is_standard_layout_v<SpatialSmoothingOverride>);
static_assert(std::is_trivially_copyable_v<SpatialSmoothingOverride>);

inline expected<std::vector<std::int64_t>, Error> from_wire(const CfarCellWindowOverride& source) {
  return std::vector<std::int64_t>{source.doppler, source.range};
}

inline expected<std::vector<std::int64_t>, Error> from_wire(
    const SpatialSmoothingOverride& source) {
  if (source.count < 0 || source.count > 2) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "spatial_smoothing wire count must be 0, 1, or 2"});
  }
  if (source.count == 0) {
    return std::vector<std::int64_t>{};
  }
  if (source.count == 1) {
    return std::vector<std::int64_t>{source.first};
  }
  return std::vector<std::int64_t>{source.first, source.second};
}

}  // namespace holoscan::radar

namespace holoscan {

template <>
struct ChannelTypeIdentity<radar::CfarCellWindowOverride> {
  static constexpr std::string_view name = "com.nvidia.holoscan.radar.CfarCellWindowOverride";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<radar::CfarCellWindowOverride> {
  static constexpr auto fields =
      pod_fields(pod_field("doppler", &radar::CfarCellWindowOverride::doppler,
                           offsetof(radar::CfarCellWindowOverride, doppler)),
                 pod_field("range", &radar::CfarCellWindowOverride::range,
                           offsetof(radar::CfarCellWindowOverride, range)));
};

template <>
struct ChannelTypeIdentity<radar::SpatialSmoothingOverride> {
  static constexpr std::string_view name = "com.nvidia.holoscan.radar.SpatialSmoothingOverride";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<radar::SpatialSmoothingOverride> {
  static constexpr auto fields =
      pod_fields(pod_field("count", &radar::SpatialSmoothingOverride::count,
                           offsetof(radar::SpatialSmoothingOverride, count)),
                 pod_field("first", &radar::SpatialSmoothingOverride::first,
                           offsetof(radar::SpatialSmoothingOverride, first)),
                 pod_field("second", &radar::SpatialSmoothingOverride::second,
                           offsetof(radar::SpatialSmoothingOverride, second)));
};

}  // namespace holoscan
