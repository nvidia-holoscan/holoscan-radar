// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/dsp/angle_convention.hpp"

namespace holoscan::radar::app {

struct PhysicalAngles {
  double az{0.0};
  double el{0.0};
};

struct PhysicalAngleBounds {
  std::array<double, 2> az{};
  std::array<double, 2> el{};
};

class PhysicalAngleCentroid {
 public:
  void add(const AngleConvention& convention, double first, double second) {
    validate_scan_grid_axes(convention, {first}, {second},
                            "physical angle centroid");
    if (convention.uses_direction_cosine() &&
        !is_physical_direction_cosine(first, second)) {
      throw std::invalid_argument(
          "physical angle centroid: direction-cosine pair lies outside the "
          "unit disk");
    }
    const CanonicalDirection direction =
        boundary_to_canonical(convention, first, second);
    sum_.x += direction.x;
    sum_.y += direction.y;
    sum_.z += direction.z;
    ++count_;
  }

  [[nodiscard]] PhysicalAngles mean() const {
    const double norm = std::sqrt(sum_.x * sum_.x + sum_.y * sum_.y +
                                  sum_.z * sum_.z);
    if (count_ == 0U || !std::isfinite(norm) || norm <= 1e-12) {
      throw std::invalid_argument(
          "physical angle centroid: mean direction is undefined");
    }
    const CanonicalDirection direction{
        sum_.x / norm, sum_.y / norm, sum_.z / norm};
    PhysicalAngles result;
    canonical_to_boundary(AngleConvention{}, direction, &result.az, &result.el);
    return result;
  }

 private:
  CanonicalDirection sum_{0.0, 0.0, 0.0};
  std::size_t count_{0U};
};

inline PhysicalAngles to_physical_angles(const AngleConvention& convention,
                                         double first, double second) {
  validate_scan_grid_axes(convention, {first}, {second},
                          "physical angle conversion");
  if (convention.uses_direction_cosine() &&
      !is_physical_direction_cosine(first, second)) {
    throw std::invalid_argument(
        "physical angle conversion: direction-cosine pair lies outside the "
        "unit disk");
  }
  const CanonicalDirection direction =
      boundary_to_canonical(convention, first, second);
  PhysicalAngles result;
  canonical_to_boundary(AngleConvention{}, direction, &result.az, &result.el);
  if (!std::isfinite(result.az) || !std::isfinite(result.el)) {
    throw std::invalid_argument(
        "physical angle conversion: result must be finite");
  }
  return result;
}

inline PhysicalAngleBounds physical_scan_grid_bounds(
    const AngleConvention& convention, const std::vector<double>& first_axis,
    const std::vector<double>& second_axis) {
  if (first_axis.empty() || second_axis.empty()) {
    throw std::invalid_argument(
        "physical scan-grid bounds: axes must be non-empty");
  }
  validate_scan_grid_axes(convention, first_axis, second_axis,
                          "physical scan-grid bounds");

  PhysicalAngleBounds result{
      .az = {std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()},
      .el = {std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()}};
  bool any = false;
  for (const double first : first_axis) {
    for (const double second : second_axis) {
      if (convention.uses_direction_cosine() &&
          !is_physical_direction_cosine(first, second)) {
        continue;
      }
      const PhysicalAngles angles =
          to_physical_angles(convention, first, second);
      result.az[0] = std::min(result.az[0], angles.az);
      result.az[1] = std::max(result.az[1], angles.az);
      result.el[0] = std::min(result.el[0], angles.el);
      result.el[1] = std::max(result.el[1], angles.el);
      any = true;
    }
  }
  if (!any) {
    throw std::invalid_argument(
        "physical scan-grid bounds: grid has no physical direction");
  }
  return result;
}

}  // namespace holoscan::radar::app
