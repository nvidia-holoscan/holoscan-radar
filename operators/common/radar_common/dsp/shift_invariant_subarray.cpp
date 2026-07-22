// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/shift_invariant_subarray.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {
namespace {

std::array<double, 3> position(const ArrayGeometry& geometry, int64_t index) {
  const size_t base = static_cast<size_t>(index * 3);
  return {
      geometry.element_positions[base],
      geometry.element_positions[base + 1],
      geometry.element_positions[base + 2],
  };
}

double norm(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

double geometry_tolerance(const ArrayGeometry& geometry) {
  double scale = geometry.wavelength;
  for (double coordinate : geometry.element_positions) {
    scale = std::max(scale, std::abs(coordinate));
  }
  return 1e-6 * scale;
}

ShiftInvariantSubarrayPair derive_ula_pair(const ArrayGeometry& geometry) {
  const int64_t channels = geometry.num_elements();
  if (channels < 2) {
    throw std::invalid_argument(
        "ShiftInvariantSubarrayPair auto: array must have at least two "
        "elements");
  }
  if (geometry.geometry_type &&
      *geometry.geometry_type != ArrayGeometryType::kUla) {
    throw std::invalid_argument(
        "ShiftInvariantSubarrayPair auto: array_geometry must be a ULA");
  }

  const auto first = position(geometry, 0);
  const auto second = position(geometry, 1);
  const std::array<double, 3> delta{
      second[0] - first[0],
      second[1] - first[1],
      second[2] - first[2],
  };
  const double tolerance = geometry_tolerance(geometry);
  for (int64_t channel = 0; channel < channels; ++channel) {
    const auto actual = position(geometry, channel);
    for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
      const double expected =
          first[coordinate] + static_cast<double>(channel) * delta[coordinate];
      if (std::abs(actual[coordinate] - expected) > tolerance) {
        throw std::invalid_argument(
            "ShiftInvariantSubarrayPair auto: array_geometry must be a "
            "uniformly spaced line in snapshot channel order");
      }
    }
  }

  ShiftInvariantSubarrayPair pair;
  pair.reference_indices.reserve(static_cast<size_t>(channels - 1));
  pair.shifted_indices.reserve(static_cast<size_t>(channels - 1));
  for (int64_t channel = 0; channel < channels - 1; ++channel) {
    pair.reference_indices.push_back(channel);
    pair.shifted_indices.push_back(channel + 1);
  }
  return pair;
}

}  // namespace

ShiftInvariantSubarrayPlan build_shift_invariant_subarray_plan(
    const ArrayGeometry& geometry,
    const ShiftInvariantSubarrayPair& requested) {
  geometry.validate();
  ShiftInvariantSubarrayPlan plan;
  plan.pair = requested.is_auto() ? derive_ula_pair(geometry) : requested;
  plan.pair.validate(geometry.num_elements());

  const auto reference = position(geometry, plan.pair.reference_indices[0]);
  const auto shifted = position(geometry, plan.pair.shifted_indices[0]);
  plan.displacement = {
      shifted[0] - reference[0],
      shifted[1] - reference[1],
      shifted[2] - reference[2],
  };
  plan.spacing = norm(plan.displacement);
  if (!(plan.spacing > 0.0) || !std::isfinite(plan.spacing)) {
    throw std::invalid_argument(
        "ShiftInvariantSubarrayPair: common displacement must be nonzero and "
        "finite");
  }

  const double tolerance = geometry_tolerance(geometry);
  for (size_t index = 0; index < plan.pair.reference_indices.size(); ++index) {
    const auto a = position(geometry, plan.pair.reference_indices[index]);
    const auto b = position(geometry, plan.pair.shifted_indices[index]);
    for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
      if (std::abs((b[coordinate] - a[coordinate]) -
                   plan.displacement[coordinate]) > tolerance) {
        throw std::invalid_argument(
            "ShiftInvariantSubarrayPair: corresponding elements must share "
            "one common displacement");
      }
    }
  }

  for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
    plan.axis[coordinate] = plan.displacement[coordinate] / plan.spacing;
  }
  const double projection = plan.axis[0];
  plan.broadside = {
      1.0 - projection * plan.axis[0],
      -projection * plan.axis[1],
      -projection * plan.axis[2],
  };
  const double broadside_norm = norm(plan.broadside);
  if (!(broadside_norm > 1e-8)) {
    throw std::invalid_argument(
        "ShiftInvariantSubarrayPair: displacement parallel to canonical "
        "boresight has no unique forward-hemisphere 1-D DOA embedding");
  }
  for (double& coordinate : plan.broadside) {
    coordinate /= broadside_norm;
  }
  return plan;
}

}  // namespace holoscan::radar
