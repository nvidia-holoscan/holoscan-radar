// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Virtual/physical antenna array for beamforming and DOA estimation. One of
// the module's six shared config-plane descriptor types.
struct ArrayGeometry {
  // Row-major (num_elements, 3) element positions in the array-local Cartesian
  // frame (meters).
  std::vector<double> element_positions{};
  double wavelength{0.0};
  std::optional<ArrayGeometryType> geometry_type{};

  int64_t num_elements() const {
    if (element_positions.size() % 3 != 0) {
      throw std::invalid_argument(
          "ArrayGeometry: element_positions length must be a multiple of 3");
    }
    return static_cast<int64_t>(element_positions.size() / 3);
  }

  void validate() const {
    const int64_t n = num_elements();
    if (n <= 0) {
      throw std::invalid_argument(
          "ArrayGeometry: num_elements must be positive");
    }
    if (!(wavelength > 0.0) || !std::isfinite(wavelength)) {
      throw std::invalid_argument(
          "ArrayGeometry: wavelength must be a positive finite scalar");
    }
  }

  friend bool operator==(const ArrayGeometry& a, const ArrayGeometry& b) {
    return a.element_positions == b.element_positions &&
           a.wavelength == b.wavelength && a.geometry_type == b.geometry_type;
  }
  friend bool operator!=(const ArrayGeometry& a, const ArrayGeometry& b) {
    return !(a == b);
  }
};

}  // namespace holoscan::radar
