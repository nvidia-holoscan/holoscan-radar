// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "radar_common/descriptors/angle_convention.hpp"

namespace holoscan::radar {

// Unit direction in the internal canonical array-local frame (+x boresight,
// +y left, +z up; az CCW from +x about +z; elevation up from the x-y plane).
struct CanonicalDirection {
  double x{1.0};
  double y{0.0};
  double z{0.0};
};

// True when (u, v) lie in the forward hemisphere of the direction-cosine grid.
inline bool is_physical_direction_cosine(double u, double v) {
  return (u * u + v * v) <= 1.0 + 1e-12;
}

// Map boundary scan coordinates to a canonical unit vector. For
// direction_cosine, `az` carries u and `el` carries v.
CanonicalDirection boundary_to_canonical(const AngleConvention& conv, double az,
                                         double el);

// Map a canonical unit vector to boundary (az, el) or (u, v) labels.
void canonical_to_boundary(const AngleConvention& conv,
                           const CanonicalDirection& dir, double* out0,
                           double* out1);

}  // namespace holoscan::radar
