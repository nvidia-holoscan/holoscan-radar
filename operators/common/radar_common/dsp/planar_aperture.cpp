// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/planar_aperture.hpp"

#include <algorithm>
#include <array>
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

std::array<double, 3> subtract(const std::array<double, 3>& a,
                               const std::array<double, 3>& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double norm(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] +
                   value[2] * value[2]);
}

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> cross(const std::array<double, 3>& a,
                            const std::array<double, 3>& b) {
  return {
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

double geometry_tolerance(const ArrayGeometry& geometry) {
  double scale = geometry.wavelength;
  for (double coordinate : geometry.element_positions) {
    scale = std::max(scale, std::abs(coordinate));
  }
  return 1e-6 * scale;
}

}  // namespace

PlanarAperturePlan analyze_planar_aperture(const ArrayGeometry& geometry) {
  geometry.validate();
  PlanarAperturePlan plan;

  const bool tagged_ura = geometry.geometry_type &&
                          *geometry.geometry_type == ArrayGeometryType::kUra;
  const bool tagged_line = geometry.geometry_type &&
                           *geometry.geometry_type == ArrayGeometryType::kUla;
  if (tagged_line) {
    return plan;  // is_ura = false; caller keeps its 1-D path.
  }

  const int64_t channels = geometry.num_elements();
  if (channels < 4) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: array_geometry tagged URA has too few "
          "elements for a 2-D aperture");
    }
    return plan;
  }

  const double tolerance = geometry_tolerance(geometry);
  const auto origin = position(geometry, 0);
  const auto delta_y = subtract(position(geometry, 1), origin);
  const double dy = norm(delta_y);
  if (!(dy > 0.0) || !std::isfinite(dy)) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: URA inner-axis pitch must be positive");
    }
    return plan;
  }

  // The inner (y) run ends at the first channel that leaves the line through
  // the origin along delta_y (make_ura_geometry resets iy to 0 and steps iz).
  int64_t ny = channels;
  for (int64_t m = 1; m < channels; ++m) {
    const std::array<double, 3> expected{
        origin[0] + static_cast<double>(m) * delta_y[0],
        origin[1] + static_cast<double>(m) * delta_y[1],
        origin[2] + static_cast<double>(m) * delta_y[2],
    };
    if (norm(subtract(position(geometry, m), expected)) > tolerance) {
      ny = m;
      break;
    }
  }
  if (ny >= channels || ny < 2) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: array_geometry tagged URA is a single "
          "line (no second aperture axis)");
    }
    return plan;  // Pure line array: leave the caller's 1-D path.
  }
  if (channels % ny != 0) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: URA channel count is not ny * nz "
          "(not a filled rectangular lattice)");
    }
    return plan;
  }
  const int64_t nz = channels / ny;
  if (nz < 2) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: array_geometry tagged URA has a single "
          "row");
    }
    return plan;
  }

  const auto delta_z = subtract(position(geometry, ny), origin);
  const double dz = norm(delta_z);
  if (!(dz > 0.0) || !std::isfinite(dz)) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: URA outer-axis pitch must be positive");
    }
    return plan;
  }

  // Validate the full filled lattice in (iz outer, iy inner) order.
  for (int64_t iz = 0; iz < nz; ++iz) {
    for (int64_t iy = 0; iy < ny; ++iy) {
      const std::array<double, 3> expected{
          origin[0] + static_cast<double>(iy) * delta_y[0] +
              static_cast<double>(iz) * delta_z[0],
          origin[1] + static_cast<double>(iy) * delta_y[1] +
              static_cast<double>(iz) * delta_z[1],
          origin[2] + static_cast<double>(iy) * delta_y[2] +
              static_cast<double>(iz) * delta_z[2],
      };
      const auto actual = position(geometry, iz * ny + iy);
      if (norm(subtract(actual, expected)) > tolerance) {
        if (tagged_ura) {
          throw std::invalid_argument(
              "analyze_planar_aperture: array_geometry tagged URA is not a "
              "consistent filled rectangular lattice in (iz, iy) channel "
              "order");
        }
        return plan;
      }
    }
  }

  std::array<double, 3> y_axis{delta_y[0] / dy, delta_y[1] / dy,
                               delta_y[2] / dy};
  std::array<double, 3> z_axis{delta_z[0] / dz, delta_z[1] / dz,
                               delta_z[2] / dz};
  if (std::abs(dot(y_axis, z_axis)) > 1e-6) {
    if (tagged_ura) {
      throw std::invalid_argument(
          "analyze_planar_aperture: the two aperture axes must be orthogonal "
          "for the separable per-axis (u, v) angle model");
    }
    // An untagged colinear/non-orthogonal lattice (e.g. a non-uniform line)
    // is not a separable URA; leave the caller on its 1-D path.
    return plan;
  }

  plan.is_ura = true;
  plan.ny = ny;
  plan.nz = nz;
  plan.dy = dy;
  plan.dz = dz;
  plan.y_axis = y_axis;
  plan.z_axis = z_axis;
  plan.normal = cross(y_axis, z_axis);
  return plan;
}

}  // namespace holoscan::radar
