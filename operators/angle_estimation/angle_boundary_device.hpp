// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <stdexcept>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/math/constants.hpp"

namespace holoscan::radar::detail {

template <typename T>
struct DeviceAngleConvention {
  AngleConventionPreset preset{AngleConventionPreset::kRadarBoresight};
  T boresight_x{1};
  T boresight_y{0};
  T boresight_z{0};
  T azimuth_x{0};
  T azimuth_y{1};
  T azimuth_z{0};
  T elevation_x{0};
  T elevation_y{0};
  T elevation_z{1};
  ElevationReference elevation_reference{ElevationReference::kHorizon};
};

inline void signed_axis_vector(SignedAxis axis, double* x, double* y,
                               double* z) {
  *x = 0.0;
  *y = 0.0;
  *z = 0.0;
  switch (axis) {
    case SignedAxis::kPlusX:
      *x = 1.0;
      return;
    case SignedAxis::kMinusX:
      *x = -1.0;
      return;
    case SignedAxis::kPlusY:
      *y = 1.0;
      return;
    case SignedAxis::kMinusY:
      *y = -1.0;
      return;
    case SignedAxis::kPlusZ:
      *z = 1.0;
      return;
    case SignedAxis::kMinusZ:
      *z = -1.0;
      return;
  }
  throw std::invalid_argument("AngleConvention: unknown signed axis");
}

template <typename T>
DeviceAngleConvention<T> make_device_angle_convention(
    const AngleConvention& convention) {
  DeviceAngleConvention<T> result;
  result.preset = convention.preset;
  result.elevation_reference = convention.elevation_reference;
  if (convention.preset != AngleConventionPreset::kCustom) {
    return result;
  }

  double bx = 0.0;
  double by = 0.0;
  double bz = 0.0;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  double ex = 0.0;
  double ey = 0.0;
  double ez = 0.0;
  signed_axis_vector(convention.boresight_axis, &bx, &by, &bz);
  signed_axis_vector(convention.azimuth_positive_axis, &ax, &ay, &az);
  signed_axis_vector(convention.elevation_positive_axis, &ex, &ey, &ez);
  const double ba = bx * ax + by * ay + bz * az;
  const double be = bx * ex + by * ey + bz * ez;
  const double ae = ax * ex + ay * ey + az * ez;
  if (std::abs(ba) > 1e-9 || std::abs(be) > 1e-9 ||
      std::abs(ae) > 1e-9) {
    throw std::invalid_argument(
        "AngleConvention: preset=custom requires boresight, azimuth, and "
        "elevation on three distinct principal axes");
  }
  result.boresight_x = static_cast<T>(bx);
  result.boresight_y = static_cast<T>(by);
  result.boresight_z = static_cast<T>(bz);
  result.azimuth_x = static_cast<T>(ax);
  result.azimuth_y = static_cast<T>(ay);
  result.azimuth_z = static_cast<T>(az);
  result.elevation_x = static_cast<T>(ex);
  result.elevation_y = static_cast<T>(ey);
  result.elevation_z = static_cast<T>(ez);
  return result;
}

#ifdef __CUDACC__
template <typename T>
__device__ inline void angle_canonical_to_boundary(
    const DeviceAngleConvention<T>& convention, T x, T y, T z_coord, T* out0,
    T* out1) {
  if (convention.preset == AngleConventionPreset::kDirectionCosine) {
    *out0 = y;
    *out1 = z_coord;
    return;
  }

  T az = atan2(y, x);
  T el = atan2(z_coord, hypot(x, y));
  if (convention.preset == AngleConventionPreset::kAerospaceNed ||
      convention.preset == AngleConventionPreset::kGeodeticNed) {
    az = -az;
    el = -el;
  } else if (convention.preset == AngleConventionPreset::kGeodeticEnu) {
    az = atan2(x, -y);
  } else if (convention.preset ==
             AngleConventionPreset::kPhysicsSpherical) {
    el = static_cast<T>(kHalfPi) - el;
  } else if (convention.preset == AngleConventionPreset::kCustom) {
    const T boresight = x * convention.boresight_x +
                        y * convention.boresight_y +
                        z_coord * convention.boresight_z;
    const T azimuth = x * convention.azimuth_x + y * convention.azimuth_y +
                      z_coord * convention.azimuth_z;
    T elevation = x * convention.elevation_x +
                  y * convention.elevation_y +
                  z_coord * convention.elevation_z;
    elevation = elevation < T{-1} ? T{-1} : elevation;
    elevation = elevation > T{1} ? T{1} : elevation;
    az = atan2(azimuth, boresight);
    el = asin(elevation);
    if (convention.elevation_reference == ElevationReference::kZenith) {
      el = static_cast<T>(kHalfPi) - el;
    }
  }
  *out0 = az;
  *out1 = el;
}
#endif  // __CUDACC__

}  // namespace holoscan::radar::detail
