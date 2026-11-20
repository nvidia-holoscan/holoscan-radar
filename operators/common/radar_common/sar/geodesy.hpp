// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>

// Host-only geodetic conversions on the WGS 84 ellipsoid. No Holoscan, no
// CUDA: these are model helpers, used both at configuration time on scalar
// geometry and inside an operator to build its rotation once.
namespace holoscan::radar::geodesy {

inline constexpr double kWgs84SemiMajor = 6378137.0;
inline constexpr double kWgs84Flattening = 1.0 / 298.257223563;

// Row-major ENU-from-ECEF rotation about `origin`, rows east / north / up.
//
// The latitude comes from Bowring's closed form rather than an iteration. Its
// two correction terms are each written more than one way in the literature --
// `e'^2 * b` and `e^2 * a^2 / b` are the same quantity -- which is how this
// file came to exist: the app and the operator carried the two spellings
// separately, and reconciling them by inspection is harder than it looks.
inline std::array<double, 9> ecef_to_enu_rotation(const std::array<double, 3>& origin) {
  constexpr double a = kWgs84SemiMajor;
  const double b = a * (1.0 - kWgs84Flattening);
  const double e2 = 1.0 - (b * b) / (a * a);
  const double ep2 = (a * a - b * b) / (b * b);
  const double lon = std::atan2(origin[1], origin[0]);
  const double p = std::hypot(origin[0], origin[1]);
  const double theta = std::atan2(origin[2] * a, p * b);
  const double slt = std::sin(theta);
  const double clt = std::cos(theta);
  const double lat =
      std::atan2(origin[2] + ep2 * b * slt * slt * slt, p - e2 * a * clt * clt * clt);
  const double sl = std::sin(lat);
  const double cl = std::cos(lat);
  const double so = std::sin(lon);
  const double co = std::cos(lon);
  return {-so, co, 0.0, -sl * co, -sl * so, cl, cl * co, cl * so, sl};
}

// Apply a rotation from `ecef_to_enu_rotation` to one point's offset from the
// origin. Split from the rotation so a per-pulse loop builds the basis once.
inline std::array<double, 3> apply_enu_rotation(const std::array<double, 9>& rotation,
                                                const std::array<double, 3>& offset) {
  std::array<double, 3> out{};
  for (int j = 0; j < 3; ++j) {
    out[static_cast<std::size_t>(j)] = rotation[static_cast<std::size_t>(3 * j)] * offset[0] +
                                       rotation[static_cast<std::size_t>(3 * j + 1)] * offset[1] +
                                       rotation[static_cast<std::size_t>(3 * j + 2)] * offset[2];
  }
  return out;
}

inline std::array<double, 3> ecef_to_scene_enu(const std::array<double, 3>& ecef,
                                               const std::array<double, 3>& origin) {
  return apply_enu_rotation(ecef_to_enu_rotation(origin),
                            {ecef[0] - origin[0], ecef[1] - origin[1], ecef[2] - origin[2]});
}

}  // namespace holoscan::radar::geodesy
