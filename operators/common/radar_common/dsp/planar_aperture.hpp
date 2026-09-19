// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "radar_common/descriptors/array_geometry.hpp"

namespace holoscan::radar {

// A filled uniform rectangular array (URA) resolved from an ArrayGeometry, in
// the make_ura_geometry channel order `(iz` outer, `iy` inner): channel
// `c = iz * ny + iy` sits at `origin + iy * dy * y_axis + iz * dz * z_axis`.
// The y aperture axis resolves the first (azimuth) direction cosine `u` and
// the z aperture axis the second (elevation) cosine `v`. The cross-product
// normal is sign-oriented toward canonical radar boresight (+x), independent
// of which aperture axis happens to be contiguous in channel order, so a
// direction is reconstructed as
// `u * y_axis + v * z_axis + sqrt(1 - u^2 - v^2) * normal`.
struct PlanarAperturePlan {
  bool is_ura{false};
  int64_t ny{0};              // elements along the y (azimuth) aperture axis
  int64_t nz{0};              // elements along the z (elevation) aperture axis
  double dy{0.0};             // y element pitch (meters)
  double dz{0.0};             // z element pitch (meters)
  std::array<double, 3> y_axis{};
  std::array<double, 3> z_axis{};
  std::array<double, 3> normal{};  // forward-hemisphere planar normal
};

// Detect whether `geometry` is a filled URA (ny > 1 && nz > 1) laid out in the
// `(iz` outer, `iy` inner) channel order used by make_ura_geometry, recovering
// ny/nz, the pitches, and the two orthonormal aperture-axis unit vectors.
//
// Returns `is_ura = false` for a line array (a ULA or a single row/column),
// leaving the caller on its existing 1-D path. Throws std::invalid_argument
// when the geometry is tagged kUra but is not a consistent filled rectangular
// lattice, when the recovered lattice does not tile the channel count, or when
// the two aperture axes are not orthogonal (the separable per-axis (u, v)
// model requires orthogonal apertures).
PlanarAperturePlan analyze_planar_aperture(const ArrayGeometry& geometry);

}  // namespace holoscan::radar
