// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/sample_geometry.hpp"

// Expansion of a uniform SampleGeometry lattice into the explicit
// (num_v, num_u, 3) point list the formation kernels consume.
//
// This is what makes BackprojectionOp's two geometry modes reach one kernel
// path: with `sample_points` connected the lattice *is* that tensor;
// otherwise it is expanded from the `sample_geometry` descriptor here. Both
// end up with the same 2D shape, so the kernel never learns which mode it
// is running in.
//
// Host-side and pure: no Holoscan, no CUDA. Callers upload the result.

namespace holoscan::radar {

// Number of elements in the expanded lattice, num_v * num_u * 3. Throws
// std::invalid_argument when that product overflows int64_t, which a
// caller-supplied count pair can do long before the allocation would fail.
inline int64_t sample_lattice_size(const SampleGeometry& geometry) {
  if (!geometry.is_valid()) {
    throw std::invalid_argument(
        "sample_lattice: geometry is default-constructed (is_valid() is "
        "false); a lattice requires origin, both axes, spacing and counts");
  }
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  const int64_t num_v = geometry.num_v();
  const int64_t num_u = geometry.num_u();
  if (num_u > kMax / num_v || num_v * num_u > kMax / 3) {
    throw std::invalid_argument(
        "sample_lattice: num_v * num_u * 3 overflows int64_t");
  }
  return num_v * num_u * 3;
}

// Write the lattice to `out` as row-major (num_v, num_u, 3) in the caller's
// precision. `out` must have room for sample_lattice_size(geometry)
// elements.
//
// Positions are accumulated in double and cast per element rather than
// through a double buffer, following the module convention of computing in
// double while letting the caller hold the result at its instantiation
// precision. Here that also matters for memory: a DEM-scale lattice is
// hundreds of megabytes, and a double intermediate would double it.
template <typename T>
void expand_sample_lattice(const SampleGeometry& geometry, T* out) {
  const int64_t size = sample_lattice_size(geometry);
  if (out == nullptr && size > 0) {
    throw std::invalid_argument("sample_lattice: out must not be null");
  }
  const int64_t num_v = geometry.num_v();
  const int64_t num_u = geometry.num_u();
  int64_t index = 0;
  for (int64_t iv = 0; iv < num_v; ++iv) {
    for (int64_t iu = 0; iu < num_u; ++iu) {
      const auto position = geometry.sample_position(iv, iu);
      out[index++] = static_cast<T>(position[0]);
      out[index++] = static_cast<T>(position[1]);
      out[index++] = static_cast<T>(position[2]);
    }
  }
}

// Convenience overload allocating the point list, for tests and small
// lattices. Hot paths should expand straight into a pinned or device
// staging buffer with the pointer overload above.
inline std::vector<double> expand_sample_lattice(
    const SampleGeometry& geometry) {
  std::vector<double> points(
      static_cast<size_t>(sample_lattice_size(geometry)));
  expand_sample_lattice(geometry, points.data());
  return points;
}

}  // namespace holoscan::radar
