// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"

namespace holoscan::radar {

struct ShiftInvariantSubarrayPlan {
  ShiftInvariantSubarrayPair pair{};
  std::array<double, 3> displacement{};
  std::array<double, 3> axis{};
  std::array<double, 3> broadside{};
  double spacing{0.0};
};

// Resolve auto or validate an explicit corresponding subarray pair. Every
// corresponding position must differ by one common nonzero displacement.
ShiftInvariantSubarrayPlan build_shift_invariant_subarray_plan(
    const ArrayGeometry& geometry,
    const ShiftInvariantSubarrayPair& requested);

// The two orthogonal maximum-overlap shift invariances of a filled URA laid
// out in `(iz` outer, `iy` inner) channel order: the y-shift (azimuth) pairs
// each element with its `+y` neighbor (`nz * (ny - 1)` pairs) and the z-shift
// (elevation) pairs each element with its `+z` neighbor (`(nz - 1) * ny`
// pairs). 2-D ESPRIT solves one rotation operator per invariance and pairs
// their eigenvalues by joint diagonalization.
struct UraInvariancePairs {
  std::vector<int64_t> y_reference;
  std::vector<int64_t> y_shifted;
  std::vector<int64_t> z_reference;
  std::vector<int64_t> z_shifted;
};

UraInvariancePairs build_ura_dual_invariance(int64_t ny, int64_t nz);

}  // namespace holoscan::radar
