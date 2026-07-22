// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

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

}  // namespace holoscan::radar
