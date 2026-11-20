// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

// Boundary AER parameterization for angle operators. Kernels run in the fixed
// internal canonical convention; this descriptor relabels scan grids and DOA
// outputs only. Default construction selects the canonical radar_boresight
// preset; callers choose another boundary convention explicitly.
struct AngleConvention {
  AngleConventionPreset preset{AngleConventionPreset::kRadarBoresight};
  SignedAxis boresight_axis{SignedAxis::kPlusX};
  SignedAxis azimuth_positive_axis{SignedAxis::kPlusY};
  SignedAxis elevation_positive_axis{SignedAxis::kPlusZ};
  ElevationReference elevation_reference{ElevationReference::kHorizon};

  bool uses_direction_cosine() const { return preset == AngleConventionPreset::kDirectionCosine; }

  friend bool operator==(const AngleConvention& a, const AngleConvention& b) {
    return a.preset == b.preset && a.boresight_axis == b.boresight_axis &&
           a.azimuth_positive_axis == b.azimuth_positive_axis &&
           a.elevation_positive_axis == b.elevation_positive_axis &&
           a.elevation_reference == b.elevation_reference;
  }
  friend bool operator!=(const AngleConvention& a, const AngleConvention& b) { return !(a == b); }
};

}  // namespace holoscan::radar
