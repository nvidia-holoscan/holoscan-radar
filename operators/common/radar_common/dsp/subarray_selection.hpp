// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Uniform sub-array selection from a (possibly holey) MIMO virtual array — the
// host bookkeeping for team-blocked item S3.
//
// A MIMO virtual array (p_virtual = p_tx + p_rx, after overlap dedup) is
// generally non-uniform / holey. The gridless estimators (Root-MUSIC, ESPRIT)
// need a ULA / shift-invariant geometry, and spatial smoothing (S1) needs a
// filled uniform sub-array. This helper selects one:
//
//   * auto (default): the largest filled ULA (1-D) / URA (2-D) sub-block of the
//     underlying lattice the holey array samples;
//   * explicit: a caller-supplied set of channel indices, validated to form a
//     filled uniform ULA/URA.
//
// It returns the channel indices to keep plus the reduced geometry; the actual
// snapshot channel gather reuses the S1 gather (run_spatial_smoothing_gather
// with a single sub-array, no forward-backward). CUDA-free (lives in
// radar_common); see docs planning S3.

#pragma once

#include <cstdint>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"

namespace holoscan::radar {

// Result of selecting a uniform sub-array from a geometry.
struct SubarraySelection {
  // false → the whole array is already a filled ULA/URA and no channel is
  // dropped (channel_index is the identity, reduced_geometry == input).
  bool reduced{false};
  int64_t num_channels{0};   // input channel count (N_virtual)
  int64_t subarray_size{0};  // L — selected channel count

  // (subarray_size,) input channel indices to keep, in reduced-geometry order
  // (row-major over the aperture axes, matching S1's sub-array element order).
  std::vector<int32_t> channel_index;

  // The selected filled ULA/URA (element positions in input order above +
  // wavelength); geometry_type set to kUla / kUra.
  ArrayGeometry reduced_geometry;
};

// Select a uniform sub-array from `geometry`.
//
// `explicit_indices` empty → auto: pick the largest filled ULA/URA sub-block
// (objective: maximum element count, tie-broken by larger azimuth aperture).
// Non-empty → use exactly those input channel indices, validated to form a
// filled uniform ULA/URA.
//
// Throws std::invalid_argument if: the aperture is >2-D; the positions are not
// on a consistent lattice; auto-selection finds no filled sub-block of at least
// `min_subarray_size`; or explicit indices are out of range, duplicated, or do
// not form a filled uniform ULA/URA.
SubarraySelection build_subarray_selection(
    const ArrayGeometry& geometry,
    const std::vector<int64_t>& explicit_indices, int64_t min_subarray_size);

}  // namespace holoscan::radar
