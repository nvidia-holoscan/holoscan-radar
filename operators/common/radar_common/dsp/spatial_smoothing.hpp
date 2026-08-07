// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Spatial-smoothing front-end for the covariance (sub-array-averaging) angle
// estimators — MVDR today, MUSIC / Root-MUSIC / ESPRIT when they land.
//
// Spatial smoothing decorrelates coherent multipath (and raises covariance
// rank) by averaging the covariance over maximally-overlapping uniform
// sub-arrays. This header owns the estimator-agnostic HOST bookkeeping:
//
//   1. Lattice detection — turn a flat ArrayGeometry (element_positions) into a
//      filled 1-D (ULA) or 2-D (URA) integer lattice, or reject a non-uniform /
//      holey geometry (which must be reduced to a uniform sub-array upstream —
//      reduced to a uniform sub-array upstream — see S3).
//   2. Sub-array enumeration — from per-aperture-axis sub-array COUNTS, build
//      the (num_subarrays, subarray_size) table of flat channel indices.
//   3. Reduced geometry — the reference sub-array's element positions, used to
//      build the steering manifold the smoothed covariance corresponds to.
//   4. Forward-backward flip — the index permutation J (reverse each lattice
//      axis) that turns a sub-array look into its conjugate-reversed partner.
//
// The device gather that materializes the sub-array "looks" from these tables
// lives in spatial_smoothing_device.{hpp,cu} (radar_common stays CUDA-free).
//
// Design: the whole thing is a front-end transform
//   (snapshot C×K, geometry over C)  ->  (ss_snapshot L×M, reduced geometry L)
// after which the estimator's existing covariance / eig / inverse / scan path
// runs UNCHANGED on L channels. 1-D vs 2-D is only how sub-arrays are
// enumerated; the reduction and estimation are identical.

#pragma once

#include <cstdint>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"

namespace holoscan::radar {

// Plan for forward(/backward) spatial smoothing of one covariance estimator's
// snapshots. Built once per (geometry, counts, forward_backward) configuration
// and cached alongside the steering manifold.
struct SpatialSmoothingPlan {
  // false → pass-through: no sub-array reduction and no forward-backward
  // averaging requested (counts all-ones and !forward_backward). The estimator
  // runs on the full array exactly as if smoothing were absent.
  bool enabled{false};
  // Include the conjugate-reversed (backward) looks — valid on the
  // centro-symmetric filled ULA/URA this validates.
  bool forward_backward{false};

  int64_t num_channels{0};   // C — full array element count
  int64_t subarray_size{0};  // L — sub-array element count (prod of axis lens)
  int64_t num_subarrays{0};  // P — sub-array count (prod of per-axis counts)

  // (num_subarrays, subarray_size) row-major table of flat channel indices:
  // subarray_index[p * subarray_size + l] is the full-array channel feeding
  // local element l of sub-array p. Local element order matches
  // reduced_geometry's element order.
  std::vector<int32_t> subarray_index;

  // Forward-backward flip permutation over the L local indices (reverse each
  // lattice axis). fb_flip[l] is the local index element l maps to under J.
  // Empty when !forward_backward.
  std::vector<int32_t> fb_flip;

  // Reference sub-array (shift 0) element positions + wavelength — the geometry
  // the smoothed covariance corresponds to; feed it to build_steering_table.
  ArrayGeometry reduced_geometry;

  // Columns contributed to the covariance per input snapshot: P (forward) or
  // 2P (forward-backward). Effective snapshot count is looks_per_snapshot * K.
  int64_t looks_per_snapshot() const {
    return forward_backward ? 2 * num_subarrays : num_subarrays;
  }
};

// Build a smoothing plan for `geometry` given per-aperture-axis sub-array
// `counts` (counts[d] = number of maximally-overlapping sub-arrays along
// aperture axis d; sub-array length is derived, L_d = M_d - counts[d] + 1).
//
// `counts` must be non-empty and its length must equal the aperture
// dimensionality (1 for a ULA, 2 for a URA). All-ones counts with
// !forward_backward yield a disabled (pass-through) plan.
//
// Throws std::invalid_argument if the geometry is not a filled, uniformly
// spaced 1-D/2-D lattice (holey / non-uniform / >2-D aperture), if counts has
// the wrong length, or if any count is < 1 or > its axis extent.
SpatialSmoothingPlan build_spatial_smoothing_plan(
    const ArrayGeometry& geometry, const std::vector<int64_t>& counts,
    bool forward_backward);

}  // namespace holoscan::radar
