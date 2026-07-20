// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// Retire MatX eig/inv plan caches while CUDA is still alive. Implemented
// in the .cu (MatX headers stay out of the public .hpp / pybind TU).
// Prefer install_matx_cache_atexit() from Python after importing cupy.
void retire_matx_solver_caches();

// Adaptive (MVDR / Capon) sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md → "MVDREstimatorOp".
//
// For each item, forms the sample covariance R̂ = (1/K) Σ_k x_k·x_kᴴ over its
// valid snapshots, applies diagonal loading (R + δI), inverts it, and scans the
// (az, el) grid for the Capon power P(θ) = 1 / (a(θ)ᴴ R̂⁻¹ a(θ)); the argmax is
// the reported DOA. The steering manifold a(θ) is the untapered array response
// (uniform weighting) — MVDR shapes the beam adaptively from the data, so no
// array taper parameter.
//
// In:  snapshot        — complex64/complex128, (num_items, num_channels) or
//                       (num_items, num_channels, num_snapshots). The 2-D form
//                       is treated as num_snapshots == 1 (rank-1 covariance —
//                       relies on diagonal loading for invertibility).
// In:  is_valid        — optional bool/uint8, (num_items, num_snapshots). When
//                       present the covariance sums only over marked slots (so
//                       the effective K is the per-item valid count); absent →
//                       all num_snapshots are used.
// Out: aoa_estimates   — float32/float64, (num_items, 2). Dominant (az, el)
//                       (or (u, v) under direction_cosine), row-aligned with
//                       the input items.
// Out: angle_spectrum  — float32/float64, (num_items, num_az, num_el). Capon
//                       spectrum. Invalid (direction-cosine invisible) beams
//                       are zero. Optional (kNone).
// Out: doa_multi       — float32/float64, (num_items, max_sources, 2). The
//                       max_sources strongest Capon peaks (strongest first;
//                       doa_multi[.,0,.] == aoa_estimates — pass 0 is the
//                       global argmax), unused rows NaN. Optional (kNone).
// Out: source_count    — int32, (num_items,). Peaks found (≤ max_sources).
//                       Optional (kNone).
// Out: eigenvalues     — float32/float64, (num_items, num_channels). Ascending
//                       eigenvalue spectrum of the RAW (pre-loading) sample
//                       covariance, for subspace / source-count diagnostics.
//                       Optional (kNone); computed every tick.
//
// Scope: the MVDR core — covariance + is_valid masking, diagonal loading,
// batched inverse, Capon spectrum, dominant DOA, the multi-source outputs
// (max_sources / doa_multi / source_count), and the covariance eigenvalues.
// The one remaining contract parameter, `spatial_smoothing`, is BLOCKED on a
// covariance-estimator team decision (see OPEN_ITEMS.md) and is not declared;
// until it lands this operator is NOT checklist-complete.
template <typename T>
class MVDREstimatorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MVDREstimatorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MVDREstimatorOp)

  MVDREstimatorOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<ArrayGeometry> array_geometry_;
  Parameter<AngleConvention> angle_convention_;
  Parameter<std::vector<double>> az_angles_;
  Parameter<std::vector<double>> el_angles_;
  // Diagonal loading δ for the R + δI regularization. Encoded as a double:
  // a negative value selects "auto" (the contract's float-or-auto), where δ is
  // scaled to a small multiple of the per-item noise-floor proxy trace(R̂)/C;
  // a non-negative value is used as the absolute δ. Default -1 (auto).
  Parameter<double> diagonal_loading_;
  // Capacity of the doa_multi / source_count outputs — the number of Capon
  // peaks to report (aoa_estimates is always the strongest). [live].
  Parameter<int64_t> max_sources_;

  ArrayGeometry effective_geometry_{};
  std::vector<double> effective_az_{};
  std::vector<double> effective_el_{};
  double effective_loading_{-1.0};
  int64_t effective_max_sources_{1};
  bool seeded_{false};

  // Untapered steering manifold a(θ) (num_beams, num_channels), beam el-major
  // (beam = ie * num_az + ia) to match the Capon reduce kernel's decode.
  // Cached, warn-once on rebuild (geometry / scan-grid change).
  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;
};

}  // namespace holoscan::radar
