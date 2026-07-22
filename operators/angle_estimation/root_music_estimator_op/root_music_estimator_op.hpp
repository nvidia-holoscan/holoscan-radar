// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"

namespace holoscan::radar {

namespace detail {
struct RootMUSICSolverState;
}  // namespace detail

// Gridless Root-MUSIC sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md -> "RootMUSICEstimatorOp".
//
// The covariance, Hermitian eigendecomposition, fixed/EFT source-count
// selection, and noise-subspace projector are the same front-end used by
// MUSICEstimatorOp. For the required uniform linear array, Root-MUSIC sums the
// projector diagonals into its conjugate-reciprocal polynomial, finds roots
// with a companion-matrix eigensolve, and maps the roots inside the unit
// circle that are closest to it to continuous (gridless) DOAs.
//
// In:  snapshot       - complex64/complex128,
//                       (num_items, num_channels, num_snapshots).
// In:  is_valid       - optional bool/uint8,
//                       (num_items, num_snapshots).
// Out: aoa_estimates  - float32/float64, (num_items, 2). The root closest to
//                       the unit circle; equals doa_multi[., 0, .].
// Out: doa_multi      - float32/float64,
//                       (num_items, max_sources, 2), unused rows NaN.
// Out: source_count   - int32, (num_items,), roots resolved.
// Out: eigenvalues    - float32/float64,
//                       (num_items, post-smoothing num_channels), ascending.
template <typename T>
class RootMUSICEstimatorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "RootMUSICEstimatorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RootMUSICEstimatorOp)

  RootMUSICEstimatorOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<ArrayGeometry> array_geometry_;
  Parameter<AngleConvention> angle_convention_;
  Parameter<int64_t> num_sources_;
  Parameter<double> source_count_pfa_;
  Parameter<int64_t> max_sources_;
  Parameter<std::vector<int64_t>> spatial_smoothing_;
  Parameter<bool> forward_backward_;

  ArrayGeometry effective_geometry_{};
  int64_t effective_num_sources_{-1};
  double effective_pfa_{1e-2};
  int64_t effective_max_sources_{4};
  std::vector<int64_t> effective_smoothing_{};
  bool effective_forward_backward_{true};
  bool seeded_{false};
  bool warned_rank_deficient_{false};

  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::unordered_set<std::string> seen_geometry_keys_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};

  std::shared_ptr<detail::RootMUSICSolverState> solver_state_{};
};

}  // namespace holoscan::radar
