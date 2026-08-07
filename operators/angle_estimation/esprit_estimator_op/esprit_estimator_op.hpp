// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "angle_estimation/angle_boundary_device.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/shift_invariant_subarray_pair.hpp"
#include "radar_common/dsp/shift_invariant_subarray.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"

namespace holoscan::radar {

// Gridless least-squares ESPRIT sparse-path DOA estimator. The covariance,
// Hermitian eigendecomposition, and fixed/EFT source-count front-end match
// MUSICEstimatorOp. The signal subspace is restricted to two corresponding
// displaced subarrays; eigenvalue rotation gives continuous 1-D DOAs.
template <typename T>
class ESPRITEstimatorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ESPRITEstimatorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ESPRITEstimatorOp)

  ESPRITEstimatorOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  // Drain the MatX eig cache + global MemTracker once at graph stop, while CUDA
  // is still valid (per-tick only clears the plan cache; see music_stage.hpp).
  void stop() override;

 private:
  Parameter<ArrayGeometry> array_geometry_;
  Parameter<AngleConvention> angle_convention_;
  Parameter<int64_t> num_sources_;
  Parameter<double> source_count_pfa_;
  Parameter<int64_t> max_sources_;
  Parameter<std::vector<int64_t>> spatial_smoothing_;
  Parameter<bool> forward_backward_;
  Parameter<ShiftInvariantSubarrayPair> subarray_spec_;

  ArrayGeometry effective_geometry_{};
  int64_t effective_num_sources_{-1};
  double effective_pfa_{1e-2};
  int64_t effective_max_sources_{4};
  std::vector<int64_t> effective_smoothing_{};
  bool effective_forward_backward_{true};
  bool seeded_{false};
  bool warned_rank_deficient_{false};
  detail::DeviceAngleConvention<T> device_angle_convention_{};

  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::string subarray_key_{};
  ShiftInvariantSubarrayPlan subarray_plan_{};
  std::unordered_set<std::string> seen_geometry_keys_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};
  std::shared_ptr<holoscan::Tensor> d_reference_indices_{};
  std::shared_ptr<holoscan::Tensor> d_shifted_indices_{};

  // URA (2-D) dual-invariance path: the two orthogonal shift pairings uploaded
  // to the device, keyed by geometry so they rebuild only when it changes.
  std::string ura_key_{};
  int64_t ura_ny_{0};
  int64_t ura_nz_{0};
  int64_t ura_y_pairs_{0};
  int64_t ura_z_pairs_{0};
  std::shared_ptr<holoscan::Tensor> d_y_reference_{};
  std::shared_ptr<holoscan::Tensor> d_y_shifted_{};
  std::shared_ptr<holoscan::Tensor> d_z_reference_{};
  std::shared_ptr<holoscan::Tensor> d_z_shifted_{};
};

}  // namespace holoscan::radar
