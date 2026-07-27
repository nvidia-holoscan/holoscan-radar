// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/dsp/spatial_smoothing.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// Retire MatX eig plan caches while CUDA is still alive. Implemented in the
// .cu (MatX headers stay out of the public .hpp / pybind TU). Prefer
// install_matx_cache_atexit() from Python after importing cupy.
void retire_music_matx_caches();

// Subspace (MUSIC) sparse-path DOA estimator. Contract reference:
// docs/planning/OPERATOR_API.md -> "MUSICEstimatorOp".
//
// For each item, forms the sample covariance R_i = (1/K) sum_k x_k x_k^H over
// its valid snapshots, eigendecomposes it (ascending eigenvalues, columns of
// eigenvectors), resolves a per-item source count d_i (fixed or per-item EFT
// on the eigenvalues), and scans the (az, el) grid for the MUSIC
// pseudospectrum P(theta) = 1 / (a(theta)^H P_n a(theta)), where P_n is the
// projector onto the noise subspace (the num_channels - d_i smallest
// eigenvectors); the argmax is the reported DOA. The steering manifold a(theta)
// is the untapered array response (uniform weighting).
//
// In:  snapshot        - complex64/complex128, (num_items, num_channels,
//                       num_snapshots).
// In:  is_valid        - optional bool/uint8, (num_items, num_snapshots).
//                       When present the covariance sums only over marked
//                       slots (effective K is the per-item valid count);
//                       absent -> all num_snapshots are used.
// Out: aoa_estimates   - float32/float64, (num_items, 2). Dominant (az, el)
//                       (or (u, v) under direction_cosine).
// Out: angle_spectrum  - float32/float64, (num_items, num_az, num_el). MUSIC
//                       pseudo-spectrum. Optional (kNone).
// Out: doa_multi       - float32/float64, (num_items, max_sources, 2). The
//                       top min(num_sources_i, max_sources) pseudospectrum
//                       peaks (strongest first; doa_multi[.,0,.] ==
//                       aoa_estimates), unused rows NaN. Optional (kNone).
// Out: source_count    - int32, (num_items,). Peaks actually resolved
//                       (<= min(num_sources_i, max_sources)). Optional
//                       (kNone).
// Out: eigenvalues     - float32/float64, (num_items, num_channels).
//                       Ascending eigenvalue spectrum of the sample
//                       covariance. Optional (kNone); computed every tick.
//
// `spatial_smoothing` (forward / forward-backward sub-array averaging on a
// filled ULA/URA) is a front-end transform, identical to MVDREstimatorOp's:
// it gathers the snapshot into overlapping sub-array "looks", forms the
// averaged covariance on the reduced sub-array geometry, and the
// covariance/eig/scan path runs unchanged on the L reduced channels.
template <typename T>
class MUSICEstimatorOp : public holoscan::Operator {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
               "MUSICEstimatorOp supports float and double precision");

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MUSICEstimatorOp)

  MUSICEstimatorOp() = default;

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
  Parameter<std::vector<double>> az_angles_;
  Parameter<std::vector<double>> el_angles_;
  // Signal-subspace dimension (model order). Encoded as an int64_t: a
  // negative value selects "auto" (the contract's int-or-auto), where the
  // per-item source count is resolved by EFT on the covariance eigenvalues;
  // a non-negative value fixes the order uniformly. Default -1 (auto).
  Parameter<int64_t> num_sources_;
  // EFT over-estimation (false-alarm) probability for num_sources: auto.
  Parameter<double> source_count_pfa_;
  // Capacity of the doa_multi / source_count outputs, and the K_max cap on
  // auto's EFT search. [live].
  Parameter<int64_t> max_sources_;
  // Per-aperture-axis sub-array COUNTS for forward spatial smoothing on a
  // filled ULA/URA: empty -> off; [P] on a ULA; [P_az, P_el] on a URA. [live].
  Parameter<std::vector<int64_t>> spatial_smoothing_;
  // Include the conjugate-reversed (backward) looks - forward-backward
  // smoothing (Pillai/Kwon). Needs a centro-symmetric ULA/URA. Default true.
  Parameter<bool> forward_backward_;

  ArrayGeometry effective_geometry_{};
  std::vector<double> effective_az_{};
  std::vector<double> effective_el_{};
  int64_t effective_num_sources_{-1};
  double effective_pfa_{1e-2};
  int64_t effective_max_sources_{4};
  std::vector<int64_t> effective_smoothing_{};
  bool effective_forward_backward_{true};
  bool seeded_{false};
  bool warned_rank_deficient_{false};

  // Untapered steering manifold a(theta) (num_beams, num_channels), beam
  // el-major (beam = ie * num_az + ia). Built against the reduced sub-array
  // geometry when smoothing is on. Cached, warn-once on rebuild.
  ConfigCache<std::shared_ptr<holoscan::Tensor>> steering_cache_;

  // Spatial-smoothing plan + uploaded index tables, rebuilt only when the
  // geometry / smoothing config changes (keyed by smoothing_key_).
  SpatialSmoothingPlan smoothing_plan_{};
  std::string smoothing_key_{};
  std::shared_ptr<holoscan::Tensor> d_subarray_index_{};
  std::shared_ptr<holoscan::Tensor> d_fb_flip_{};
};

}  // namespace holoscan::radar
