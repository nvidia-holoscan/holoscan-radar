// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace holoscan::radar {

struct CFARParams;

inline constexpr std::uint64_t kCfarCalibrationSeed = 0x434641524d433031ULL;
inline constexpr std::uint32_t kCfarCalibrationVersion = 1;

class CfarCalibrationCancelled : public std::runtime_error {
 public:
  CfarCalibrationCancelled() : std::runtime_error("CFAR calibration cancelled") {}
};

struct CfarCalibrationBudget {
  std::uint64_t trials{0};
  std::uint64_t tail_rank{0};
  double coverage{0.0};  // Only meaningful when trials is nonzero.
};

struct CfarCalibrationGeometry {
  std::int64_t lut_index{0};
  int doppler{0};
  int range{0};
};

struct CfarCalibrationPlan {
  std::shared_ptr<const CFARParams> params;
  std::vector<std::int64_t> input_shape;
  std::int64_t effective_looks{1};
  std::uint64_t seed{kCfarCalibrationSeed};
  std::uint32_t algorithm_version{kCfarCalibrationVersion};
  CfarCalibrationBudget budget;
  std::vector<CfarCalibrationGeometry> geometries;
  std::uint64_t total_trials{0};
  std::size_t lut_size{0};
  double estimated_seconds{0.0};
  std::string method;
};

struct CfarCalibrationProgress {
  std::uint64_t completed_trials{0};
  std::uint64_t total_trials{0};
  std::size_t completed_geometries{0};
  std::size_t total_geometries{0};
  double elapsed_seconds{0.0};
  double remaining_seconds{0.0};
  std::string started_at;
  std::string status;
};
using CfarCalibrationCallback = std::function<void(const CfarCalibrationProgress&)>;

struct CfarCalibrationResult {
  CfarCalibrationPlan plan;
  std::vector<double> alpha_lut;
  std::vector<double> geometry_alphas;
  std::string started_at;
  std::string finished_at;
  double elapsed_seconds{0.0};
};

// Confidence is coverage of relative Pfa error per geometry, not alpha error
// or simultaneous coverage across a table. Pure planning: no Monte Carlo draws.
// trials=0 derives an accuracy budget; positive trials fixes the budget and reports actual
// coverage.
CfarCalibrationBudget cfar_calibration_budget(double pfa, double relative_error = 0.10,
                                              double confidence = 0.95, std::uint64_t trials = 0);
// Benchmark a bounded representative batch when Monte Carlo is required.
CfarCalibrationPlan plan_cfar_calibration(const CFARParams& params,
                                          const std::vector<std::int64_t>& input_shape);
CfarCalibrationResult calibrate_cfar(const CFARParams& params,
                                     const std::vector<std::int64_t>& input_shape,
                                     CfarCalibrationCallback progress = {},
                                     std::stop_token stop = {});

namespace detail {
// Host-only geometry/budget preparation shared by runtime, offline and tests.
CfarCalibrationPlan prepare_cfar_calibration(const CFARParams& params,
                                             const std::vector<std::int64_t>& input_shape);
CfarCalibrationResult execute_cfar_calibration(CfarCalibrationPlan plan,
                                               CfarCalibrationCallback progress = {},
                                               std::stop_token stop = {});
// GPU engine: no CPU sampling fallback. benchmark=true performs one bounded batch.
double calibrate_cfar_geometry_gpu(const CfarCalibrationPlan& plan,
                                   const CfarCalibrationGeometry& geometry,
                                   const std::function<void(std::uint64_t)>& batch_done,
                                   std::stop_token stop, bool benchmark = false,
                                   std::uint32_t batch_size = 65536);
}  // namespace detail
}  // namespace holoscan::radar
